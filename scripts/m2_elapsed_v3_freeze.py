"""Freeze revised elapsed-stop manifests before new acceptance seeds."""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


p = argparse.ArgumentParser()
p.add_argument('--pilots', default='/results/m2-elapsed-v3-calibration')
p.add_argument('--failed', default='/results/m2-elapsed-v2/fixed-cartpole/seed-0')
p.add_argument('--configs', default='/configs-out')
p.add_argument('--report', default='/reports/m2-elapsed-v3-freeze.json')
args = p.parse_args()
failed = Path(args.failed)
failure = json.loads((failed/'budget-failure.json').read_text())
assert failure['status'] == 'over-budget' and failure['generation'] == 4905
failure_hash = hashlib.sha256((failed/'budget-failure.json').read_bytes()).hexdigest()
cells = (
    ('b0', 'cartpole', 'v2', 'v3', 53000),
    ('b0', 'memory', 'v1', 'v2', 39000),
    ('fixed', 'cartpole', 'v1', 'v2', 5700),
    ('fixed', 'memory', 'v1', 'v2', 5200),
)
rows = []
for method, task, old_version, version, cap in cells:
    old = json.loads(Path(f'configs/m2-elapsed-{method}-{task}-acceptance-{old_version}.json').read_text())
    pilot_path = Path(args.pilots)/f'{method}-{task}'/'pilot.json'
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    assert pilot['passes'] and pilot['pilot_seed'] == 900
    assert cap <= pilot['allowable_generations']
    revised = {**old,
        'id': f'm2-elapsed-{method}-{task}-acceptance-{version}',
        'generations': cap,
        'elapsed_stop_seconds': 6480,
        'comparison': 'quality at 6480-second elapsed stop within 7200-second G9 research budget',
        'calibration': {
            'pilot_sha256': hashlib.sha256(pilot_bytes).hexdigest(),
            'pilot_image_digest': pilot['hardware']['image_digest'],
            'warm_generation_seconds': pilot['warm_generation_seconds'],
            'fixed_overhead_seconds': pilot['fixed_overhead_seconds'],
            'allowable_generations': pilot['allowable_generations'],
            'cap_fraction_of_allowable': cap/pilot['allowable_generations'],
            'tuning_trials': 0,
        },
        'elapsed_revision': {
            'reason': 'fixed CartPole seed 0 exceeded 7200 seconds after topology growth; end at a common elapsed cutoff before final selection',
            'failed_run_budget_record_sha256': failure_hash,
            'prior_manifest': old['id'],
        },
    }
    target = Path(args.configs)/f'm2-elapsed-{method}-{task}-acceptance-{version}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(revised, indent=2)+'\n')
    rows.append({'method': method, 'task': task, 'version': version,
                 'generation_cap': cap, 'elapsed_stop_seconds': 6480,
                 'allowable_generations': pilot['allowable_generations'],
                 'config_sha256': hashlib.sha256(canonical(revised)).hexdigest()})
report = Path(args.report)
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps({'schema': 1, 'status': 'frozen-before-v3-acceptance',
    'failed_run_budget_record_sha256': failure_hash, 'rows': rows}, indent=2)+'\n')
print(json.dumps({'status': 'frozen', 'rows': rows}))
