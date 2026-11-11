"""Lower the two B0 caps after matching v3 pilot misses, before acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


p = argparse.ArgumentParser()
p.add_argument('--pilots', default='/results/m2-elapsed-v3')
p.add_argument('--configs', default='/configs-out')
p.add_argument('--report', default='/reports/m2-elapsed-v4-revision.json')
args = p.parse_args()
rows = []
for task, old_version, version, cap in (
        ('cartpole', 'v3', 'v4', 48000), ('memory', 'v2', 'v3', 36000)):
    old = json.loads(Path(f'configs/m2-elapsed-b0-{task}-acceptance-{old_version}.json').read_text())
    pilot_bytes = (Path(args.pilots)/f'b0-{task}'/'pilot.json').read_bytes()
    pilot = json.loads(pilot_bytes)
    assert not pilot['passes'] and pilot['config_sha256'] == hashlib.sha256(canonical(old)).hexdigest()
    assert cap < pilot['allowable_generations'] < old['generations']
    new = {**old, 'id': f'm2-elapsed-b0-{task}-acceptance-{version}',
        'generations': cap,
        'elapsed_revision': {**old['elapsed_revision'],
            'matching_pilot_failure_sha256': hashlib.sha256(pilot_bytes).hexdigest(),
            'pilot_allowable_generations': pilot['allowable_generations'],
            'revised_generation_cap': cap}}
    target = Path(args.configs)/f'm2-elapsed-b0-{task}-acceptance-{version}.json'
    target.write_text(json.dumps(new, indent=2)+'\n')
    rows.append({'task': task, 'old_cap': old['generations'], 'new_cap': cap,
        'failed_pilot_sha256': hashlib.sha256(pilot_bytes).hexdigest(),
        'new_config_sha256': hashlib.sha256(canonical(new)).hexdigest()})
report = Path(args.report)
report.write_text(json.dumps({'schema': 1, 'status': 'revised-before-v4-acceptance',
    'rows': rows}, indent=2)+'\n')
print(json.dumps({'status': 'revised', 'rows': rows}))
