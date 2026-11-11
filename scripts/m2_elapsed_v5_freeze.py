"""Freeze true wall-stopped M2 comparison manifests before acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


p = argparse.ArgumentParser()
p.add_argument('--pilots', default='/results/m2-elapsed-v4')
p.add_argument('--configs', default='/configs-out')
p.add_argument('--report', default='/reports/m2-elapsed-v5-freeze.json')
args = p.parse_args()
cells = (
    ('b0', 'cartpole', 'v4', 'v5'),
    ('b0', 'memory', 'v3', 'v4'),
    ('fixed', 'cartpole', 'v2', 'v3'),
    ('fixed', 'memory', 'v2', 'v3'),
)
rows = []
for method, task, old_version, version in cells:
    old = json.loads(Path(f'configs/m2-elapsed-{method}-{task}-acceptance-{old_version}.json').read_text())
    pilot_bytes = (Path(args.pilots)/f'{method}-{task}'/'pilot.json').read_bytes()
    pilot = json.loads(pilot_bytes)
    assert pilot['passes'] and pilot['config_sha256'] == hashlib.sha256(canonical(old)).hexdigest()
    new = {**old,
        'id': f'm2-elapsed-{method}-{task}-acceptance-{version}',
        'generations': 100000,
        'comparison': 'wall stop at 6480 seconds, final selection within 7200-second G9 budget',
        'elapsed_revision': {**old['elapsed_revision'],
            'reason': 'use the same wall cutoff for every method; generation cap is a safety ceiling',
            'prior_manifest': old['id'],
            'prior_matching_pilot_sha256': hashlib.sha256(pilot_bytes).hexdigest(),
        },
    }
    target = Path(args.configs)/f'm2-elapsed-{method}-{task}-acceptance-{version}.json'
    target.write_text(json.dumps(new, indent=2)+'\n')
    rows.append({'method': method, 'task': task, 'version': version,
        'generation_safety_ceiling': 100000, 'elapsed_stop_seconds': 6480,
        'config_sha256': hashlib.sha256(canonical(new)).hexdigest(),
        'prior_matching_pilot_sha256': hashlib.sha256(pilot_bytes).hexdigest()})
report = Path(args.report)
report.write_text(json.dumps({'schema': 1, 'status': 'frozen-before-v5-acceptance',
    'rows': rows}, indent=2)+'\n')
print(json.dumps({'status': 'frozen', 'rows': rows}))
