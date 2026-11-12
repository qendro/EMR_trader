"""Audit matched full-generation CPU/GPU M4 G5 probes."""
import argparse
import json
from pathlib import Path
import statistics


def audit(root):
    root=Path(root)
    rows={}
    for profile in ('cpu','rtx3070','rtx3060'):
        rows[profile]=[json.loads((root/f'm4-g5-{profile}-rep{rep}'/'pilot.json').read_text())
                       for rep in (1,2,3)]
    all_rows=[row for runs in rows.values() for row in runs]
    if len({row['config_sha256'] for row in all_rows})!=1 or \
            len({json.dumps(row['provenance'],sort_keys=True) for row in all_rows})!=1:
        raise ValueError('M4 G5 source or manifest mismatch')
    if any(not row['passes'] for row in all_rows):
        raise ValueError('M4 G5 probe exceeded its budget')
    for profile,runs in rows.items():
        expected='cpu' if profile=='cpu' else 'gpu'
        if any(row['hardware']['backend']!=expected for row in runs):
            raise ValueError('M4 G5 backend mismatch')
        if profile!='cpu' and any(profile[3:] not in row['hardware']['gpu'][1]
                                  for row in runs):
            raise ValueError('M4 G5 GPU identity mismatch')
    medians={name:float(statistics.median(row['warm_generation_seconds'] for row in runs))
             for name,runs in rows.items()}
    speedup={name:medians['cpu']/medians[name] for name in ('rtx3070','rtx3060')}
    return {'config_sha256':all_rows[0]['config_sha256'],
        'engine_sha256':all_rows[0]['provenance']['engine_sha256'],
        'warm_generation_replicates_seconds':{
            name:[row['warm_generation_seconds'] for row in runs]
            for name,runs in rows.items()},
        'median_seconds':medians,'gpu_speedup_over_cpu':speedup,
        'passes':{name:ratio>=1.2 for name,ratio in speedup.items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(args.root),indent=2)+'\n')
