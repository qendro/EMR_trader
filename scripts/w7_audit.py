"""Audit a single-process W7 stability run without inferring a pass from length."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(run_dir):
    directory=Path(run_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    history=json.loads((directory/'history.json').read_text())
    generations=len(history)
    warmup=min(50,max(20,generations//10))
    signatures=[row['rollout_signatures'] for row in history]
    rss=np.array([row['rss_bytes'] for row in history[warmup:]],np.float64)
    first=np.median(rss[:max(1,len(rss)//5)]) if len(rss) else np.nan
    last=np.median(rss[-max(1,len(rss)//5):]) if len(rss) else np.nan
    growth=float(last/first-1) if first>0 else float('inf')
    device=np.array([row['device_memory']['bytes_in_use']
                     for row in history[warmup:] if row['device_memory'] is not None],np.float64)
    device_first=np.median(device[:max(1,len(device)//5)]) if len(device) else np.nan
    device_last=np.median(device[-max(1,len(device)//5):]) if len(device) else np.nan
    device_growth=float(device_last/device_first-1) if device_first>0 else float('inf')
    # Final-generation validation/held-out shapes may compile separately.
    fixed=bool(len(set(signatures[warmup:-1]))<=1)
    no_swap=summary['memory']['cgroup_swap_bytes']==0
    return {'manifest':manifest['config']['id'],'generations':generations,
        'wall_seconds':summary['wall_seconds'],'warmup_generations':warmup,
        'rollout_signatures_at_warmup':signatures[warmup] if generations>warmup else None,
        'rollout_signatures_at_last_training_generation':signatures[-2] if generations>1 else None,
        'rollout_signatures_after_final_selection':signatures[-1] if generations else None,
        'new_training_signatures_after_warmup':not fixed,
        'warm_rss_first_quintile_median':int(first) if np.isfinite(first) else None,
        'warm_rss_last_quintile_median':int(last) if np.isfinite(last) else None,
        'rss_growth_fraction':growth if np.isfinite(growth) else None,
        'device_live_first_quintile_median':int(device_first) if np.isfinite(device_first) else None,
        'device_live_last_quintile_median':int(device_last) if np.isfinite(device_last) else None,
        'device_live_growth_fraction':device_growth if np.isfinite(device_growth) else None,
        'cgroup_swap_bytes':summary['memory']['cgroup_swap_bytes'],
        'passes':bool(generations>=100 and summary['wall_seconds']>=900 and
                      fixed and growth<.10 and device_growth<.10 and no_swap)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(args.run),indent=2)+'\n')
