"""Copy bounded scientific artifacts from the persistent volume; omit caches.

Run in Docker with /results mounted read-only and /evidence bound to the host.
The index records the exact bytes copied, including failures and checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--source',default='/results')
p.add_argument('--output',default='/evidence')
p.add_argument('--telemetry',default=None)
args=p.parse_args()
source,output=Path(args.source),Path(args.output)
roots=[]
for pattern in ('qualification-*','m1-final-*','m1-current-*','m1-resume-*',
                'g2-*','g4-*','q2-v2','q2-complete','q2-native*','validation','diagnostics',
                'development','h1-*','b0-v1','b0-resume-v1','b0-qualification-v1','b0-development',
                'h2-fixed-*','h2-direct-*','h2-visual-*',
                'm2-g2-*','m2-g3-*','m2-g4-*','m2-g5-*','w8-v2'):
    roots.extend(source.glob(pattern))
rows=[]
for root in sorted(set(roots)):
    for original in sorted(root.rglob('*')):
        if not original.is_file() or original.suffix not in ('.json','.jsonl','.xml','.npz','.bin'): continue
        if any('cache' in part for part in original.relative_to(root).parts[:-1]): continue
        relative=original.relative_to(source)
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        rows.append({'path':str(relative),'bytes':original.stat().st_size,
                     'sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
if args.telemetry:
    telemetry=Path(args.telemetry)
    for original in sorted(telemetry.glob('*')):
        if not original.is_file() or original.suffix not in ('.csv','.jsonl'): continue
        relative=Path('m2-g3-telemetry')/original.name
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        rows.append({'path':str(relative),'bytes':original.stat().st_size,
                     'sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
output.mkdir(parents=True,exist_ok=True)
(output/'index.json').write_text(json.dumps({'schema':1,'files':rows},indent=2)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
