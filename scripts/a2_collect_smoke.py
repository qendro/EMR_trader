"""Copy only the bounded A2 development runs, including the retained cap failure."""
import hashlib
import json
from pathlib import Path
import shutil


source=Path('/results')
target=Path('benchmarks/results/evidence/a2-development-v1')
names=(
    'a2-memory-smoke-v1',
    'a2-memory-cpu-final-v2',
    'a2-memory-gpu-smoke-v2',
    'a2-cartpole-cpu-final-v1',
    'a2-cartpole-gpu-smoke-v1',
    'a2-memory-full-v3',
    'a2-memory-resume-v2',
    'a2-memory-acceptance-v1-rtx3070-image',
    'a2-cartpole-acceptance-v1-rtx3070-image',
)
rows=[]
for name in names:
    root=source/name
    if not root.is_dir():
        raise FileNotFoundError(root)
    for original in sorted(root.rglob('*')):
        if not original.is_file() or original.suffix not in ('.json','.npz','.bin'):
            continue
        relative=original.relative_to(source)
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        raw=original.read_bytes()
        rows.append({'path':str(relative).replace('\\','/'),
                     'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
target.mkdir(parents=True,exist_ok=True)
(target/'index.json').write_text(json.dumps({'schema':1,'files':rows},indent=2)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
