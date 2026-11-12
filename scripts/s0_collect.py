"""Copy bounded S0 CPU development smokes with byte hashes."""
import hashlib
import json
from pathlib import Path
import shutil

source=Path('/results')
target=Path('benchmarks/results/evidence/s0-development-v1')
rows=[]
for name in ('s0-memory-smoke-v1','s0-cartpole-smoke-v1'):
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
