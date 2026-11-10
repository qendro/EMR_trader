"""Check the exported evidence index and current package identity in Docker."""
import argparse
import hashlib
import json
from pathlib import Path
import neuroevo

p=argparse.ArgumentParser()
p.add_argument('--evidence',default='/evidence')
p.add_argument('--status',default='/reports/status.json')
p.add_argument('--milestone',choices=['m1','h1'],default='m1')
args=p.parse_args()
root=Path(args.evidence)
index=json.loads((root/'index.json').read_text())
for row in index['files']:
    target=root/row['path']
    assert target.stat().st_size==row['bytes'],target
    assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'],target
package=Path(neuroevo.__file__).parent
engine=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(package.rglob('*.py')))).hexdigest()
status=json.loads(Path(args.status).read_text())
for profile in status['profiles'].values():
    recorded=(profile['learning']['engine_sha256'] if args.milestone=='m1' else
              profile['large']['provenance']['engine_sha256'])
    assert recorded==engine
print(json.dumps({'status':'pass','files':len(index['files']),'bytes':sum(r['bytes'] for r in index['files']),
                  'engine_sha256':engine,'complete':status['m1_closed'] if args.milestone=='m1' else status['H0_H1_complete']}))
