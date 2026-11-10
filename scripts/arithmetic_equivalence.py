"""Prove B0 uses the exact qualified H1 runtime, evaluator and dependency lane."""
import argparse
import hashlib
import json
from pathlib import Path
import neuroevo

p=argparse.ArgumentParser()
p.add_argument('--previous',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
old=Path(args.previous)
package=Path(neuroevo.__file__).parent
files=[]
for name in ('runtime.py','substrate.py','contracts.py','planning.py','reference.py',
             'evaluation.py','environments.py','statistics.py','randomness.py'):
    current=(package/name).read_bytes()
    assert current==(old/'src/neuroevo'/name).read_bytes(),name
    files.append({'path':name,'sha256':hashlib.sha256(current).hexdigest()})
for name in ('requirements.txt','compatibility.txt','sources.json'):
    assert Path('/app/locks',name).read_bytes()==(old/'locks'/name).read_bytes(),name
engine=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(package.rglob('*.py')))).hexdigest()
value={'status':'pass','current_engine_sha256':engine,
       'previous_engine_sha256':json.loads((old/'source.json').read_text())['engine_sha256'],
       'runtime_evaluator_dependencies_identical':True,'files':files,
       'interpretation':'B0 adds evolution/controller support; qualified H1 arithmetic and evaluator files unchanged; no new large-size qualification claim'}
Path(args.output).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':'pass','engine_sha256':engine,'files':len(files)}))
