"""Version the B0 CartPole cap after its first matching G9 pilot missed."""
import argparse
import hashlib
import json
from pathlib import Path


parser=argparse.ArgumentParser()
parser.add_argument('--pilot',default='/results/m2-elapsed-v1/b0-cartpole/pilot.json')
parser.add_argument('--configs',default='/configs-out')
parser.add_argument('--report',default='/reports/m2-elapsed-revision.json')
args=parser.parse_args()
pilot_bytes=Path(args.pilot).read_bytes()
pilot=json.loads(pilot_bytes)
old=json.loads(Path('configs/m2-elapsed-b0-cartpole-acceptance-v1.json').read_text())
assert not pilot['passes'] and pilot['config_sha256']==hashlib.sha256(json.dumps(
    old,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert old['generations']==47900 and 45000<pilot['allowable_generations']<47900
new={**old,'id':'m2-elapsed-b0-cartpole-acceptance-v2','generations':45000,
     'revision':{'reason':'first matching G9 pilot missed 47900-generation cap after warm-time variation',
                 'failed_pilot_sha256':hashlib.sha256(pilot_bytes).hexdigest(),
                 'failed_allowable_generations':pilot['allowable_generations'],
                 'revised_cap':45000}}
path=Path(args.configs)/'m2-elapsed-b0-cartpole-acceptance-v2.json'
path.write_text(json.dumps(new,indent=2)+'\n')
target=Path(args.report)
target.write_text(json.dumps({'schema':1,'status':'revised-before-acceptance',
    'old_config_sha256':pilot['config_sha256'],
    'new_config_sha256':hashlib.sha256(json.dumps(
        new,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
    'old_cap':47900,'new_cap':45000,
    'failed_pilot_sha256':hashlib.sha256(pilot_bytes).hexdigest()},indent=2)+'\n')
print(json.dumps({'status':'revised','old_cap':47900,'new_cap':45000}))
