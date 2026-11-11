"""Verify the three-profile M2 G2 compilation-stability runs."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


parser=argparse.ArgumentParser()
parser.add_argument('--source',default='/results/m2-g2-v2')
parser.add_argument('--output',default='/reports/m2-g2-status.json')
args=parser.parse_args()
root=Path(args.source)
config=read(Path('configs/m2-g2-fixed-memory-v1.json'))
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
profiles={}
engine=None
image=None
for profile in ('cpu','rtx3070','rtx3060'):
    folder=root/profile
    manifest=read(folder/'manifest.json')
    result=read(folder/'g2-result.json')
    history=read(folder/'history.json')
    assert manifest['config']==config and manifest['config_sha256']==config_hash
    assert result['config_sha256']==config_hash
    assert result['hardware']==manifest['hardware'] and result['provenance']==manifest['provenance']
    assert result['status']=='pass' and result['post_warm_backend_compiles']==0
    assert result['generations']==102 and result['post_warm_generations']==100
    assert [r['generation'] for r in history]==list(range(102))
    counts=[r['backend_compile_calls_total'] for r in history]
    assert all(count==counts[1] for count in counts[2:101])
    assert result['compiles_at_warm_end']==counts[1]
    assert result['compiles_before_final_selection']==counts[1]
    assert result['terminal_selection_backend_compiles']==counts[-1]-counts[1]
    assert len(set(result['rollout_signatures'][1:]))==1
    assert len(set(result['tell_signatures'][1:]))==1
    assert len(set(result['observer_signatures'][1:]))==1
    engine=engine or result['provenance']['engine_sha256']
    image=image or result['hardware']['image_digest']
    assert engine==result['provenance']['engine_sha256']
    assert image==result['hardware']['image_digest']
    profiles[profile]={'post_warm_backend_compiles':0,
                       'terminal_selection_backend_compiles':result['terminal_selection_backend_compiles'],
                       'warm_signatures':{k:result[k][1] for k in ('rollout_signatures','tell_signatures','observer_signatures')},
                       'memory':read(folder/'summary.json')['memory']}
status={'schema':1,'status':'pass','config_sha256':config_hash,
        'engine_sha256':engine,'image_digest':image,'profiles':profiles}
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':'pass','post_warm_backend_compiles':{k:0 for k in profiles}}))
