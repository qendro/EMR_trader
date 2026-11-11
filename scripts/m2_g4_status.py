"""Verify three matched M2 fixed-CPPN controller/driver pairs."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


parser=argparse.ArgumentParser()
parser.add_argument('--source',default='/results/m2-g4-v1')
parser.add_argument('--output',default='/reports/m2-g4-status.json')
parser.add_argument('--profile',default='rtx3070')
args=parser.parse_args()
root=Path(args.source)
config=read(Path('configs/m2-g4-fixed-cartpole-v1.json'))
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
pairs=[]
for trial in range(3):
    controller=root/f'{args.profile}-controller-{trial}'/'seed-900'
    driver=root/f'{args.profile}-driver-{trial}'
    cm,dm=read(controller/'manifest.json'),read(driver/'manifest.json')
    ch,dh=read(controller/'history.json'),read(driver/'history.json')
    dr=read(driver/'result.json')
    assert cm['config']==dm['config']==config
    assert cm['config_sha256']==dm['config_sha256']==dr['config_sha256']==config_hash
    assert cm['hardware']==dm['hardware']==dr['hardware']
    assert cm['provenance']==dm['provenance']==dr['provenance']
    assert [r['generation'] for r in ch]==[r['generation'] for r in dh]==list(range(12))
    for a,b in zip(ch,dh):
        for field in ('best','mean','valid','species','stagnation_removed','nodes_max','edges_max',
                      'phenotype_nodes_max','phenotype_edges_max','cppn_queries_total'):
            assert a[field]==b[field],(trial,a['generation'],field,a[field],b[field])
    controller_window=[r['total_seconds'] for r in ch[2:11]]
    driver_window=[r['total_seconds'] for r in dh[2:11]]
    assert abs(median(driver_window)-dr['warm_window_median_seconds'])<1e-9
    pairs.append({'trial':trial,'controller_median_seconds':median(controller_window),
                  'driver_median_seconds':median(driver_window),
                  'controller_range':[min(controller_window),max(controller_window)],
                  'driver_range':[min(driver_window),max(driver_window)]})
controller=median([p['controller_median_seconds'] for p in pairs])
driver=median([p['driver_median_seconds'] for p in pairs])
ratio=controller/driver
status={'schema':1,'status':'measured','profile':args.profile,
        'config_sha256':config_hash,'pairs':pairs,
        'controller_median_seconds':controller,'driver_median_seconds':driver,
        'controller_over_driver_ratio':ratio,'g4_passes':ratio<=1.2}
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':'pass','g4_passes':status['g4_passes'],
                  'controller_over_driver_ratio':ratio}))
