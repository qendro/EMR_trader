"""Freeze once from recorded CPU and gating-card measurements, excluding 3060."""
import argparse
import json
import math
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('cpu')
p.add_argument('gating')
p.add_argument('--output',required=True)
p.add_argument('--id',default='h1-fanin-coefficient-v1')
args=p.parse_args()
a,b=(json.loads(Path(path).read_text()) for path in (args.cpu,args.gating))
assert a['hardware']['backend']=='cpu'
assert b['hardware']['gpu'][1]=='NVIDIA GeForce RTX 3070'
assert a['status']==b['status']=='calibration-measured'
assert a['fixture_sha256']==b['fixture_sha256'] and a['script_sha256']==b['script_sha256']
assert a['provenance']==b['provenance']
maximum=max(a['maximum_error_over_uS'],b['maximum_error_over_uS'])
value={'schema':1,'id':args.id,'precision':'float32-HIGHEST',
       'fixture_sha256':a['fixture_sha256'],'calibration_script_sha256':a['script_sha256'],
       'coefficient':max(1,math.ceil(2*maximum)),'observed_maximum_error_over_uS':maximum,
       'rule':'max(1,ceil(2*max(error/(2^-24*S)))) from CPU and RTX3070 only',
       'provenance':a['provenance'],'calibration_hardware':[a['hardware'],b['hardware']],
       'second_card':'RTX3060 validation only; cannot change coefficient'}
target=Path(args.output)
if target.exists() and json.loads(target.read_text())!=value:
    raise ValueError('Coefficient already frozen; a revision requires a dated decision and G1 reruns')
target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'coefficient':value['coefficient'],'fixture_sha256':value['fixture_sha256']}))
