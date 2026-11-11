"""Measure decoder query slab sizes on one frozen 1024-node CPPN phenotype."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from neuroevo.backend import initialize
hardware=initialize()
import numpy as np
from neuroevo.adapters.cppn import CPPNQuery
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.artifacts import write_json
from neuroevo.fixed import FixedDecoder
from neuroevo.runner import provenance


parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--seed',type=int,default=9100)
args=parser.parse_args()
config=json.loads(Path('configs/m2-g5-fixed-1024-v1.json').read_text())
adapter=TensorNEATAdapter(config['adapter'])
batch=adapter.ask(adapter.initialize(args.seed),'slab-probe')
query=CPPNQuery(adapter)
genes=(batch.nodes[0],batch.conns[0])
reference=None
rows=[]
for slab in (4096,16384,65536):
    decoder=FixedDecoder(query,{**config['decoder'],'query_slab':slab})
    start=time.perf_counter()
    first=decoder.decode(*genes)
    cold=time.perf_counter()-start
    start=time.perf_counter()
    second=decoder.decode(*genes)
    warm=time.perf_counter()-start
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
    if reference is None: reference=first
    else:
        for a,b in zip(reference,second): np.testing.assert_array_equal(a,b)
    rows.append({'query_slab':slab,'first_seconds':cold,'warm_seconds':warm,
                 'queries':decoder.last_stats['queries'],
                 'phenotype_sha256':hashlib.sha256(b''.join(v.tobytes() for v in second)).hexdigest()})
write_json(args.output,{'schema':1,'status':'pass','seed':args.seed,'hardware':hardware,
                        'provenance':provenance(),'rows':rows,
                        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
print(json.dumps({'status':'pass','profile':hardware['backend'],
                  'rows':[(r['query_slab'],r['warm_seconds']) for r in rows]}))
