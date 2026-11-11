"""Count low-level backend compilations over 100 post-warm H2 generations."""
import argparse
import hashlib
import json
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
from jax._src import compiler
from neuroevo.artifacts import write_json
from neuroevo.runner import load_config, run, provenance


parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/m2-g2-fixed-memory-v1.json')
parser.add_argument('--output',required=True)
args=parser.parse_args()
config=load_config(args.config)
assert config['id']=='m2-g2-fixed-memory-v1' and config['generations']==102
original=compiler.backend_compile
compile_calls=0


def counted_compile(*a,**kw):
    global compile_calls
    compile_calls+=1
    return original(*a,**kw)


compiler.backend_compile=counted_compile
from neuroevo import runner
original_write=runner.write_json
original_atomic=runner.atomic_genes
selection_compile_start=None


def counted_atomic(path,archive):
    global selection_compile_start
    if Path(path).name=='validation-finalists-genes.npz':
        selection_compile_start=compile_calls
    return original_atomic(path,archive)


def counted_write(path,value):
    if Path(path).name=='history.json' and value and isinstance(value,list):
        value[-1]['backend_compile_calls_total']=compile_calls
    return original_write(path,value)


runner.write_json=counted_write
runner.atomic_genes=counted_atomic
root=Path(args.output)
summary,history=run(config,900,root,hardware)
counts=[row['backend_compile_calls_total'] for row in history]
assert len(counts)==102
assert selection_compile_start is not None
post_warm=selection_compile_start-counts[1]
terminal_selection=counts[-1]-selection_compile_start
result={'schema':1,'status':'pass' if post_warm==0 else 'fail',
        'label':'M2-G2-100-post-warm-generations',
        'hardware':hardware,'provenance':provenance(),
        'config_sha256':summary['config_sha256'],
        'seed':900,'generations':102,'warm_generations':2,
        'post_warm_generations':100,'compiles_at_warm_end':counts[1],
        'compiles_before_final_selection':selection_compile_start,
        'compiles_at_final':counts[-1],'post_warm_backend_compiles':post_warm,
        'terminal_selection_backend_compiles':terminal_selection,
        'rollout_signatures':[row['rollout_signatures'] for row in history],
        'tell_signatures':[row['tell_signatures'] for row in history],
        'observer_signatures':[row['observer_signatures'] for row in history],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_json(root/'g2-result.json',result)
print(json.dumps({'status':result['status'],'post_warm_backend_compiles':post_warm,
                  'terminal_selection_backend_compiles':terminal_selection,
                  'profile':hardware['backend']}))
if post_warm:
    raise AssertionError('Unexpected post-warm backend compilations')
