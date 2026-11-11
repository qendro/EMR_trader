"""Fresh-process integrated M2 G5 generation at the named 1024-node cell."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.adapters.cppn import CPPNQuery, query_scores
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.artifacts import canonical, write_json
from neuroevo.environments import cartpole
from neuroevo.evaluation import evaluate_population, make_cartpole_rollout
from neuroevo.fixed import FixedDecoder
from neuroevo.randomness import key
from neuroevo.resources import memory_record
from neuroevo.runner import provenance


parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/m2-g5-fixed-1024-v1.json')
parser.add_argument('--output',required=True)
parser.add_argument('--seed',type=int,required=True)
args=parser.parse_args()
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
config=json.loads(Path(args.config).read_text())
assert config['schema']==1 and config['id']=='m2-g5-fixed-1024-v1'
assert hashlib.sha256(Path(config['source_config']).read_bytes()).hexdigest()==config['source_config_sha256']
assert args.seed in config['seeds'] and hardware['backend'] in ('cpu','gpu')
work=config['workload']
assert (work['population'],work['episodes'],work['horizon'],work['total_nodes'],work['profile'])==(128,8,500,1024,'layered')
assert work['microbatch']==16 and config['decoder']['recurrent']
assert sum(config['decoder'][k] for k in ('inputs','hidden','outputs'))==1024
source=provenance()
write_json(root/'manifest.json',{'config':config,'seed':args.seed,'hardware':hardware,
                                 'provenance':source,'config_sha256':hashlib.sha256(canonical(config)).hexdigest()})
write_json(root/'runtime.json',{'jit_enabled':not jax.config.jax_disable_jit,
    'matmul_precision':jax.config.jax_default_matmul_precision,
    'x64':jax.config.jax_enable_x64,
    'compilation_cache_dir':os.environ.get('JAX_COMPILATION_CACHE_DIR'),
    'measurement':'steady second generation; first generation warms the same signatures'})

adapter=TensorNEATAdapter(config['adapter'])
state=adapter.initialize(args.seed)
decoder=FixedDecoder(CPPNQuery(adapter),config['decoder'])
env,params=cartpole()
evaluator=make_cartpole_rollout(env,params,work['horizon'],'layered')
shared=jnp.zeros((work['horizon'],4),jnp.float32)
rows=[]
try:
    for generation in range(2):
        start=time.perf_counter()
        batch=adapter.ask(state,f'g5-{args.seed}')
        plans=[]
        recurrent_edges=[]
        for nodes,conns in zip(batch.nodes,batch.conns):
            phenotype=decoder.decode(nodes,conns)
            plans.append(phenotype)
            recurrent_edges.append(int(phenotype.mask_delay.sum()))
        decode_seconds=time.perf_counter()-start
        rollout_start=time.perf_counter()
        keys=jnp.stack([key(args.seed,'train',generation,e) for e in range(work['episodes'])])
        values=evaluate_population(plans,evaluator,keys,shared,work['microbatch'])
        scores=values[0].mean(axis=1)
        valid=np.asarray(values[4]).all(axis=1)&np.isfinite(scores)
        rollout_seconds=time.perf_counter()-rollout_start
        if not np.any(valid):
            raise ValueError('All 128 G5 candidates invalid')
        scores=np.where(valid,scores,-np.inf)
        tell_start=time.perf_counter()
        state,removed=adapter.tell(state,batch,batch.ids,scores)
        tell_seconds=time.perf_counter()-tell_start
        row={'generation':generation,'total_seconds':time.perf_counter()-start,
             'decode_plan_seconds':decode_seconds,'rollout_reduce_seconds':rollout_seconds,
             'tell_seconds':tell_seconds,'valid_candidates':int(valid.sum()),
             'best_return':float(np.max(scores[valid])),
             'median_return':float(np.median(scores[valid])),
             'recurrent_edges_min':min(recurrent_edges),
             'recurrent_edges_median':float(np.median(recurrent_edges)),
             'recurrent_edges_max':max(recurrent_edges),
             'cppn_queries_per_candidate':decoder.last_stats['queries'],
             'stagnation_removed':removed,
             'rollout_signatures':evaluator._cache_size(),
             'query_signatures':query_scores._cache_size(),
             'tell_signatures':adapter._tell._cache_size(),
             'observer_signatures':adapter._observe._cache_size(),
             'device_memory':jax.devices()[0].memory_stats()}
        rows.append(row)
        write_json(root/'history.json',rows)
        print(json.dumps({'generation':generation,'total_seconds':row['total_seconds'],
                          'decode_plan_seconds':decode_seconds,'rollout_reduce_seconds':rollout_seconds,
                          'valid_candidates':row['valid_candidates']}),flush=True)
        if row['total_seconds']>7200:
            raise RuntimeError('G5 generation exceeded 2-hour research budget')
    assert rows[1]['recurrent_edges_max']>0
    assert all(row['rollout_signatures']==rows[0]['rollout_signatures'] and
               row['query_signatures']==rows[0]['query_signatures'] for row in rows)
    result={'schema':1,'status':'measured','label':'M2-G5-integrated-one-generation',
            'hardware':hardware,'provenance':source,
            'config_sha256':hashlib.sha256(canonical(config)).hexdigest(),
            'seed':args.seed,'warm_generation':rows[0],
            'measured_generation':rows[1],
            'candidate_episode_evaluations':128*8,
            'memory':memory_record(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(root/'result.json',result)
except Exception as error:
    write_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),
                                    'history':rows,'hardware':hardware,'provenance':source})
    raise
