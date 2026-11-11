"""Matched fixed-CPPN CartPole generation floor for the M2 G4 controller check."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from neuroevo.adapters.cppn import CPPNQuery
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.artifacts import canonical, export, save_checkpoint, write_json
from neuroevo.environments import cartpole
from neuroevo.evaluation import evaluate_population, make_cartpole_rollout
from neuroevo.fixed import FixedDecoder
from neuroevo.randomness import key
from neuroevo.resources import memory_record
from neuroevo.runner import atomic_genes, load_config, provenance


parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/m2-g4-fixed-cartpole-v1.json')
parser.add_argument('--output',required=True)
args=parser.parse_args()
config=load_config(args.config)
assert config['id']=='m2-g4-fixed-cartpole-v1' and config['generations']==12
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
source=provenance()
adapter=TensorNEATAdapter(config['adapter'])
state=adapter.initialize(900)
decoder=FixedDecoder(CPPNQuery(adapter),config['decoder'])
env,params=cartpole()
evaluator=make_cartpole_rollout(env,params,config['horizon'],'layered')
shared=jnp.zeros((config['horizon'],4),jnp.float32)
write_json(root/'manifest.json',{'config':config,'seed':900,'hardware':hardware,
                                  'provenance':source,'config_sha256':hashlib.sha256(canonical(config)).hexdigest()})
rows=[]
start_run=time.perf_counter()
for generation in range(config['generations']):
    start=time.perf_counter()
    batch=adapter.ask(state,'seed-900')
    plans=[]
    stats=[]
    for nodes,conns in zip(batch.nodes,batch.conns):
        plans.append(decoder.decode(nodes,conns))
        stats.append(decoder.last_stats)
    decode_seconds=time.perf_counter()-start
    eval_start=time.perf_counter()
    keys=jnp.stack([key(900,'train',generation,e) for e in range(config['episodes'])])
    values=evaluate_population(plans,evaluator,keys,shared,config['microbatch'])
    scores=values[0].mean(axis=1)
    valid=np.asarray(values[4]).all(axis=1)&np.isfinite(scores)
    scores=np.where(valid,scores,-np.inf)
    assert valid.all()
    rollout_seconds=time.perf_counter()-eval_start
    winner=int(np.argmax(scores))
    archive={'id':batch.ids[winner],'score':float(scores[winner]),
             'nodes':batch.nodes[winner],'conns':batch.conns[winner]}
    export(root/'evaluated-winner.npz',plans[winner],'layered',
           {'candidate_id':archive['id'],'training_score':archive['score'],
            'action':config['action'],'provenance':source,**decoder.interpretation()})
    atomic_genes(root/'evaluated-winner-genes.npz',archive)
    tell_start=time.perf_counter()
    state,removed=adapter.tell(state,batch,batch.ids,scores)
    evolve_seconds=time.perf_counter()-tell_start
    row={'generation':generation,'best':float(scores[valid].max()),
         'mean':float(scores[valid].mean()),'valid':int(valid.sum()),
         'species':int(np.isfinite(np.asarray(state.species.species_keys)).sum()),
         'stagnation_removed':removed,
         'nodes_max':int((~np.isnan(batch.nodes[:,:,0])).sum(axis=1).max()),
         'edges_max':int((~np.isnan(batch.conns[:,:,0])).sum(axis=1).max()),
         'decode_plan_seconds':decode_seconds,'rollout_seconds':rollout_seconds,
         'evolve_seconds':evolve_seconds,'selection_seconds':0.,
         'rollout_signatures':evaluator._cache_size(),
         'tell_signatures':adapter._tell._cache_size(),
         'observer_signatures':adapter._observe._cache_size(),
         'rss_bytes':psutil.Process().memory_info().rss,
         'device_memory':jax.devices()[0].memory_stats(),
         'phenotype_nodes_max':max(s['phenotype_nodes'] for s in stats),
         'phenotype_edges_max':max(s['expressed_edges'] for s in stats),
         'cppn_queries_total':sum(s['queries'] for s in stats),
         'phenotype_host_bytes_total':sum(s['host_array_bytes'] for s in stats),
         'decode_errors':{}}
    rows.append(row)
    row['total_seconds']=time.perf_counter()-start
    if (generation+1)%config['checkpoint_every']==0 or generation==config['generations']-1:
        save_checkpoint(root/'checkpoint.bin',state,{**config,'run_seed':900},source,
                        rows,archive,time.perf_counter()-start_run)
    row['total_seconds']=time.perf_counter()-start
    write_json(root/'history.json',rows)
    row['total_seconds']=time.perf_counter()-start
    print(json.dumps({'generation':generation,'best':row['best'],
                      'total_seconds':row['total_seconds']}),flush=True)
write_json(root/'result.json',{'schema':1,'label':'M2-G4-matched-minimal-driver',
    'hardware':hardware,'provenance':source,
    'config_sha256':hashlib.sha256(canonical(config)).hexdigest(),
    'seed':900,'warm_window_generations':list(range(2,11)),
    'warm_window_median_seconds':float(np.median([r['total_seconds'] for r in rows[2:11]])),
    'memory':memory_record(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
