"""Matched W3 warm-generation floor for G4 (not a learning experiment).

Same adapter, planner, evaluator, archive, summaries and checkpoint frequency as
the controller. Final validation is excluded from both warm-time samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.runner import load_config,provenance,atomic_genes
from neuroevo.artifacts import export,save_checkpoint,write_json,canonical
from neuroevo.planning import plan
from neuroevo.environments import cartpole
from neuroevo.evaluation import make_cartpole_rollout,evaluate_population
from neuroevo.randomness import key
from neuroevo.resources import memory_record

# Pinned JAX 0.6.1 observation hook: count real backend compilations, including
# helpers below the three top-level jit cache counters. Calls remain unchanged.
from jax._src import compiler
backend_compile=compiler.backend_compile
compile_calls=0
def count_compile(*a,**kw):
    global compile_calls
    compile_calls+=1
    return backend_compile(*a,**kw)
compiler.backend_compile=count_compile

parser=argparse.ArgumentParser()
parser.add_argument("--config",default="configs/m1-acceptance-v1.json")
parser.add_argument("--output",required=True)
parser.add_argument("--seed",type=int,default=900)
parser.add_argument("--generations",type=int,default=12)
args=parser.parse_args()
config=load_config(args.config)
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
cache_dir=os.environ.get('JAX_COMPILATION_CACHE_DIR')
write_json(root/'runtime.json',{'compilation_cache_dir':cache_dir,
    'cache_entries_at_start':len(list(Path(cache_dir).glob('*-cache'))) if cache_dir else 0,
    'jit_enabled':not jax.config.jax_disable_jit,
    'matmul_precision':jax.config.jax_default_matmul_precision,
    'jax_enable_x64':jax.config.jax_enable_x64,
    'compile_counter':'pinned-jax._src.compiler.backend_compile-pass-through'})
source=provenance()
adapter=TensorNEATAdapter(config["adapter"])
state=adapter.initialize(args.seed)
env,params=cartpole()
evaluator=make_cartpole_rollout(env,params,config["horizon"],config["adapter"]["profile"])
shared=jnp.zeros((config["horizon"],4),jnp.float32)
rows=[]
begin=time.perf_counter()
for generation in range(args.generations):
    start=time.perf_counter()
    compile_before=compile_calls
    batch=adapter.ask(state,f"seed-{args.seed}")
    plans=[plan(adapter.decode(n,c),config["adapter"]["nodes"],config["adapter"]["edges"])
           for n,c in zip(batch.nodes,batch.conns)]
    decode_seconds=time.perf_counter()-start
    eval_start=time.perf_counter()
    keys=jnp.stack([key(args.seed,"train",generation,e) for e in range(config["episodes"])])
    result=evaluate_population(plans,evaluator,keys,shared,config["microbatch"])
    scores=result[0].mean(axis=1)
    rollout_seconds=time.perf_counter()-eval_start
    assert np.all(result[4]) and np.isfinite(scores).all()
    winner=int(np.argmax(scores))
    archive={"id":batch.ids[winner],"score":float(scores[winner]),
             "nodes":batch.nodes[winner],"conns":batch.conns[winner]}
    export(root/"evaluated-winner.npz",plans[winner],config["adapter"]["profile"],
           {"candidate_id":archive["id"],"training_score":archive["score"],
            "action":config["action"],"provenance":source})
    atomic_genes(root/"evaluated-winner-genes.npz",archive)
    tell_start=time.perf_counter()
    state,removed=adapter.tell(state,batch,batch.ids,scores)
    evolve_seconds=time.perf_counter()-tell_start
    row={"generation":generation,"best":float(scores.max()),"mean":float(scores.mean()),
         "valid":len(scores),"species":int(np.isfinite(np.asarray(state.species.species_keys)).sum()),
         "stagnation_removed":removed,
         "nodes_max":int((~np.isnan(batch.nodes[:,:,0])).sum(axis=1).max()),
         "edges_max":int((~np.isnan(batch.conns[:,:,0])).sum(axis=1).max()),
         "decode_plan_seconds":decode_seconds,"rollout_seconds":rollout_seconds,
         "evolve_seconds":evolve_seconds,"selection_seconds":0.,"decode_errors":{},
         "rollout_signatures":evaluator._cache_size(),"tell_signatures":adapter._tell._cache_size(),
         "observer_signatures":adapter._observe._cache_size(),
         "rss_bytes":psutil.Process().memory_info().rss,"device_memory":jax.devices()[0].memory_stats()}
    rows.append(row)
    row["total_seconds"]=time.perf_counter()-start
    row['backend_compile_calls']=compile_calls-compile_before
    if (generation+1) % config["checkpoint_every"] == 0 or generation==args.generations-1:
        save_checkpoint(root/"checkpoint.bin",state,{**config,"run_seed":args.seed},source,rows,archive,
                        time.perf_counter()-begin)
    row["total_seconds"]=time.perf_counter()-start
    write_json(root/"history.json",rows)
    print(json.dumps({k:row[k] for k in ("generation","best","mean","nodes_max","total_seconds")}))
    row["total_seconds"]=time.perf_counter()-start
    row['backend_compile_calls']=compile_calls-compile_before
write_json(root/"history.json",rows)
write_json(root/"result.json",{"schema":1,"label":"G4-matched-minimal-driver","hardware":hardware,
    "provenance":source,"config_sha256":hashlib.sha256(canonical(config)).hexdigest(),
    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seed":args.seed,"warm_median_seconds":float(np.median([r["total_seconds"] for r in rows[2:]])),
    "warm_generations":len(rows)-2,"selection":"excluded-from-both-warm-time-samples",
    "memory":memory_record(),
    "post_warm_backend_compile_calls":sum(r['backend_compile_calls'] for r in rows[2:])})
