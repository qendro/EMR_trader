"""Q2 v2: native operators with an explicit finite-output fitness boundary."""
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
from tensorneat.algorithm import NEAT
from tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT
from tensorneat.algorithm.hyperneat.substrate import MLPSubstrate
from tensorneat.genome import DefaultGenome
from tensorneat.genome.gene import DefaultNode,DefaultConn
from tensorneat.genome.operations import DefaultMutation
from tensorneat.common import ACT,AGG,State
from neuroevo.artifacts import write_json
from neuroevo.resources import memory_record

p=argparse.ArgumentParser()
p.add_argument("--mode",choices=["direct","fixed"],required=True)
p.add_argument("--seed",type=int,default=700)
p.add_argument("--manifest",required=True)
p.add_argument("--output",required=True)
args=p.parse_args()
m=json.loads(Path(args.manifest).read_text())
if m.get('id')!='q2-native-direct-fixed-v2':
    raise ValueError('This runner requires the frozen v2 invalid-candidate policy')
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
cache_dir=os.environ.get('JAX_COMPILATION_CACHE_DIR')
write_json(root/'runtime.json',{'compilation_cache_dir':cache_dir,
    'cache_entries_at_start':len(list(Path(cache_dir).glob('*-cache'))) if cache_dir else 0,
    'jit_enabled':not jax.config.jax_disable_jit,
    'matmul_precision':jax.config.jax_default_matmul_precision,
    'jax_enable_x64':jax.config.jax_enable_x64})
write_json(root/"manifest.json",{"workload":m,"mode":args.mode,"seed":args.seed,"hardware":hardware,
    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "sources":json.loads(Path("locks/sources.json").read_text())})
inputs=np.random.Generator(np.random.PCG64(700)).uniform(-1,1,(64,4)).astype(np.float32)
targets=(inputs[:,:2]+np.float32(1))/np.float32(2)
np.savez(root/"dataset.npz",inputs=inputs,targets=targets)
fixed=args.mode=="fixed"
gene=DefaultGenome(4,1 if fixed else 2,max_nodes=m["genome_capacity_nodes"],max_conns=m["genome_capacity_edges"],
    node_gene=DefaultNode(activation_options=ACT.tanh,aggregation_options=AGG.sum,**m["node"]),
    conn_gene=DefaultConn(**m["connection"]),mutation=DefaultMutation(**m["mutation"]),
    output_transform=ACT.tanh if fixed else ACT.sigmoid)
neat=NEAT(gene,m["population"],verbose=False,**m["species"])
algo=HyperNEAT(MLPSubstrate(m["fixed_layers"],tuple(m["coordinate_range"])),neat,
    weight_threshold=m["fixed_weight_threshold"],max_weight=m["fixed_max_weight"],
    aggregation=AGG.sum,activation=ACT.tanh,activate_time=3,output_transform=ACT.sigmoid) if fixed else neat
ask=jax.jit(algo.ask)
transform=jax.jit(jax.vmap(algo.transform,in_axes=(None,0)))
forward=jax.jit(jax.vmap(jax.vmap(algo.forward,in_axes=(None,None,0)),in_axes=(None,0,None)))
tell=jax.jit(algo.tell)
try:
    begin=time.perf_counter()
    state=algo.setup(State(randkey=jax.random.PRNGKey(args.seed)))
    rows=[]
    for generation in range(m["cold_generations"]+m["warm_generations"]):
        start=time.perf_counter()
        genes=ask(state)
        phenotypes=transform(state,genes)
        outputs=forward(state,phenotypes,jnp.asarray(inputs))
        fitness=-jnp.mean((outputs-jnp.asarray(targets))**2,axis=(1,2))
        valid=jnp.all(jnp.isfinite(outputs),axis=(1,2)) & jnp.isfinite(fitness)
        if not bool(jnp.any(valid)):
            raise ValueError('All native candidates have nonfinite outputs or fitness')
        if not (root/'first-invalid.npz').exists() and bool(jnp.any(~valid)):
            slot=int(np.flatnonzero(~np.asarray(valid))[0])
            np.savez_compressed(root/'first-invalid.npz',inputs=inputs,
                outputs=np.asarray(outputs[slot]),nodes=np.asarray(genes[0][slot]),
                conns=np.asarray(genes[1][slot]))
            write_json(root/'first-invalid.json',{'seed':args.seed,'generation':generation,
                'slot':slot,'reason':'nonfinite-output-or-fitness',
                'nonfinite_outputs':int((~np.isfinite(np.asarray(outputs[slot]))).sum())})
        scores=jnp.where(valid,fitness,jnp.float32(-1e30))
        state=tell(state,scores)
        jax.block_until_ready(state)
        nodes,edges=jax.device_get(genes)
        counts=lambda a: {"min":int(a.min()),"median":float(np.median(a)),"max":int(a.max())}
        row={"generation":generation,"phase":"cold" if generation<m["cold_generations"] else "warm",
             "genotype_nodes":counts((~np.isnan(nodes[:,:,0])).sum(axis=1)),
             "genotype_edges":counts((~np.isnan(edges[:,:,0])).sum(axis=1)),
             "best_fitness":float(jnp.max(jnp.where(valid,fitness,-jnp.inf))),
             "valid_candidates":int(jnp.sum(valid)),"invalid_candidates":int(jnp.sum(~valid)),
             "memory":memory_record()}
        row["seconds"]=time.perf_counter()-start
        rows.append(row)
        write_json(root/"generations.json",rows)
        print(json.dumps({"mode":args.mode,"generation":generation,"seconds":row["seconds"]}),flush=True)
    write_json(root/"result.json",{"status":"measured","mode":args.mode,"hardware":hardware,
        "warm_median_seconds":float(np.median([r["seconds"] for r in rows if r["phase"]=="warm"])),
        "wall_seconds":time.perf_counter()-begin,"memory":memory_record(),
        "invalid_candidate_evaluations":sum(r['invalid_candidates'] for r in rows),
        "manifest_id":m['id'],
        "interpretation":"upstream-system-baseline-not-encoding-isolating-or-persistent-rollout"})
except Exception as error:
    write_json(root/"failure.json",{"status":"failed","type":type(error).__name__,"message":str(error)})
    raise
