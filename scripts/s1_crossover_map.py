"""S1 dense/edge crossover by graph size, degree distribution and episodes."""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.runtime import block_step
from neuroevo.sparse_runtime import edge_step, from_blocks
from neuroevo.substrate import blocks


def make_graph(inputs, hidden, edges, degree, seed):
    rng=np.random.default_rng(seed)
    wi=rng.normal(0,.1,(inputs,hidden)).astype(np.float32)
    wo=rng.normal(0,.1,(hidden,1)).astype(np.float32)
    if degree=='uniform':
        pairs=rng.choice(hidden*hidden,edges,replace=False)
    elif degree=='hub':
        hub=np.arange(hidden,dtype=np.int64)*hidden
        remaining=rng.choice(np.setdiff1d(np.arange(hidden*hidden),hub),
                             edges-hidden,replace=False)
        pairs=np.concatenate((hub,remaining))
    elif degree=='local':
        width=(edges+hidden-1)//hidden
        pairs=np.array([(source*hidden+(source+offset)%hidden)
                        for source in range(hidden) for offset in range(width)],
                       np.int64)[:edges]
    else:
        raise ValueError('Unknown degree distribution')
    wd=np.zeros((hidden,hidden),np.float32)
    mask=np.zeros_like(wd,bool)
    wd.flat[pairs]=rng.uniform(.01,.1,edges).astype(np.float32)
    mask.flat[pairs]=True
    return blocks(wi,wd,wo,np.ones_like(wi,bool),mask,np.ones_like(wo,bool))


def measure(step, plan, observations, repeats):
    @jax.jit
    def run(p,x):
        initial=jnp.zeros((x.shape[1],len(p.node_mask)),jnp.float32)
        def body(state,obs):
            heads,nxt=jax.vmap(lambda carry,value:step(p,carry,value))(state,obs)
            return nxt,heads
        return jax.lax.scan(body,initial,x)[1]
    begin=time.perf_counter()
    output=run(plan,observations)
    output.block_until_ready()
    compile_seconds=time.perf_counter()-begin
    times=[]
    for _ in range(repeats):
        begin=time.perf_counter()
        run(plan,observations).block_until_ready()
        times.append(time.perf_counter()-begin)
    return np.asarray(output),compile_seconds,float(np.median(times))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') and (
            len(jax.devices())!=1 or jax.devices()[0].platform!='gpu'):
        raise RuntimeError('S1 crossover card selection failed')
    rows=[]
    for inputs,hidden,edges in ((4,20,100),(25,128,1024),(400,512,8192)):
        for degree in ('uniform','hub','local'):
            dense=make_graph(inputs,hidden,edges,degree,hidden+edges)
            sparse=from_blocks(dense,edges)
            for episodes in (1,4,16):
                observations=jnp.asarray(np.random.default_rng(episodes+hidden).normal(
                    0,.1,(128,episodes,inputs)).astype(np.float32))
                a,ac,at=measure(block_step,dense,observations,args.repeats)
                b,bc,bt=measure(edge_step,sparse,observations,args.repeats)
                error=float(np.max(np.abs(a-b)))
                if error>2e-4:
                    raise AssertionError(f'S1 crossover parity failed: {error}')
                rows.append({'inputs':inputs,'hidden':hidden,'recurrent_edges':edges,
                    'degree_distribution':degree,'episodes':episodes,'steps':128,
                    'carry_steps':1,'dense_host_bytes':sum(np.asarray(v).nbytes for v in dense),
                    'edge_host_bytes':sum(np.asarray(v).nbytes for v in sparse),
                    'dense_compile_seconds':ac,'edge_compile_seconds':bc,
                    'dense_warm_seconds':at,'edge_warm_seconds':bt,
                    'edge_speedup':at/bt,'max_abs_error':error})
                print(json.dumps(rows[-1]),flush=True)
    with open(args.output,'w') as stream:
        json.dump({'device_kind':jax.devices()[0].device_kind,
                   'visible_devices':len(jax.devices()),
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'cells':rows},stream,indent=2)
