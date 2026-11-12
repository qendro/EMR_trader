"""W6 bounded-microbatch irregular and dense-input execution fixtures."""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.contracts import stack_plans
from neuroevo.runtime import population_step
from neuroevo.sparse_runtime import EdgeBlock, checked_edges
from neuroevo.substrate import blocks
from neuroevo.resources import memory_record


def graph(layout, inputs, hidden, total_edges, seed):
    rng=np.random.default_rng(seed)
    if layout=='edge':
        wi=np.zeros((inputs,hidden),np.float32)
        wi[np.arange(hidden)%inputs,np.arange(hidden)]=rng.normal(0,.1,hidden)
        wo=rng.normal(0,.1,(hidden,1)).astype(np.float32)
        recurrent=total_edges-hidden-hidden
        if recurrent < 0 or recurrent > hidden*hidden:
            raise ValueError('Invalid sparse W6 edge budget')
        chosen=rng.choice(hidden*hidden,recurrent,replace=False)
        src=(chosen//hidden).astype(np.int32)
        dst=(chosen%hidden).astype(np.int32)
        weights=rng.normal(0,.1,recurrent).astype(np.float32)
        src,dst,weights=checked_edges(src,dst,weights,hidden)
        return EdgeBlock(wi,wo,wi!=0,wo!=0,src,dst,weights,
                         np.ones(recurrent,bool),np.zeros(hidden,np.float32),
                         np.zeros(1,np.float32),np.ones(inputs+hidden+1,bool))
    if layout=='dense':
        wi=rng.normal(0,.1,(inputs,hidden)).astype(np.float32)
        wo=rng.normal(0,.1,(hidden,1)).astype(np.float32)
        wd=np.zeros((hidden,hidden),np.float32)
        mask=np.zeros((hidden,hidden),bool)
        recurrent=min(total_edges,hidden*hidden)
        chosen=rng.choice(hidden*hidden,recurrent,replace=False)
        wd.flat[chosen]=rng.normal(0,.1,recurrent)
        mask.flat[chosen]=True
        return blocks(wi,wd,wo,np.ones_like(wi,bool),mask,np.ones_like(wo,bool))
    raise ValueError('Unknown W6 layout')


@jax.jit
def rollout(plans, observations):
    population=plans.node_mask.shape[0]
    episodes=observations.shape[1]
    nodes=plans.node_mask.shape[1]
    def body(state,obs):
        shared=jnp.broadcast_to(obs,(population,*obs.shape))
        _,nxt=population_step(plans,state,shared,profile='layered')
        return nxt,None
    state,_=jax.lax.scan(body,jnp.zeros((population,episodes,nodes),jnp.float32),
                         observations)
    return state


def run_cell(layout, population, hidden, edges, inputs, episodes, horizon, microbatch):
    observations=jnp.asarray(np.random.default_rng(47).normal(
        0,.1,(horizon,episodes,inputs)).astype(np.float32))
    samples=[]
    compiled_before=rollout._cache_size()
    compiled_after_first=None
    active_edges=[]
    for start in range(0,population,microbatch):
        count=min(microbatch,population-start)
        plans=[graph(layout,inputs,hidden,edges,1000+start+i) for i in range(count)]
        active_edges.extend([int(np.count_nonzero(p.valid)+np.count_nonzero(p.mask_in)+
                                 np.count_nonzero(p.mask_out)) if layout=='edge'
                             else int(np.count_nonzero(p.mask_delay)+np.count_nonzero(p.mask_in)+
                                      np.count_nonzero(p.mask_out)) for p in plans])
        padded=plans+[plans[0]]*(microbatch-count)
        stacked=stack_plans(padded)
        began=time.perf_counter()
        final=rollout(stacked,observations)
        final.block_until_ready()
        if not np.isfinite(np.asarray(final)[:count]).all():
            raise AssertionError('Nonfinite W6 rollout')
        samples.append(time.perf_counter()-began)
        if compiled_after_first is None:
            compiled_after_first=rollout._cache_size()
    return {'layout':layout,'population':population,'hidden':hidden,'inputs':inputs,
            'episodes':episodes,'horizon':horizon,'carry_steps':1,
            'microbatch':microbatch,
            'active_nodes_per_graph':inputs+hidden+1,
            'min_active_edges':min(active_edges),'max_active_edges':max(active_edges),
            'compiled_signatures':rollout._cache_size()-compiled_before,
            'new_signatures_after_first_chunk':rollout._cache_size()-compiled_after_first,
            'first_chunk_seconds':samples[0],
            'warm_chunk_median_seconds':float(np.median(samples[1:])),
            'total_chunk_seconds':float(sum(samples)),
            'memory':memory_record()}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--population',type=int,default=256)
    parser.add_argument('--nodes',type=int,default=2048)
    parser.add_argument('--edges',type=int,default=32768)
    parser.add_argument('--episodes',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=128)
    parser.add_argument('--microbatch',type=int,default=4)
    args=parser.parse_args()
    if args.nodes <= 401:
        raise ValueError('W6 node count must fit the 400-input dense fixture')
    if os.environ.get('CUDA_VISIBLE_DEVICES') and (
            len(jax.devices())!=1 or jax.devices()[0].platform!='gpu'):
        raise RuntimeError('W6 selected GPU is unavailable or fell back to CPU')
    cells=[run_cell('edge',args.population,args.nodes-5,args.edges,4,
                    args.episodes,args.horizon,args.microbatch),
           run_cell('dense',args.population,args.nodes-401,args.edges,400,
                    args.episodes,args.horizon,args.microbatch)]
    with open(args.output,'w') as output:
        json.dump({'device':str(jax.devices()[0]),
                   'device_kind':jax.devices()[0].device_kind,
                   'visible_devices':len(jax.devices()),
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'cells':cells},output,indent=2)
    for cell in cells:
        print(json.dumps({k:v for k,v in cell.items() if k!='memory'}),flush=True)
