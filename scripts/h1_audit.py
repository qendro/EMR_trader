"""Post-calibration 256/1024-node development, execution and portability audit."""
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
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.adapters.cppn import CPPNQuery,query_scores
from neuroevo.fixed import FixedDecoder
from neuroevo.runtime import step,block_preactivation
from neuroevo.substrate import blocks,reference_step
from neuroevo.artifacts import write_json,export,reload
from neuroevo.runner import provenance,atomic_genes
from neuroevo.resources import memory_record

p=argparse.ArgumentParser()
p.add_argument('--coefficient',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
frozen=json.loads(Path(args.coefficient).read_text())
source=provenance()
assert frozen['provenance']==source,'Arithmetic source changed after calibration'
c=frozen['coefficient']

@jax.jit
def trajectory(p,observations,initial):
    def body(state,obs):
        output,new=step(p,state,obs,profile='layered')
        return new,(output,new)
    return jax.lax.scan(body,initial,observations)

preactivation=jax.jit(block_preactivation)
base=json.loads(Path('configs/h1-fixed-cartpole-smoke-v1.json').read_text())
adapter=TensorNEATAdapter(base['adapter'])
batch=adapter.ask(adapter.initialize(8200),'synthetic')
nodes,conns=batch.nodes[0].copy(),batch.conns[0].copy()
nodes[nodes[:,0]==4,1]=np.float32(np.arctanh(.6))
conns[~np.isnan(conns[:,0]),2]=0.
rows=[]
for total in (256,1024):
    h=total-6
    config={**base['decoder'],'hidden':h,'query_slab':4096}
    decoder=FixedDecoder(CPPNQuery(adapter),config)
    start=time.perf_counter()
    developed=decoder.decode(nodes,conns)
    decode_seconds=time.perf_counter()-start
    assert len(developed.node_mask)==total and developed.mask_delay.all()
    assert np.all(developed.w_delay!=0)
    export(root/f'decoded-{total}.npz',developed,'layered',
           {'action':'argmax-first-tie',**decoder.interpretation(),'provenance':source,
            'label':'synthetic-connected-CPPN-constant-score-not-evolved-agent'})
    atomic_genes(root/f'cppn-{total}.npz',{'nodes':nodes,'conns':conns})
    restored,_=reload(root/f'decoded-{total}.npz')
    for a,b in zip(developed,restored): np.testing.assert_array_equal(a,b)
    rng=np.random.default_rng(8200+total)
    observations=rng.uniform(-.2,.2,(128,4)).astype(np.float32)
    start=time.perf_counter()
    final,(outputs,states)=jax.device_get(trajectory(restored,observations,jnp.zeros(total,jnp.float32)))
    cold_seconds=time.perf_counter()-start
    # Decoded dense recurrent graph actually executes, with retained state.
    reference_state=np.zeros(total,np.float64)
    decoded_error=0.
    for t,obs in enumerate(observations):
        y,reference_state=reference_step(restored,reference_state,obs)
        np.testing.assert_allclose(outputs[t],y,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(states[t],reference_state,atol=1e-4,rtol=1e-4)
        decoded_error=max(decoded_error,float(np.max(np.abs(states[t]-reference_state))))
    assert not np.array_equal(states[0],states[1])
    reset_output,_=step(restored,jnp.zeros(total,jnp.float32),observations[1],profile='layered')
    carry_effect=float(np.max(np.abs(outputs[1]-np.asarray(reset_output))))
    assert carry_effect>1e-4,'Connected recurrence must affect the retained trajectory'
    incoming_hidden=states[-1,4:4+h].astype(np.float64)
    expected_output=incoming_hidden@restored.w_out.astype(np.float64)+restored.bias_output
    output_scale=np.abs(incoming_hidden)@np.abs(restored.w_out.astype(np.float64))+np.abs(restored.bias_output)
    output_ratio=float(np.max(np.abs(outputs[-1]-expected_output)/(c*2**-24*output_scale)))
    if output_ratio>1:
        write_json(root/'threshold-failure.json',{'status':'failed','nodes':total,
            'operation':'decoded-output-preactivation','coefficient':c,'ratio':output_ratio,
            'actual':outputs[-1].tolist(),'float64_same_incoming_hidden':expected_output.tolist(),
            'hardware':hardware,'provenance':source})
        raise AssertionError('Decoded output exceeds frozen preactivation threshold')
    # Separate contractive random fixture assesses stable unsaturated dynamics.
    wi=rng.normal(0,.1,(4,h)).astype(np.float32)
    wd=rng.normal(0,.15/h,(h,h)).astype(np.float32)
    wo=rng.normal(0,.1/h,(h,2)).astype(np.float32)
    fixture=blocks(wi,wd,wo,np.ones(wi.shape,bool),np.ones(wd.shape,bool),np.ones(wo.shape,bool),
                   rng.normal(0,.01,h).astype(np.float32))
    incoming=rng.normal(0,.2,total).astype(np.float32)
    obs=observations[0]
    z=np.asarray(preactivation(fixture,incoming,obs))
    expected=obs.astype(np.float64)@wi.astype(np.float64)+incoming[4:4+h].astype(np.float64)@wd.astype(np.float64)+fixture.bias_hidden.astype(np.float64)
    scale=np.abs(obs.astype(np.float64))@np.abs(wi.astype(np.float64))+np.abs(incoming[4:4+h].astype(np.float64))@np.abs(wd.astype(np.float64))+np.abs(fixture.bias_hidden.astype(np.float64))
    ratio=float(np.max(np.abs(z-expected)/(c*2**-24*scale)))
    assert ratio<=1,'Frozen calibrated preactivation threshold exceeded'
    _,(outputs,states)=jax.device_get(trajectory(fixture,observations,jnp.zeros(total,jnp.float32)))
    reference_state=np.zeros(total,np.float64)
    max_error=0.
    for t,obs in enumerate(observations):
        # One-step reference starts with identical FP32 state, separate from trajectory.
        same=np.zeros(total,np.float32) if t==0 else states[t-1]
        one,one_state=reference_step(fixture,same,obs)
        if h+4<=32:
            np.testing.assert_allclose(states[t],one_state,atol=1e-6,rtol=1e-5)
        y,reference_state=reference_step(fixture,reference_state,obs)
        np.testing.assert_allclose(outputs[t],y,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(states[t],reference_state,atol=1e-4,rtol=1e-4)
        max_error=max(max_error,float(np.max(np.abs(states[t]-reference_state))))
    signatures=trajectory._cache_size()
    changed=fixture._replace(w_delay=fixture.w_delay*.9,mask_delay=np.eye(h,dtype=bool))
    jax.block_until_ready(trajectory(changed,observations*.9,jnp.zeros(total,jnp.float32)))
    assert trajectory._cache_size()==signatures
    rows.append({'nodes':total,'hidden':h,'steps':128,'p':1,'B':1,'kind':'synthetic-not-H2-learning-or-G5',
        'decode_seconds':decode_seconds,'first_compiled_scan_seconds':cold_seconds,
        'decode':decoder.last_stats,'query_signatures':query_scores._cache_size(),
        'trajectory_error_max':max_error,'decoded_trajectory_error_max':decoded_error,
        'decoded_carry_vs_reset_output_difference':carry_effect,
        'decoded_output_preactivation_error_to_threshold_ratio':output_ratio,
        'preactivation_error_to_frozen_threshold_ratio':ratio,'trajectory_signatures':signatures})
write_json(root/'result.json',{'schema':1,'status':'pass','hardware':hardware,'provenance':source,
    'coefficient':c,'calibration_fixture_sha256':frozen['fixture_sha256'],'fixtures':rows,
    'memory':memory_record(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'not_qualified':['H2 learning','B0','M2 G2/G4/G5','W8 sequence environment']})
print(json.dumps({'status':'pass','fixtures':rows}))
