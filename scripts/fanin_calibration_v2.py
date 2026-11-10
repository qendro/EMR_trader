"""Frozen H1 preactivation fixtures. Only CPU/3070 determine the coefficient."""
import argparse
import hashlib
import json
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.contracts import stack_plans
from neuroevo.substrate import blocks
from neuroevo.runtime import block_preactivation,step
from neuroevo.artifacts import write_json
from neuroevo.runner import provenance

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--coefficient')
args=p.parse_args()
m=json.loads(Path('/manifests/h1-contract-v2.json').read_text())
frozen=json.loads(Path(args.coefficient).read_text()) if args.coefficient else None
if frozen is None and hardware['backend']=='gpu' and hardware['gpu'][1]!='NVIDIA GeForce RTX 3070':
    raise ValueError('Only CPU and gating RTX3070 may calibrate')
evaluate=jax.jit(jax.vmap(lambda plan,state,obs:jax.vmap(
    lambda s,x:block_preactivation(plan,s,x))(state,obs)))
single=jax.jit(block_preactivation)
outputs=jax.jit(lambda p,s,x:step(p,s,x,profile='layered'))
rows=[]
digest=hashlib.sha256()
for fanin in m['calibration_fanins']:
    h=fanin-4
    for seed in m['calibration_seeds']:
        for pattern in ('positive','mixed','cancellation'):
            rng=np.random.default_rng(seed+fanin)
            x=rng.normal(0,.2,(2,8,fanin)).astype(np.float32)
            w=rng.normal(0,1/fanin,(2,fanin,h)).astype(np.float32)
            bias=rng.normal(0,.01,(2,h)).astype(np.float32)
            if pattern=='positive': x,w,bias=np.abs(x),np.abs(w),np.abs(bias)
            if pattern=='cancellation':
                x[:,:,1::2]=x[:,:,::2]
                w[:,1::2]=-w[:,::2]
                bias[:]=0
            for value in (x,w,bias): digest.update(value.tobytes())
            plans=stack_plans([blocks(w[j,:4],w[j,4:],np.zeros((h,2),np.float32),
                np.ones((4,h),bool),np.ones((h,h),bool),np.zeros((h,2),bool),bias[j]) for j in range(2)])
            state=np.zeros((2,8,fanin+2),np.float32)
            state[:,:,4:4+h]=x[:,:,4:]
            actual=np.asarray(evaluate(plans,state,x[:,:,:4]))
            expected=np.stack([x[j].astype(np.float64)@w[j].astype(np.float64)+bias[j].astype(np.float64) for j in range(2)])
            scale=np.stack([np.abs(x[j].astype(np.float64))@np.abs(w[j].astype(np.float64))+np.abs(bias[j].astype(np.float64)) for j in range(2)])
            error=np.abs(actual-expected)
            assert np.isfinite(actual).all() and np.all(error[scale==0]==0)
            ratio=float(np.max(np.divide(error,2**-24*scale,out=np.zeros_like(error),where=scale>0)))
            rows.append({'fanin':fanin,'seed':seed,'pattern':pattern,'max_error':float(error.max()),
                         'max_error_over_uS':ratio})
            # A single-candidate GEMV is a different reduction layout from batched GEMM.
            single_actual=np.asarray(single(type(plans)(*(v[0] for v in plans)),state[0,0],x[0,0,:4]))
            single_error=np.abs(single_actual-expected[0,0])
            single_ratio=float(np.max(np.divide(single_error,2**-24*scale[0,0],out=np.zeros_like(single_error),where=scale[0,0]>0)))
            rows.append({'fanin':fanin,'seed':seed,'pattern':pattern+'-single',
                         'max_error':float(single_error.max()),'max_error_over_uS':single_ratio})
        # Repeated nonbinary terms expose long output-head accumulations. These
        # are independently declared constants, not the observed failing weight.
        value=np.float32(m['constant_output_weights'][m['calibration_seeds'].index(seed)])
        fixture=blocks(np.zeros((4,h),np.float32),np.zeros((h,h),np.float32),
            np.full((h,2),value,np.float32),np.zeros((4,h),bool),np.zeros((h,h),bool),
            np.ones((h,2),bool),np.full(h,20,np.float32))
        digest.update(np.asarray([value,h],np.float32).tobytes())
        actual,new=outputs(fixture,jnp.zeros(fanin+2,jnp.float32),jnp.zeros(4,jnp.float32))
        incoming=np.asarray(new)[4:4+h].astype(np.float64)
        expected=incoming@fixture.w_out.astype(np.float64)
        scale=np.abs(incoming)@np.abs(fixture.w_out.astype(np.float64))
        error=np.abs(np.asarray(actual)-expected)
        ratio=float(np.max(error/(2**-24*scale)))
        rows.append({'fanin':h,'seed':seed,'pattern':'constant-output-single',
                     'weight':float(value),'max_error':float(error.max()),'max_error_over_uS':ratio})
fixture_sha=digest.hexdigest()
maximum=max(r['max_error_over_uS'] for r in rows)
if frozen:
    assert frozen['fixture_sha256']==fixture_sha and frozen['precision']=='float32-HIGHEST'
    assert maximum<=frozen['coefficient'],'Frozen larger-fan-in threshold exceeded'
write_json(args.output,{'schema':1,'status':'validation-pass' if frozen else 'calibration-measured',
    'hardware':hardware,'provenance':provenance(),'fixture_sha256':fixture_sha,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'precision':'float32-HIGHEST','maximum_error_over_uS':maximum,
    'coefficient':frozen['coefficient'] if frozen else None,'fixtures':rows,
    'signatures':{'batched_hidden':evaluate._cache_size(),'single_hidden':single._cache_size(),
                  'single_output':outputs._cache_size()},'protocol_id':m['id']})
print(json.dumps({'status':'validation-pass' if frozen else 'calibration-measured',
                  'fixtures':len(rows),'maximum_error_over_uS':maximum,'fixture_sha256':fixture_sha}))
