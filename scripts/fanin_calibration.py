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
from neuroevo.runtime import block_preactivation
from neuroevo.artifacts import write_json
from neuroevo.runner import provenance

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--coefficient')
args=p.parse_args()
m=json.loads(Path('/app/configs/h1-contract-v1.json').read_text())
frozen=json.loads(Path(args.coefficient).read_text()) if args.coefficient else None
if frozen is None and hardware['backend']=='gpu' and hardware['gpu'][1]!='NVIDIA GeForce RTX 3070':
    raise ValueError('Only CPU and gating RTX3070 may calibrate')
evaluate=jax.jit(jax.vmap(lambda plan,state,obs:jax.vmap(
    lambda s,x:block_preactivation(plan,s,x))(state,obs)))
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
    'signatures':evaluate._cache_size()})
print(json.dumps({'status':'validation-pass' if frozen else 'calibration-measured',
                  'fixtures':len(rows),'maximum_error_over_uS':maximum,'fixture_sha256':fixture_sha}))
