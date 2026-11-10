"""Numerical G1 audit against the independent float64 scalar reference.

Run inside the pinned image; this is fixture execution, never a throughput benchmark.
"""
import argparse
import hashlib
import time
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.contracts import Graph
from neuroevo.planning import plan
from neuroevo.runtime import compiled_step
from neuroevo.reference import step as reference
from neuroevo.artifacts import write_json

parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
args=parser.parse_args()
rng=np.random.default_rng(810)
rows=[]
for profile in ("dag","synchronous"):
    for fixture in range(8):
        n=16
        ids=tuple(map(int,rng.permutation(np.arange(100,100+n))))
        edges=[]
        for i in range(n):
            for j in range(2,n):
                if (profile == "synchronous" or i < j) and rng.uniform() < .25:
                    edges.append((ids[i],ids[j],float(np.float32(rng.normal(0,.1))),0))
        activation=np.ones(n,np.int32)
        activation[:2]=activation[-2:]=0
        g=Graph(ids,ids[:2],ids[-2:],rng.normal(0,.05,n).astype(np.float32),
                np.ones(n,np.float32),activation,tuple(edges),profile)
        p=plan(g)
        state=jnp.zeros(32,jnp.float32)
        scalar=np.zeros(n,np.float64)
        one_error=trajectory_error=ratio=0.
        first_call_seconds=None
        for t in range(128):
            observation=rng.uniform(-1,1,2).astype(np.float32)
            # One-step comparison isolates arithmetic using identical FP32 incoming state.
            expected_one,expected_state=reference(g,np.asarray(state)[:n].astype(np.float64),observation)
            expected_trajectory,scalar=reference(g,scalar,observation)
            start=time.perf_counter()
            actual,state=compiled_step(p,state,observation,profile=profile)
            jax.block_until_ready((actual,state))
            if t == 0 and fixture == 0:
                first_call_seconds=time.perf_counter()-start
            actual=np.asarray(actual)
            np.testing.assert_allclose(actual,expected_one,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(np.asarray(state)[:n],expected_state,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(actual,expected_trajectory,atol=1e-4,rtol=1e-4)
            one_error=max(one_error,float(np.max(np.abs(np.asarray(state)[:n]-expected_state))))
            trajectory_error=max(trajectory_error,float(np.max(np.abs(actual-expected_trajectory))))
            ratio=max(ratio,float(np.max(np.abs(np.asarray(state)[:n]-expected_state)/(1e-6+1e-5*np.abs(expected_state)))))
        timings=[]
        for _ in range(30):
            start=time.perf_counter()
            jax.block_until_ready(compiled_step(p,state,observation,profile=profile))
            timings.append(time.perf_counter()-start)
        rows.append({"profile":profile,"fixture":fixture,"steps":128,"nodes":n,"capacity":32,
                     "p":1,"B":1,"K":1,"edges":len(edges),"cold_first_step_seconds":first_call_seconds,
                     "warm_step_median_seconds":float(np.median(timings)),
                     "warm_step_p95_seconds":float(np.quantile(timings,.95)),
                     "one_step_max_abs_error":one_error,"trajectory_max_abs_error":trajectory_error,
                     "one_step_max_error_threshold_ratio":ratio})
write_json(args.output,{"schema":1,"status":"pass","hardware":hardware,"seed":810,
                       "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "fan_in_limit":32,"precision":"float32-HIGHEST","reference":"independent-float64",
                       "engine_sha256":hashlib.sha256(b"".join(p.read_bytes() for p in sorted(
                           Path(__import__("neuroevo").__file__).parent.rglob("*.py")))).hexdigest(),
                       "workload":"W2-small-G1-audit-not-G5",
                       "compilation_signatures":compiled_step._cache_size(),"fixtures":rows})
