"""Strict frozen W8 statistics audit, separate from the future sequence environment."""
import argparse
from neuroevo.backend import initialize
hardware=initialize()
import jax.numpy as jnp
import numpy as np
from neuroevo.artifacts import write_json
from neuroevo.statistics import empty,accumulate,merge,ratio

p=argparse.ArgumentParser()
p.add_argument("--output",required=True)
args=p.parse_args()
fixtures={"zero":np.zeros(10000,np.float32),"positive":np.full(10000,.5,np.float32),
          "negative":np.full(10000,-.5,np.float32),
          "near-constant":(.5+np.random.default_rng(12).choice([-1,1],10000)*np.float32(2**-24)).astype(np.float32),
          "full-range":np.random.default_rng(3).uniform(-1,1,10000).astype(np.float32)}
rows=[]
def check(a,b):
    np.testing.assert_allclose(a[0],b[0],atol=1e-5,rtol=0)
    np.testing.assert_allclose(a[1],b[1],atol=1e-5,rtol=1e-3)
    np.testing.assert_allclose(a[2],b[2],atol=1e-3,rtol=1e-3)
for name,rewards in fixtures.items():
    reference=rewards.astype(np.float64)
    expected=(reference.mean(),reference.std(),reference.mean()/(reference.std()+.01))
    results=[]
    for partition in ([10000],[2500]*4,[1,31,997,8971],[9999,1]):
        current=empty()
        start=0
        for size in partition:
            part=accumulate(empty(),jnp.asarray(rewards[start:start+size]),jnp.ones(size,bool))
            current=merge(current,part)
            start+=size
        score,std,valid=ratio(current)
        assert valid and current.count==10000
        observed=(float(current.mean),float(std),float(score))
        check(observed,expected)
        if results:
            check(observed,results[0])
        results.append(observed)
        rows.append({"fixture":name,"partition":partition,"mean":observed[0],"std":observed[1],
                     "score":observed[2],"abs_error_vs_float64":np.abs(np.array(observed)-expected).tolist()})
write_json(args.output,{"schema":1,"status":"pass","hardware":hardware,"steps":10000,
                       "protocol":"configs/w8-v1.json","sequence_environment_status":"M2-not-implemented",
                       "fixtures":rows})
