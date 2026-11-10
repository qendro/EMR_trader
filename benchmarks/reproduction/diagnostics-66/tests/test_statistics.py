import jax.numpy as jnp
import numpy as np
import pytest
from neuroevo.statistics import empty,accumulate,merge,ratio


@pytest.mark.parametrize("rewards", [np.zeros(10000,np.float32),np.full(10000,.5,np.float32),
    np.full(10000,-.5,np.float32),
    (.5+np.random.default_rng(12).choice([-1,1],10000)*np.float32(2**-24)).astype(np.float32),
    np.random.default_rng(3).uniform(-1,1,10000).astype(np.float32)])
def test_long_reducer_and_chunk_merge(rewards):
    reference=rewards.astype(np.float64)
    mean,std=reference.mean(),reference.std()
    expected_score=mean/(std+.01)
    results=[]
    for sizes in ([10000],[2500]*4,[1,31,997,8971],[9999,1]):
        combined=empty()
        start=0
        for size in sizes:
            chunk=accumulate(empty(),jnp.asarray(rewards[start:start+size]),jnp.ones(size,bool))
            combined=merge(combined,chunk)
            start+=size
        score,actual_std,valid=ratio(combined)
        assert valid and int(combined.count)==10000
        np.testing.assert_allclose(combined.mean,mean,atol=1e-5,rtol=0)
        np.testing.assert_allclose(actual_std,std,atol=1e-5,rtol=1e-3)
        np.testing.assert_allclose(score,expected_score,atol=1e-3,rtol=1e-3)
        results.append((float(combined.mean),float(actual_std),float(score)))
    for result in results:
        np.testing.assert_allclose(result,results[0],atol=1e-3,rtol=1e-3)


def test_short_invalid_empty_identity_and_masks():
    s=accumulate(empty(),jnp.ones(32),jnp.arange(32)<31)
    assert not bool(ratio(s)[2])
    both=merge(empty(),s)
    for a,b in zip(both,s):
        np.testing.assert_array_equal(a,b)
    zero=merge(empty(),empty())
    assert np.isfinite(zero.mean) and int(zero.count)==0
