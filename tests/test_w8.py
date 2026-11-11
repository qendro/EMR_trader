import jax.numpy as jnp
import numpy as np
import pytest

from neuroevo.contracts import stack_plans
from neuroevo.substrate import blocks
from neuroevo.w8 import chunk, evaluate, initial, sequence


def policy():
    p = blocks(np.array([[.5], [-.25]], np.float32),
               np.array([[.4]], np.float32),
               np.array([[1.]], np.float32),
               np.ones((2, 1), bool), np.ones((1, 1), bool),
               np.ones((1, 1), bool))
    return stack_plans([p])


def test_long_shared_sequence_carry_reducer_and_causality():
    plans = policy()
    values = sequence(700, 10000)
    result = evaluate(plans, values, chunk_steps=250)
    assert result["valid"].all()
    assert int(result["carry"].moments.count[0, 0]) == 10000
    assert result["actions"].shape == result["rewards"].shape == (10000, 1, 1)
    assert np.min(result["rewards"]) >= -1 and np.max(result["rewards"]) <= 1
    rewards = result["rewards"].astype(np.float64)
    reference = rewards.mean() / (rewards.std() + .01)
    np.testing.assert_allclose(result["score"], reference, atol=1e-3, rtol=1e-3)
    assert chunk._cache_size() == 1

    changed = values.copy()
    changed[501:] *= -.5
    other = evaluate(plans, changed, chunk_steps=250)
    np.testing.assert_array_equal(result["actions"][:500], other["actions"][:500])
    assert not np.array_equal(result["actions"][501:], other["actions"][501:])
    assert chunk._cache_size() == 1  # changed values stay dynamic

    repartitioned = evaluate(plans, values, chunk_steps=500)
    np.testing.assert_allclose(result["actions"], repartitioned["actions"], atol=0, rtol=0)
    np.testing.assert_allclose(result["score"], repartitioned["score"], atol=1e-5)
    assert chunk._cache_size() == 2  # one executable per chunk shape, not horizon

    longer = evaluate(plans, sequence(701, 20000), chunk_steps=250)
    assert longer["valid"].all() and chunk._cache_size() == 2


def test_w8_rejects_bad_partition_and_short_sequence():
    plans = policy()
    with pytest.raises(ValueError, match="chunk partition"):
        evaluate(plans, sequence(4, 100), chunk_steps=32)
    with pytest.raises(ValueError, match="at least 32"):
        sequence(4, 31)
