import jax.numpy as jnp
import numpy as np
import pytest

from neuroevo.evaluation import cartpole_observation
from neuroevo.p0 import paired_quality


def test_partial_observation_hides_velocity_only():
    values = jnp.array([.2, 3., -.4, 5.], jnp.float32)
    np.testing.assert_array_equal(cartpole_observation(values, 'position-angle'),
                                  np.array([.2, 0, -.4, 0], np.float32))
    np.testing.assert_array_equal(cartpole_observation(values, 'full'), values)
    with pytest.raises(ValueError):
        cartpole_observation(values, 'unknown')


def test_quality_bootstrap_uses_training_runs_as_units():
    random = np.full((5, 100), 20., np.float32)
    policy = np.full((5, 100), 150., np.float32)
    report = paired_quality(policy, random, bootstrap_samples=1000)
    assert report['passes']
    assert report['median_improvement'] == .26
    assert report['bootstrap_95_interval'][0] > 0
    with pytest.raises(ValueError):
        paired_quality(policy[:4], random[:4], bootstrap_samples=1000)
