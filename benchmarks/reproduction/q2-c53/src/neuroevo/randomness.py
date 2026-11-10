"""Semantic RNG namespaces, independent of microbatches and ordering."""
import jax
import jax.numpy as jnp

NAMESPACES = {"evolution": 11, "train": 23, "validation": 37,
              "interpolation": 41, "generalization": 53, "oracle": 67, "q2": 79}


def key(seed, namespace, generation=0, episode=0, time=0):
    result = jax.random.PRNGKey(seed)
    for value in (NAMESPACES[namespace], generation, episode, time):
        result = jax.random.fold_in(result, jnp.uint32(value))
    return result
