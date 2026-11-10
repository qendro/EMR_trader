"""Welford/Chan sufficient statistics, including empty-chunk identity."""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class Moments(NamedTuple):
    count: object
    mean: object
    m2: object


def empty():
    return Moments(jnp.int32(0), jnp.float32(0), jnp.float32(0))


@jax.jit
def accumulate(initial, rewards, valid):
    def body(s, item):
        value, active = item
        count = s.count + active.astype(jnp.int32)
        count = jnp.where((s.count < 0) | (count < s.count), jnp.int32(-1), count)
        delta = value-s.mean
        mean = s.mean+jnp.where(active, delta/jnp.maximum(count, 1), 0)
        m2 = s.m2+jnp.where(active, delta*(value-mean), 0)
        return Moments(count, mean, m2), None
    return jax.lax.scan(body, initial, (rewards, valid))[0]


@jax.jit
def merge(a, b):
    count = a.count+b.count
    count = jnp.where((a.count < 0) | (b.count < 0) | (count < a.count), jnp.int32(-1), count)
    delta = b.mean-a.mean
    denom = jnp.maximum(count, 1).astype(jnp.float32)
    mean = a.mean+delta*b.count/denom
    m2 = a.m2+b.m2+delta**2*(a.count.astype(jnp.float32)*b.count.astype(jnp.float32)/denom)
    return Moments(count, jnp.where(count > 0, mean, 0), m2)


def ratio(s, epsilon=.01, minimum_count=32):
    valid = ((s.count >= minimum_count) & (s.m2 >= -1e-12)
             & jnp.isfinite(s.mean) & jnp.isfinite(s.m2))
    std = jnp.sqrt(jnp.maximum(s.m2, 0)/jnp.maximum(s.count, 1))
    return jnp.where(valid, s.mean/(std+epsilon), -jnp.inf), std, valid
