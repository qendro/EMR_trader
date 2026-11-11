"""W8 shared-sequence tracking with persistent policy and Welford state."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .runtime import population_step
from .statistics import Moments, ratio


class SequenceCarry(NamedTuple):
    policy: object
    previous_action: object
    moments: Moments
    valid: object


def initial(plans, episodes=1):
    population, nodes = plans.node_mask.shape
    shape = (population, episodes)
    return SequenceCarry(jnp.zeros((*shape, nodes), jnp.float32),
                         jnp.zeros(shape, jnp.float32),
                         Moments(jnp.zeros(shape, jnp.int32),
                                 jnp.zeros(shape, jnp.float32),
                                 jnp.zeros(shape, jnp.float32)),
                         jnp.ones(shape, bool))


@partial(jax.jit, static_argnames=("profile",))
def chunk(plans, carry, shared_values, *, profile="layered"):
    """Consume T+1 shared rows to score T actions; carry spans adjacent chunks."""
    if shared_values.shape[1] != plans.inputs.shape[1]:
        raise ValueError("Shared sequence channels must match policy inputs")

    def body(state, item):
        observation, next_value = item
        population, episodes = state.previous_action.shape
        obs = jnp.broadcast_to(observation, (population, episodes, len(observation)))
        head, policy = population_step(plans, state.policy, obs, profile=profile)
        action = jnp.tanh(head[..., 0])
        finite = (jnp.all(jnp.isfinite(policy), axis=-1) & jnp.isfinite(action)
                  & jnp.isfinite(next_value))
        # Current channels are causal observations. Channel zero at the next
        # time is the target. Clipping keeps reward inside the frozen [-1, 1].
        reward = jnp.clip(1.0 - jnp.abs(action - next_value)
                          - 0.5 * jnp.abs(action - state.previous_action), -1.0, 1.0)
        count = state.moments.count + 1
        delta = reward - state.moments.mean
        mean = state.moments.mean + delta / count
        m2 = state.moments.m2 + delta * (reward - mean)
        moments = Moments(count, mean, m2)
        return SequenceCarry(policy, action, moments, state.valid & finite), (action, reward)

    final, (actions, rewards) = jax.lax.scan(
        body, carry, (shared_values[:-1], shared_values[1:, 0]))
    return final, actions, rewards


def sequence(seed, horizon, channels=2):
    if type(seed) is not int or type(horizon) is not int or horizon < 32 or channels != 2:
        raise ValueError("W8 requires an integer seed, at least 32 steps and two channels")
    return np.random.default_rng(seed).uniform(-1., 1., (horizon + 1, channels)).astype(np.float32)


def evaluate(plans, shared_values, *, chunk_steps=250, profile="layered", episodes=1):
    shared_values = np.asarray(shared_values, dtype=np.float32)
    if (shared_values.ndim != 2 or shared_values.shape[1] != plans.inputs.shape[1]
            or len(shared_values) < 33 or chunk_steps < 1
            or (len(shared_values) - 1) % chunk_steps != 0
            or not np.isfinite(shared_values).all()):
        raise ValueError("Invalid W8 sequence or chunk partition")
    state = initial(plans, episodes)
    actions, rewards = [], []
    for start in range(0, len(shared_values) - 1, chunk_steps):
        state, action, reward = chunk(
            plans, state, jnp.asarray(shared_values[start:start + chunk_steps + 1]),
            profile=profile)
        actions.append(np.asarray(action))
        rewards.append(np.asarray(reward))
    score, std, valid = ratio(state.moments)
    valid = valid & state.valid
    return {"score": np.asarray(score), "std": np.asarray(std),
            "valid": np.asarray(valid), "actions": np.concatenate(actions),
            "rewards": np.concatenate(rewards), "carry": state}
