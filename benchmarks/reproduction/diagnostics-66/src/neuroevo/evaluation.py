"""Compiled time scans; shared scenarios stay outside the population axis."""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .runtime import population_step, step
from .contracts import stack_plans


class MemoryCarry(NamedTuple):
    policy: object
    correct: object
    valid: object


@partial(jax.jit, static_argnames=("profile", "reset_each_step"))
def memory_chunk(plans, carry, shared_observations, cues, delays, times, *,
                 profile="synchronous", reset_each_step=False):
    def body(c, item):
        time, obs = item
        broadcast = jnp.broadcast_to(obs, (c.policy.shape[0], *obs.shape))
        actions, new = population_step(plans, jnp.zeros_like(c.policy) if reset_each_step else c.policy,
                                       broadcast, profile=profile)
        active = time <= delays
        finite = jnp.all(jnp.isfinite(new), axis=-1)
        valid = c.valid & (finite | ~active[None, :])
        new = jnp.where(jnp.isfinite(new), new, 0)
        policy = jnp.where(active[None, :, None], new, c.policy)
        query = time == delays
        correct = c.correct + (query[None, :] & ((jnp.where(actions[..., 0] >= 0, 1, -1)) == cues))
        return MemoryCarry(policy, correct, valid), None
    return jax.lax.scan(body, carry, (times, shared_observations.swapaxes(0, 1)))[0]


def memory_initial(plans, episodes):
    p, n = plans.node_mask.shape
    return MemoryCarry(jnp.zeros((p, episodes, n), jnp.float32),
                       jnp.zeros((p, episodes), jnp.int32), jnp.ones((p, episodes), bool))


def make_cartpole_rollout(env, params, horizon=500, profile="dag"):
    # One compiled function per immutable workload signature, reused each generation.
    @jax.jit
    def evaluate(plans, episode_keys, shared_data):
        # shared_data is a dynamic read-only causal observation offset fixture.
        def one_candidate(p):
            def one_episode(root):
                reset_key, rollout_key = jax.random.split(root)
                observation, state = env.reset_env(reset_key, params)
                initial = (jnp.zeros(p.node_mask.shape, jnp.float32), state, observation,
                           jnp.bool_(True), jnp.float32(0), jnp.int32(0), jnp.bool_(True), jnp.bool_(False))
                def body(c, time):
                    policy, es, obs, active, reward_sum, length, valid, terminated = c
                    head, new_policy = step(p, policy, obs + shared_data[time], profile=profile)
                    finite = jnp.all(jnp.isfinite(new_policy))
                    action = jnp.argmax(head).astype(jnp.int32)  # first index wins ties
                    no, ns, reward, done, _ = env.step_env(jax.random.fold_in(rollout_key, time), es, action, params)
                    new_policy = jnp.where(jnp.isfinite(new_policy), new_policy, 0)
                    choose = lambda a, b: jnp.where(active, a, b)
                    # Gymnax done includes time limit; distinguish physical termination.
                    physical = ((jnp.abs(ns.x) > params.x_threshold)
                                | (jnp.abs(ns.theta) > params.theta_threshold_radians))
                    return (choose(new_policy, policy), jax.tree.map(choose, ns, es), choose(no, obs),
                            active & ~done, reward_sum + jnp.where(active, reward, 0),
                            length + active.astype(jnp.int32), valid & (finite | ~active),
                            terminated | (active & physical)), None
                final, _ = jax.lax.scan(body, initial, jnp.arange(horizon))
                policy, _, _, active, total, length, valid, terminated = final
                truncated = ~terminated & (length >= min(horizon, params.max_steps_in_episode))
                return total, length, terminated, truncated, valid, policy
            return jax.vmap(one_episode)(episode_keys)
        return jax.vmap(one_candidate)(plans)
    return evaluate


def evaluate_population(plans, evaluator, keys, shared_data, microbatch=32):
    if not plans or microbatch < 1:
        raise ValueError("Empty population or invalid microbatch")
    results = []
    for start in range(0, len(plans), microbatch):
        part = plans[start:start+microbatch]
        padded = part + [part[0]] * (microbatch-len(part))
        values = jax.device_get(evaluator(stack_plans(padded), keys, shared_data))
        results.append(tuple(np.asarray(v)[:len(part)] for v in values))
    return tuple(np.concatenate([r[i] for r in results]) for i in range(len(results[0])))
