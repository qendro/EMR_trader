"""Application-independent memory scenarios and the qualified Gymnax adapter."""
import numpy as np
import jax
import jax.numpy as jnp
from .contracts import Graph
from .randomness import key


def memory_scenarios(seed, namespace, generation=0, *, delays=range(1, 9), repeats=2,
                     distractors=True):
    pairs = [(cue, delay) for delay in delays for cue in (-1, 1) for _ in range(repeats)]
    cues = np.array([p[0] for p in pairs], np.int32)
    delay_array = np.array([p[1] for p in pairs], np.int32)
    horizon = 17
    if not pairs or delay_array.max() >= horizon:
        raise ValueError("Memory delay outside frozen 17-step horizon")
    keys = jnp.stack([key(seed, namespace, generation, episode=i) for i in range(len(pairs))])
    noise = jax.vmap(lambda k: jax.random.uniform(k, (horizon,), minval=-.25, maxval=.25))(keys)
    obs = jnp.zeros((len(pairs), horizon, 2), jnp.float32)
    obs = obs.at[:, :, 0].set(noise if distractors else 0)
    obs = obs.at[:, 0, 0].set(cues)
    obs = obs.at[jnp.arange(len(pairs)), delay_array, 1].set(1)
    return obs, jnp.asarray(cues), jnp.asarray(delay_array)


def memory_oracle():
    return Graph((0, 1, 2), (0, 1), (2,), np.zeros(3, np.float32),
                 np.ones(3, np.float32), np.array([0, 0, 1]),
                 ((0, 2, 1., 0), (2, 2, 2., 0)), "synchronous")


def cartpole():
    import gymnax
    env, params = gymnax.make("CartPole-v1")
    return env, params.replace(max_steps_in_episode=500)
