"""Frozen one-step 5x5 stroke-orientation diagnostic."""
import jax
import jax.numpy as jnp
import numpy as np
from .randomness import key


GRID = 5
TRAIN_POSITIONS = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
HELDOUT_POSITIONS = ((0, 1), (1, 0), (1, 2), (2, 1))


def input_coordinates():
    return [[-1. + x/2., -1. + y/2.] for y in range(GRID) for x in range(GRID)]


def visual_scenarios(seed, namespace, generation, episodes):
    positions = HELDOUT_POSITIONS if namespace in ('interpolation', 'generalization') else TRAIN_POSITIONS
    cell_count = len(positions)*2
    if episodes < cell_count or episodes % cell_count:
        raise ValueError('Visual episodes must balance positions and orientations')
    observations = np.zeros((episodes, GRID*GRID), np.float32)
    labels = np.empty(episodes, np.int32)
    sites = np.empty(episodes, np.int32)
    for e in range(episodes):
        site = (e//2) % len(positions)
        vertical = e % 2 == 1
        y, x = positions[site]
        for offset in range(3):
            observations[e, (y+offset if vertical else y)*GRID + (x if vertical else x+offset)] = 1.
        labels[e] = 1 if vertical else -1
        sites[e] = site
    keys = jnp.stack([key(seed, namespace, generation, episode=e) for e in range(episodes)])
    noise = jax.vmap(lambda k: jax.random.uniform(k, (GRID*GRID,), minval=-.15, maxval=.15))(keys)
    return jnp.asarray(observations) + noise, jnp.asarray(labels), jnp.asarray(sites)
