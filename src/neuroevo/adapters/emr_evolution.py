"""A2: genuine EMR discovery with externally scored persistent phenotypes."""
from dataclasses import dataclass
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.adapters.emr import translate
from neuroevo.adapters.emr_phenotype import extract_public
from neuroevo.substrate import blocks


@dataclass(frozen=True)
class AdaptiveBatch:
    token: str
    ids: tuple[str, ...]
    nodes: np.ndarray
    conns: np.ndarray
    phenotypes: tuple
    plans: tuple
    padded_hidden: int


def pad_population(phenotypes, node_capacity):
    """Give variable active sets one bounded block signature for this generation."""
    if not phenotypes:
        raise ValueError('Empty adaptive population')
    inputs = phenotypes[0].plan.w_in.shape[0]
    outputs = phenotypes[0].plan.w_out.shape[1]
    counts = [len(p.grid_indices) for p in phenotypes]
    maximum = max(counts)
    available = node_capacity - inputs - outputs
    if maximum > available:
        raise OverflowError('Adaptive population exceeds node capacity')
    width = min(available, 1 << (maximum-1).bit_length()) if maximum else 0
    result = []
    for phenotype, count in zip(phenotypes, counts):
        p = phenotype.plan
        if p.w_in.shape[0] != inputs or p.w_out.shape[1] != outputs:
            raise ValueError('Adaptive population has mixed task interfaces')
        wi = np.zeros((inputs, width), np.float32)
        wd = np.zeros((width, width), np.float32)
        wo = np.zeros((width, outputs), np.float32)
        mi = np.zeros(wi.shape, bool)
        md = np.zeros(wd.shape, bool)
        mo = np.zeros(wo.shape, bool)
        wi[:, :count], wd[:count, :count], wo[:count] = p.w_in, p.w_delay, p.w_out
        mi[:, :count], md[:count, :count], mo[:count] = p.mask_in, p.mask_delay, p.mask_out
        padded = blocks(wi, wd, wo, mi, md, mo)
        node_mask = np.concatenate((np.ones(inputs, bool),
                                    np.arange(width) < count, np.ones(outputs, bool)))
        result.append(padded._replace(node_mask=node_mask))
    return tuple(result), width


class EMREvolution:
    """One A0-configured EMR population; TensorNEAT remains the sole evolver."""

    def __init__(self, config, q2_manifest, problem, *, node_capacity, edge_capacity):
        from emr_hyperneat import EMRHyperNEAT

        self.config = dict(config)
        self.node_capacity = node_capacity
        self.edge_capacity = edge_capacity
        self.problem = problem
        self.algo = EMRHyperNEAT()
        self.native_config = self.algo.create_config(translate(config, q2_manifest))
        self.pending = None

    def initialize(self, seed):
        state = self.algo.initialize(self.native_config, self.problem, seed=seed)
        self._tell = self.algo._compiled_tell
        controller = self.algo.neat_algo.species_controller
        def observe_stagnation(current, fitness):
            species_fitness = controller._update_species_fitness(current.species, fitness)
            after, _ = controller._stagnation(current.species, species_fitness,
                                             current.generation+1)
            removed = jnp.isfinite(current.species.species_keys) & ~jnp.isfinite(after.species_keys)
            return jnp.where(removed, current.species.species_keys, -1)
        self._observe = jax.jit(observe_stagnation)
        return state

    def ask(self, state, run_id):
        if self.pending is not None:
            raise ValueError('Previous adaptive population has not been scored')
        nodes, conns = (np.asarray(x).copy() for x in self.algo._compiled_ask(state)[:2])
        generation = int(state.generation)
        substrate = self.algo.population_substrates(state)
        if substrate.mask.shape[0] != len(nodes):
            raise RuntimeError('EMR substrate population does not match TensorNEAT ask')
        phenotypes = tuple(extract_public(substrate, i, generation,
            node_capacity=self.node_capacity, edge_capacity=self.edge_capacity)
            for i in range(len(nodes)))
        plans, width = pad_population(phenotypes, self.node_capacity)
        ids = tuple(f'{run_id}:{generation}:{i}' for i in range(len(nodes)))
        digest = hashlib.sha256(nodes.tobytes()+conns.tobytes()).hexdigest()
        token = hashlib.sha256(('|'.join(ids)+digest).encode()).hexdigest()
        nodes.flags.writeable = conns.flags.writeable = False
        self.pending = (token, ids)
        return AdaptiveBatch(token, ids, nodes, conns, phenotypes, plans, width)

    def tell(self, state, batch, ids, scores):
        if self.pending != (batch.token, tuple(ids)) or tuple(ids) != batch.ids:
            raise ValueError('Stale or misordered adaptive fitness association')
        current = self.algo._compiled_ask(state)
        digest = hashlib.sha256(np.asarray(current[0]).tobytes()+
                                np.asarray(current[1]).tobytes()).hexdigest()
        token = hashlib.sha256(('|'.join(batch.ids)+digest).encode()).hexdigest()
        if token != batch.token:
            raise ValueError('Adaptive population changed after ask')
        scores = np.asarray(scores, np.float32)
        if scores.shape != (len(batch.ids),) or np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError('Invalid adaptive fitness vector')
        if not np.isfinite(scores).any():
            raise ValueError('All adaptive candidates invalid')
        fitness = jnp.asarray(np.where(np.isfinite(scores), scores, -1e30))
        removed_keys = np.asarray(self._observe(state, fitness))
        removed = list(map(int, removed_keys[removed_keys >= 0]))
        result = self.algo._compiled_tell(*self.algo._before_tell(state, fitness))
        jax.block_until_ready(result)
        self.pending = None
        return result, removed
