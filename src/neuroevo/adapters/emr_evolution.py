"""A2: genuine EMR discovery with externally scored persistent phenotypes."""
from dataclasses import dataclass
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.adapters.emr import translate
from neuroevo.adapters.emr_phenotype import extract_public
from neuroevo.packing import BucketFamily


@dataclass(frozen=True)
class AdaptiveBatch:
    token: str
    ids: tuple[str, ...]
    nodes: np.ndarray
    conns: np.ndarray
    phenotypes: tuple
    plans: tuple
    padded_hidden: int


def pad_population(phenotypes, node_capacity, *, family=None):
    """Route each phenotype to its fixed width bucket; preserve input order."""
    if not phenotypes:
        raise ValueError('Empty adaptive population')
    inputs = phenotypes[0].plan.w_in.shape[0]
    outputs = phenotypes[0].plan.w_out.shape[1]
    family = family or BucketFamily.create(inputs, outputs, node_capacity)
    if (family.inputs, family.outputs)!=(inputs, outputs) or \
            family.max_width != node_capacity-inputs-outputs:
        raise ValueError('Adaptive bucket family does not match task capacity')
    result=tuple(family.pack(p) for p in phenotypes)
    return result, max(p.w_in.shape[1] for p in result)


class EMREvolution:
    """One A0-configured EMR population; TensorNEAT remains the sole evolver."""

    def __init__(self, config, q2_manifest, problem, *, node_capacity, edge_capacity):
        from emr_hyperneat import EMRHyperNEAT

        self.config = dict(config)
        self.node_capacity = node_capacity
        self.edge_capacity = edge_capacity
        self.bucket_family = BucketFamily.create(len(config['input_coords']),
                                                 len(config['output_coords']), node_capacity)
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
        plans, width = pad_population(phenotypes, self.node_capacity,
                                      family=self.bucket_family)
        ids = tuple(f'{run_id}:{generation}:{i}' for i in range(len(nodes)))
        digest = hashlib.sha256(nodes.tobytes()+conns.tobytes()).hexdigest()
        token = hashlib.sha256(('|'.join(ids)+digest).encode()).hexdigest()
        nodes.flags.writeable = conns.flags.writeable = False
        self.pending = (token, ids)
        return AdaptiveBatch(token, ids, nodes, conns, phenotypes, plans, width)

    def promote_node_capacity(self, new_capacity):
        if self.pending is not None:
            raise ValueError('Cannot promote capacity while a population is pending')
        self.bucket_family = self.bucket_family.promote(new_capacity)
        self.node_capacity = new_capacity

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
