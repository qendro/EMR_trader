"""A2: genuine EMR discovery with externally scored persistent phenotypes."""
from dataclasses import dataclass
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.adapters.emr import translate
from neuroevo.adapters.emr_phenotype import extract_public_population
from neuroevo.packing import BucketFamily
from neuroevo.memory_schedule import estimate_population_storage, plan_bytes


@dataclass(frozen=True)
class AdaptiveBatch:
    token: str
    ids: tuple[str, ...]
    nodes: np.ndarray
    conns: np.ndarray
    phenotypes: tuple
    plans: tuple
    padded_hidden: int
    host_storage_bytes: int


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

    def __init__(self, config, q2_manifest, problem, *, node_capacity, edge_capacity,
                 phenotype_storage_budget_bytes=None, discovery_microbatch=None):
        from emr_hyperneat import EMRHyperNEAT

        self.config = dict(config)
        self.node_capacity = node_capacity
        self.edge_capacity = edge_capacity
        self.phenotype_storage_budget_bytes = phenotype_storage_budget_bytes
        if discovery_microbatch is not None and (type(discovery_microbatch) is not int or
                not 1 <= discovery_microbatch <= config['population']):
            raise ValueError('Invalid adaptive discovery microbatch')
        self.discovery_microbatch = discovery_microbatch
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
        substrates = self._substrates(state)
        if sum(s.mask.shape[0] for s in substrates) != len(nodes):
            raise RuntimeError('EMR substrate population does not match TensorNEAT ask')
        active_counts=np.concatenate([np.asarray(s.mask).sum(axis=1) for s in substrates])
        estimated=estimate_population_storage(active_counts,self.bucket_family)
        if (self.phenotype_storage_budget_bytes is not None and
                estimated > self.phenotype_storage_budget_bytes):
            raise OverflowError('Adaptive phenotype storage estimate exceeds declared budget')
        phenotypes = tuple(phenotype for substrate in substrates
                           for phenotype in extract_public_population(
                               substrate, generation, node_capacity=self.node_capacity,
                               edge_capacity=self.edge_capacity))
        plans, width = pad_population(phenotypes, self.node_capacity,
                                      family=self.bucket_family)
        storage=sum(plan_bytes(p.plan)+plan_bytes(q)+p.grid_indices.nbytes+
                    p.coordinates.nbytes for p,q in zip(phenotypes,plans))
        if (self.phenotype_storage_budget_bytes is not None and
                storage > self.phenotype_storage_budget_bytes):
            raise OverflowError('Adaptive phenotype storage exceeds declared budget')
        ids = tuple(f'{run_id}:{generation}:{i}' for i in range(len(nodes)))
        digest = hashlib.sha256(nodes.tobytes()+conns.tobytes()).hexdigest()
        token = hashlib.sha256(('|'.join(ids)+digest).encode()).hexdigest()
        nodes.flags.writeable = conns.flags.writeable = False
        self.pending = (token, ids)
        return AdaptiveBatch(token, ids, nodes, conns, phenotypes, plans, width, storage)

    def _substrates(self, state):
        if self.discovery_microbatch is None:
            return (self.algo.population_substrates(state),)
        from emr_hyperneat.substrate import (build_substrate,
            query_substrate,build_recurrent_substrate,query_recurrent_substrate)
        config=self.algo.substrate_config()
        population=self.algo._compiled_ask(state)
        transformed=self.algo._compiled_transform_batch(state,population)
        base=build_substrate(*query_substrate(self.algo._jitted_cppn_forward,
                             state,transformed,config),config)
        chunks=[]
        for start in range(0,base.mask.shape[0],self.discovery_microbatch):
            stop=min(start+self.discovery_microbatch,base.mask.shape[0])
            fields={name:(getattr(base,name)[start:stop] if getattr(base,name) is not None
                          else None) for name in ('mask','W1','W2','act_indices','agg_indices')}
            part=base._replace(**fields)
            if config.hidden_to_hidden:
                genomes=tuple(array[start:stop] for array in transformed)
                pairs=query_recurrent_substrate(self.algo._jitted_cppn_forward,
                                                state,genomes,part)
                part=build_recurrent_substrate(part,pairs)
            chunks.append(part)
        return tuple(chunks)

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
