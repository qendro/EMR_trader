"""Pinned TensorNEAT evolution only; no reimplementation of NEAT operators."""
from dataclasses import dataclass
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome, RecurrentGenome
from tensorneat.genome.gene import DefaultNode, DefaultConn
from tensorneat.genome.operations import DefaultMutation
from tensorneat.common import ACT, AGG, State
from neuroevo.contracts import Graph
from neuroevo.randomness import key


@dataclass(frozen=True)
class Batch:
    token: str
    ids: tuple[str, ...]
    nodes: np.ndarray
    conns: np.ndarray


class TensorNEATAdapter:
    def __init__(self, config):
        self.config = dict(config)
        profile = config["profile"]
        if profile not in ("dag", "synchronous"):
            raise ValueError("Direct genomes require dag or synchronous")
        genome_class = DefaultGenome if profile == "dag" else RecurrentGenome
        node = DefaultNode(activation_options=ACT.tanh, aggregation_options=AGG.sum,
                           bias_init_std=1., response_mutate_rate=0., response_replace_rate=0.)
        self.genome = genome_class(config["inputs"], config["outputs"],
            max_nodes=config["nodes"], max_conns=config["edges"], node_gene=node,
            conn_gene=DefaultConn(), mutation=DefaultMutation(**config["mutation"]),
            **({"activate_time": 1} if profile == "synchronous" else {}))
        self.algorithm = NEAT(self.genome, config["population"], verbose=False,
                              **config["species"])
        self._tell = jax.jit(self.algorithm.tell)
        self.pending = None

    def initialize(self, seed):
        return self.algorithm.setup(State(randkey=key(seed, "evolution")))

    def ask(self, state, run_id):
        nodes, conns = jax.device_get(self.algorithm.ask(state))
        digest = hashlib.sha256(nodes.tobytes()+conns.tobytes()).hexdigest()
        ids = tuple(f"{run_id}:{int(state.generation)}:{i}" for i in range(len(nodes)))
        token = hashlib.sha256(("|".join(ids)+digest).encode()).hexdigest()
        nodes, conns = nodes.copy(), conns.copy()
        nodes.flags.writeable = conns.flags.writeable = False
        self.pending = (token, ids)
        return Batch(token, ids, nodes, conns)

    def tell(self, state, batch, ids, scores):
        if self.pending != (batch.token, tuple(ids)) or tuple(ids) != batch.ids:
            raise ValueError("Stale or misordered fitness association")
        current = hashlib.sha256(np.asarray(state.pop_nodes).tobytes()+np.asarray(state.pop_conns).tobytes()).hexdigest()
        token = hashlib.sha256(("|".join(batch.ids)+current).encode()).hexdigest()
        if token != batch.token:
            raise ValueError("Population changed after ask")
        scores = np.asarray(scores, np.float32)
        if scores.shape != (len(ids),):
            raise ValueError("Wrong fitness vector shape")
        finite = np.isfinite(scores)
        if not finite.any():
            raise ValueError("All candidates invalid; refusing reproduction")
        fitness = jnp.asarray(np.where(finite, scores, -1e30))
        controller = self.algorithm.species_controller
        # Observe the actual upstream stagnation operation before new speciation.
        species_fitness = controller._update_species_fitness(state.species, fitness)
        after, _ = controller._stagnation(state.species, species_fitness, state.generation+1)
        removed = np.asarray(state.species.species_keys)[
            np.isfinite(np.asarray(state.species.species_keys)) & ~np.isfinite(np.asarray(after.species_keys))]
        result = self._tell(state, fitness)
        jax.block_until_ready(result)
        self.pending = None
        return result, list(map(int, removed))

    def decode(self, nodes, conns):
        nodes = np.asarray(nodes)
        conns = np.asarray(conns)
        active = nodes[~np.isnan(nodes[:, 0])]
        connections = conns[~np.isnan(conns[:, 0])]
        if not np.isfinite(active).all() or not np.isfinite(connections).all():
            raise ValueError("Nonfinite active genes")
        if np.any(active[:, 0] != np.floor(active[:, 0])):
            raise ValueError("Nonintegral semantic node IDs")
        if np.any(connections[:, :2] != np.floor(connections[:, :2])):
            raise ValueError("Nonintegral edge endpoints")
        # DefaultNode columns: semantic ID, bias, response, aggregation, activation.
        if np.any(active[:, 3] != 0) or np.any(~np.isin(active[:, 4], [-1, 0])):
            raise ValueError("Unsupported gene function")
        inputs = tuple(map(int, self.genome.input_idx))
        outputs = tuple(map(int, self.genome.output_idx))
        activation = np.where(active[:, 4] == -1, 0, 1).astype(np.int32)
        activation[np.isin(active[:, 0], outputs)] = 0  # upstream output bypass
        return Graph(tuple(map(int, active[:, 0])), inputs, outputs, active[:, 1],
                     active[:, 2], activation,
                     tuple((int(c[0]), int(c[1]), float(c[2]), 0) for c in connections),
                     self.config["profile"])
