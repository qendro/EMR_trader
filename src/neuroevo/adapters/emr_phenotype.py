"""A1: export the last evaluated EMR generation into the layered block schema.

This is a version-locked seam to the pinned EMR generation caches. Its public
``population_substrates`` method decodes the *next* generation, so it cannot
represent the population whose fitness was just recorded.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from neuroevo.contracts import BlockPlan
from neuroevo.substrate import blocks, to_graph


@dataclass(frozen=True)
class EvaluatedPhenotype:
    plan: BlockPlan
    grid_indices: np.ndarray
    coordinates: np.ndarray
    settling_steps: int
    generation: int

    def graph(self):
        """Layered persistent graph; output is a linear logit."""
        return to_graph(self.plan)


def _row(value, index, shape, dtype, name):
    if value is None or not hasattr(value, 'shape') or len(value.shape) != len(shape) + 1:
        raise RuntimeError(f'Pinned EMR {name} cache unavailable or changed')
    if not 0 <= index < value.shape[0] or tuple(value.shape[1:]) != tuple(shape):
        raise RuntimeError(f'Pinned EMR {name} cache shape changed')
    actual = np.asarray(value[index])
    if actual.dtype != dtype:
        raise RuntimeError(f'Pinned EMR {name} cache dtype changed')
    return actual


def extract_evaluated(algo, index, *, node_capacity, edge_capacity):
    """Extract one genome from the most recent evaluated generation.

    Coordinates and indices refer to the full hierarchical grid. Connection
    masks mean nonzero evaluated weight; EMR has already applied its thresholds.
    Zero active hidden positions produce an empty hidden block and sigmoid(0).
    """
    from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid

    record = getattr(algo, '_last_eval', None)
    if not isinstance(record, dict) or 'generation' not in record:
        raise RuntimeError('No evaluated EMR generation is available')
    if (getattr(algo, 'dynamic_functions_mode', None) != 'disabled' or
            getattr(algo, 'agg_mode', None) != 'disabled' or
            getattr(algo, '_settling_rule', None) != 'emr' or
            getattr(algo, 'network_model', None) != 'grid'):
        raise ValueError('EMR phenotype uses an unsupported activation, aggregation or settling mode')
    positions = np.asarray(get_hierarchical_grid(algo.max_depth).all_positions)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise RuntimeError('Pinned EMR grid layout changed')
    recurrent = bool(getattr(getattr(algo, 'extended_config', None),
                             'allow_hidden_to_hidden', False))
    return _extract(index, positions, algo._cached_masks_A, algo._cached_W1,
                    algo._cached_W2, algo._cached_sparse_hh, recurrent,
                    len(algo.substrate_input_coords), len(algo.substrate_output_coords),
                    int(algo.activate_time) if recurrent else 1,
                    int(record['generation']), node_capacity, edge_capacity)


def extract_public(substrate, index, generation, *, node_capacity, edge_capacity):
    """Convert a public `population_substrates(state)` member before scoring it."""
    c = substrate.config
    if (c.activation_mode != 'disabled' or c.aggregation_mode != 'disabled' or
            c.hidden_activation != 'tanh' or c.output_activation != 'sigmoid' or
            c.hh_aggregation_mode != 'sum' or c.hh_activation_mode != 'every_iteration' or
            c.settling_rule != 'emr' or c.neuromodulation is not None):
        raise ValueError('Public EMR substrate uses an unsupported function mode')
    sparse = None
    if c.hidden_to_hidden:
        sparse = SimpleNamespace(from_indices=substrate.hh_from, to_indices=substrate.hh_to,
                                 weights=substrate.hh_weights, valid_mask=substrate.hh_valid,
                                 num_valid=np.asarray(substrate.hh_valid).sum(axis=1).astype(np.int32)
                                 if substrate.hh_valid is not None else None)
    elif any(getattr(substrate, name) is not None for name in
             ('hh_from', 'hh_to', 'hh_weights', 'hh_valid')):
        raise RuntimeError('Unexpected recurrent fields in feed-forward substrate')
    positions = np.asarray(substrate.positions)
    return _extract(index, positions, substrate.mask, substrate.W1, substrate.W2,
                    sparse, c.hidden_to_hidden, len(c.input_coords), len(c.output_coords),
                    int(c.activate_time) if c.hidden_to_hidden else 1,
                    generation, node_capacity, edge_capacity)


def _extract(index, positions, masks, w1, w2, sparse, recurrent, inputs, outputs,
             steps, generation, node_capacity, edge_capacity):
    if type(index) is not int or type(node_capacity) is not int or type(edge_capacity) is not int:
        raise ValueError('Index and capacities must be integers')
    if node_capacity < 1 or edge_capacity < 1:
        raise ValueError('Capacities must be positive')
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise RuntimeError('Pinned EMR grid layout changed')
    total = positions.shape[0]
    mask = _row(masks, index, (total,), np.dtype('bool'), 'mask')
    active = np.flatnonzero(mask).astype(np.int32)
    if inputs + len(active) + outputs > node_capacity:
        raise OverflowError('EMR phenotype exceeds node capacity')
    wi_full = _row(w1, index, (inputs, total), np.dtype('float32'), 'W1')
    wo_full = _row(w2, index, (total, outputs), np.dtype('float32'), 'W2')
    if (not np.isfinite(wi_full).all() or not np.isfinite(wo_full).all() or
            np.any(wi_full[:, ~mask] != 0) or np.any(wo_full[~mask] != 0)):
        raise RuntimeError('Pinned EMR weight cache contains invalid or inactive weights')
    wi, wo = wi_full[:, active], wo_full[active]
    delay = np.zeros((len(active), len(active)), np.float32)
    delay_mask = np.zeros(delay.shape, bool)
    if recurrent:
        if sparse is None:
            raise RuntimeError('Pinned EMR recurrent edge cache unavailable')
        n_valid = _row(sparse.num_valid, index, (), np.dtype('int32'), 'recurrent count')
        valid = _row(sparse.valid_mask, index, (sparse.valid_mask.shape[1],),
                     np.dtype('bool'), 'recurrent validity')
        source = _row(sparse.from_indices, index, valid.shape, np.dtype('int32'), 'recurrent source')
        target = _row(sparse.to_indices, index, valid.shape, np.dtype('int32'), 'recurrent target')
        weight = _row(sparse.weights, index, valid.shape, np.dtype('float32'), 'recurrent weight')
        if int(n_valid) != int(valid.sum()) or not np.isfinite(weight[valid]).all():
            raise RuntimeError('Pinned EMR recurrent edge cache is inconsistent')
        compact = np.full(total, -1, np.int32)
        compact[active] = np.arange(len(active), dtype=np.int32)
        for src, dst, value in zip(source[valid], target[valid], weight[valid]):
            if src < 0 or src >= total or dst < 0 or dst >= total or compact[src] < 0 or compact[dst] < 0:
                raise RuntimeError('EMR recurrent edge refers to an inactive or invalid position')
            a, b = int(compact[src]), int(compact[dst])
            if delay_mask[a, b] or value == 0:
                raise RuntimeError('EMR recurrent edge is duplicate or zero')
            delay[a, b] = value
            delay_mask[a, b] = True
    elif sparse is not None:
        raise RuntimeError('Unexpected EMR recurrent edge cache in feed-forward mode')
    edges = int(np.count_nonzero(wi) + np.count_nonzero(wo) + delay_mask.sum())
    if edges > edge_capacity:
        raise OverflowError('EMR phenotype exceeds edge capacity')
    plan = blocks(wi, delay, wo, wi != 0, delay_mask, wo != 0)
    if steps < 1:
        raise RuntimeError('Invalid EMR settling step count')
    return EvaluatedPhenotype(plan, active, positions[active].copy(), steps,
                              int(generation))


def settled_reference(phenotype, observations):
    """Independent FP64 upstream sample settling; returns sigmoid outputs.

    Each sample begins at zero hidden state. The input drive is injected on
    every settling step. Persistent rollout instead uses ``block_step`` and
    carries the previous hidden state across environment timesteps.
    """
    p = phenotype.plan
    x = np.asarray(observations, np.float64)
    if x.ndim != 2 or x.shape[1] != p.w_in.shape[0] or not np.isfinite(x).all():
        raise ValueError('Invalid EMR reference observations')
    drive = x @ p.w_in.astype(np.float64)
    hidden = np.zeros_like(drive)
    for _ in range(phenotype.settling_steps):
        hidden = np.tanh(drive + hidden @ p.w_delay.astype(np.float64))
    logits = hidden @ p.w_out.astype(np.float64)
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
