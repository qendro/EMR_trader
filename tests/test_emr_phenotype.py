"""A1 evaluated-cache extraction and upstream/platform output contracts."""
from types import SimpleNamespace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuroevo.adapters.emr_phenotype import extract_evaluated, settled_reference
from neuroevo.runtime import block_step
from neuroevo.substrate import from_graph


def fixture(recurrent):
    from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid

    count = len(get_hierarchical_grid(1).all_positions)
    mask = np.zeros((1, count), bool)
    mask[0, [0, 2]] = True
    w1 = np.zeros((1, 2, count), np.float32)
    w2 = np.zeros((1, count, 1), np.float32)
    w1[0, :, 0] = [.75, -.5]
    w1[0, 0, 2] = np.nextafter(np.float32(0), np.float32(1))
    w2[0, 0, 0], w2[0, 2, 0] = .8, -.4
    sparse = None
    if recurrent:
        sparse = SimpleNamespace(
            num_valid=np.array([2], np.int32),
            valid_mask=np.array([[True, True, False]]),
            from_indices=np.array([[0, 2, 0]], np.int32),
            to_indices=np.array([[2, 2, 0]], np.int32),
            weights=np.array([[.3, -.2, 0]], np.float32),
        )
    algo = SimpleNamespace(
        _last_eval={'generation': 7}, max_depth=1,
        dynamic_functions_mode='disabled', agg_mode='disabled',
        _settling_rule='emr', network_model='grid',
        substrate_input_coords=[[-1, -1], [1, -1]],
        substrate_output_coords=[[0, 1]], activate_time=4,
        extended_config=SimpleNamespace(allow_hidden_to_hidden=recurrent),
        _cached_masks_A=mask, _cached_W1=w1, _cached_W2=w2,
        _cached_sparse_hh=sparse,
    )
    return algo


@pytest.mark.parametrize('recurrent', [False, True])
def test_a1_exact_graph_and_upstream_output_parity(recurrent):
    from emr_hyperneat.engine.evaluation import forward_hybrid_vmapped
    from emr_hyperneat.engine.functions import dense_substrate_forward

    algo = fixture(recurrent)
    phenotype = extract_evaluated(algo, 0, node_capacity=5, edge_capacity=7)
    p = phenotype.plan
    assert phenotype.generation == 7
    np.testing.assert_array_equal(phenotype.grid_indices, [0, 2])
    assert len(phenotype.graph().edges) == (7 if recurrent else 5)
    assert p.mask_in[0, 1]  # A positive subnormal weight survives threshold boundaries.
    assert np.count_nonzero(p.w_delay) == (2 if recurrent else 0)
    assert phenotype.settling_steps == (4 if recurrent else 1)
    observations = np.array([[.2, -.4], [-.8, .7], [1, 1]], np.float32)
    if recurrent:
        s = algo._cached_sparse_hh
        upstream = forward_hybrid_vmapped(jnp.asarray(observations),
            jnp.asarray(algo._cached_W1[0]), jnp.asarray(algo._cached_W2[0]),
            jnp.asarray(s.from_indices[0]), jnp.asarray(s.to_indices[0]),
            jnp.asarray(s.weights[0]), jnp.asarray(s.valid_mask[0]), 4)
    else:
        upstream = dense_substrate_forward(jnp.asarray(observations),
            jnp.asarray(algo._cached_W1[0]), jnp.asarray(algo._cached_W2[0]))
    np.testing.assert_allclose(settled_reference(phenotype, observations),
                               np.asarray(upstream), atol=2e-6, rtol=2e-6)
    if not recurrent:
        layered = [jax.nn.sigmoid(block_step(p, jnp.zeros(5), jnp.asarray(row))[0])
                   for row in observations]
        np.testing.assert_allclose(np.asarray(layered), np.asarray(upstream),
                                   atol=2e-6, rtol=2e-6)


def test_a1_capacity_and_invalid_cache_fail_closed():
    algo = fixture(True)
    with pytest.raises(OverflowError, match='node capacity'):
        extract_evaluated(algo, 0, node_capacity=4, edge_capacity=7)
    with pytest.raises(OverflowError, match='edge capacity'):
        extract_evaluated(algo, 0, node_capacity=5, edge_capacity=6)
    algo._cached_sparse_hh.to_indices[0, 0] = 1  # Inactive position.
    with pytest.raises(RuntimeError, match='inactive'):
        extract_evaluated(algo, 0, node_capacity=5, edge_capacity=7)
    algo._cached_sparse_hh.to_indices[0, 0] = 2
    algo._cached_sparse_hh.from_indices[0, 1] = 0
    with pytest.raises(RuntimeError, match='duplicate'):
        extract_evaluated(algo, 0, node_capacity=5, edge_capacity=7)


def test_a1_empty_hidden_is_an_exact_zero_edge_graph():
    algo = fixture(False)
    algo._cached_masks_A[:] = False
    algo._cached_W1[:] = 0
    algo._cached_W2[:] = 0
    phenotype = extract_evaluated(algo, 0, node_capacity=3, edge_capacity=1)
    assert phenotype.grid_indices.size == 0
    assert phenotype.graph().edges == ()
    assert from_graph(phenotype.graph()).w_in.shape == (2, 0)
    np.testing.assert_array_equal(settled_reference(phenotype,
        np.array([[.2, -.4]], np.float32)), [[.5]])
    logits, _ = block_step(phenotype.plan, jnp.zeros(3), jnp.array([.2, -.4]))
    np.testing.assert_array_equal(np.asarray(logits), [0])


def test_a1_rejects_unavailable_generation_and_unsupported_mode():
    algo = fixture(False)
    algo._last_eval = None
    with pytest.raises(RuntimeError, match='No evaluated'):
        extract_evaluated(algo, 0, node_capacity=5, edge_capacity=6)
    algo._last_eval = {'generation': 7}
    algo._settling_rule = 'drive_once'
    with pytest.raises(ValueError, match='unsupported'):
        extract_evaluated(algo, 0, node_capacity=5, edge_capacity=6)


@pytest.mark.parametrize('recurrent', [False, True])
def test_a1_pinned_generation_export(recurrent):
    from emr_hyperneat import EMRHyperNEAT
    from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid
    from emr_hyperneat.engine.evaluation import forward_hybrid_vmapped
    from emr_hyperneat.engine.functions import dense_substrate_forward
    from neuroevo.adapters.emr import translate

    base = json.loads(Path('configs/q2-capacity-v2.json').read_text())
    config = json.loads(Path('configs/a0-adapter-v1.json').read_text())
    config.update(population=8, max_depth=1)
    if recurrent:
        config.update(max_sparse_conns=256, settling_steps=3)
    else:
        config = {k: v for k, v in config.items()
                  if k not in ('band_threshold', 'max_sparse_conns', 'settling_steps')}
        config['recurrent'] = False
    inputs = np.array([[.2, -.4, .7, -.1], [-.8, .7, .1, .3]], np.float32)
    targets = np.array([[.4, .6], [.1, .9]], np.float32)

    class Problem:
        input_shape = (4,)
        output_shape = (2,)
        jitable = True
        use_bias = False
        is_exhaustive_dataset = False

        def get_data(self):
            return list(zip(inputs.tolist(), targets.tolist()))

    algo = EMRHyperNEAT()
    state = algo.initialize(algo.create_config(translate(config, base)), Problem(), seed=19)
    algo.run_generation(state, Problem(), verbose=False)
    active_counts = np.asarray(algo._cached_masks_A).sum(axis=1)
    index = int(active_counts.argmax())
    phenotype = extract_evaluated(algo, index, node_capacity=32, edge_capacity=512)
    assert len(phenotype.grid_indices) == active_counts[index]
    np.testing.assert_array_equal(phenotype.coordinates,
        np.asarray(get_hierarchical_grid(1).all_positions)[phenotype.grid_indices])
    graph_edges = phenotype.graph().edges
    offset = len(config['input_coords'])
    hidden = len(phenotype.grid_indices)
    actual_in = {(src, phenotype.grid_indices[dst-offset], np.float32(weight))
                 for src, dst, weight, delay in graph_edges if src < offset}
    expected_in = {(src, dst, np.float32(algo._cached_W1[index, src, dst]))
                   for src, dst in np.argwhere(np.asarray(algo._cached_W1[index]) != 0)}
    assert actual_in == expected_in
    actual_out = {(phenotype.grid_indices[src-offset], dst-offset-hidden, np.float32(weight))
                  for src, dst, weight, delay in graph_edges if dst >= offset + hidden}
    expected_out = {(src, dst, np.float32(algo._cached_W2[index, src, dst]))
                    for src, dst in np.argwhere(np.asarray(algo._cached_W2[index]) != 0)}
    assert actual_out == expected_out
    if recurrent:
        s = algo._cached_sparse_hh
        valid = np.asarray(s.valid_mask[index])
        actual_hh = {(phenotype.grid_indices[src-offset], phenotype.grid_indices[dst-offset],
                      np.float32(weight))
                     for src, dst, weight, delay in phenotype.graph().edges if delay}
        expected_hh = {(src, dst, weight) for src, dst, weight in zip(
            np.asarray(s.from_indices[index])[valid],
            np.asarray(s.to_indices[index])[valid],
            np.asarray(s.weights[index])[valid])}
        assert actual_hh == expected_hh
        upstream = forward_hybrid_vmapped(jnp.asarray(inputs), algo._cached_W1[index],
            algo._cached_W2[index], s.from_indices[index], s.to_indices[index],
            s.weights[index], s.valid_mask[index], phenotype.settling_steps)
    else:
        upstream = dense_substrate_forward(jnp.asarray(inputs), algo._cached_W1[index],
                                           algo._cached_W2[index])
    np.testing.assert_allclose(settled_reference(phenotype, inputs),
                               np.asarray(upstream), atol=2e-5, rtol=2e-5)
    if not recurrent:
        layered = [jax.nn.sigmoid(block_step(phenotype.plan,
            jnp.zeros(len(phenotype.plan.node_mask)), jnp.asarray(row))[0])
            for row in inputs]
        np.testing.assert_allclose(np.asarray(layered), np.asarray(upstream),
                                   atol=2e-5, rtol=2e-5)
