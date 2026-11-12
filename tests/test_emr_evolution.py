"""A2 public discovery, persistent rollout and native fitness handoff."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from neuroevo.adapters.emr_evolution import EMREvolution, pad_population
from neuroevo.adapters.emr_phenotype import settled_reference
from neuroevo.contracts import stack_plans
from neuroevo.environments import memory_scenarios
from neuroevo.evaluation import memory_chunk, memory_initial
from neuroevo.runtime import block_step
from neuroevo.packing import route


def test_a2_native_evolution_uses_persistent_adaptive_rollout():
    base = json.loads(Path('configs/q2-capacity-v2.json').read_text())
    config = json.loads(Path('configs/a0-adapter-v1.json').read_text())
    config.update(population=8, max_depth=1, max_sparse_conns=256,
                  settling_steps=3, input_coords=[[-1, -1], [1, -1]],
                  output_coords=[[0, 1]])

    class Problem:
        input_shape = (2,)
        output_shape = (1,)
        jitable = True
        use_bias = False
        is_exhaustive_dataset = False

        def get_data(self):
            return [([0., 0.], [0.]), ([1., -1.], [1.])]

    adapter = EMREvolution(config, base, Problem(), node_capacity=20, edge_capacity=512)
    state = adapter.initialize(19)
    batch = adapter.ask(state, 'a2-test')
    assert len(batch.plans) == len(batch.phenotypes) == config['population']
    assert all(p.plan.w_in.shape[0] == 2 for p in batch.phenotypes)
    assert len(route(batch.plans)) <= len(adapter.bucket_family.widths)
    assert batch.padded_hidden >= max(len(p.grid_indices) for p in batch.phenotypes)
    assert batch.host_storage_bytes > 0
    adapter.pending = None
    adapter.discovery_microbatch = 3
    sliced = adapter.ask(state, 'a2-test')
    for full, chunk in zip(batch.phenotypes, sliced.phenotypes):
        np.testing.assert_array_equal(full.grid_indices,chunk.grid_indices)
        for left,right in zip(full.plan,chunk.plan):
            np.testing.assert_array_equal(left,right)
    batch = sliced

    # Padding does not change a phenotype's persistent layered output or state.
    observation = jnp.array([.2, -.4], jnp.float32)
    second_observation = jnp.array([-.6, .3], jnp.float32)
    for compact, padded in zip(batch.phenotypes, batch.plans):
        a, ca = block_step(compact.plan, jnp.zeros(len(compact.plan.node_mask)), observation)
        b, cb = block_step(padded, jnp.zeros(len(padded.node_mask)), observation)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-6)
        np.testing.assert_allclose(np.asarray(ca[2:2+len(compact.grid_indices)]),
                                   np.asarray(cb[2:2+len(compact.grid_indices)]), atol=2e-6)
        a2, ca2 = block_step(compact.plan, ca, second_observation)
        b2, cb2 = block_step(padded, cb, second_observation)
        np.testing.assert_allclose(np.asarray(a2), np.asarray(b2), atol=2e-6)
        np.testing.assert_allclose(np.asarray(ca2[2:2+len(compact.grid_indices)]),
                                   np.asarray(cb2[2:2+len(compact.grid_indices)]), atol=2e-6)

    obs, cues, delays = memory_scenarios(19, 'train', 0, delays=range(1, 9), repeats=2)
    scores = np.empty(config['population'], np.float32)
    for _, indices in route(batch.plans):
        packed = stack_plans([batch.plans[i] for i in indices])
        result = memory_chunk(packed, memory_initial(packed, 32), obs, cues, delays,
                              jnp.arange(17), profile='layered')
        assert np.asarray(result.valid).all()
        scores[list(indices)] = np.asarray(result.correct, np.float32).mean(axis=1)
    next_state, removed = adapter.tell(state, batch, batch.ids, scores)
    assert removed == []
    assert int(next_state.generation) == int(state.generation) + 1
    with pytest.raises(ValueError, match='Stale'):
        adapter.tell(state, batch, batch.ids, scores)
    next_batch = adapter.ask(next_state, 'a2-test')
    assert next_batch.ids != batch.ids
    assert len(next_batch.phenotypes) == config['population']
    adapter.pending = None
    adapter.phenotype_storage_budget_bytes = 1
    with pytest.raises(OverflowError, match='storage estimate'):
        adapter.ask(next_state, 'a2-test')


def test_a2_padding_rejects_insufficient_capacity():
    from neuroevo.adapters.emr_phenotype import EvaluatedPhenotype
    from neuroevo.substrate import blocks

    plan = blocks(np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32),
                  np.ones((2, 1), np.float32), np.ones((2, 2), bool),
                  np.zeros((2, 2), bool), np.ones((2, 1), bool))
    phenotype = EvaluatedPhenotype(plan, np.array([0, 1], np.int32),
                                   np.zeros((2, 2), np.float32), 1, 0)
    with pytest.raises(OverflowError, match='node capacity'):
        pad_population((phenotype,), 4)
