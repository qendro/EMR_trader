"""S1 edge-list prototype parity and explicit duplicate/capacity policy."""
import jax.numpy as jnp
import numpy as np
import pytest

from neuroevo.runtime import block_step
from neuroevo.contracts import stack_plans
from neuroevo.evaluation import memory_chunk, memory_initial
from neuroevo.sparse_runtime import checked_edges, compiled_edge_step, from_blocks
from neuroevo.substrate import blocks


@pytest.mark.parametrize('hidden,edges', [(0,0),(8,20),(32,100)])
def test_s1_recurrent_edge_step_matches_dense_over_carried_sequence(hidden,edges):
    rng=np.random.default_rng(hidden+edges+19)
    wi=rng.uniform(-.2,.2,(2,hidden)).astype(np.float32)
    wo=rng.uniform(-.2,.2,(hidden,1)).astype(np.float32)
    wd=np.zeros((hidden,hidden),np.float32)
    if edges:
        pairs=rng.choice(hidden*hidden,size=edges,replace=False)
        wd.flat[pairs]=rng.uniform(-.2,.2,edges).astype(np.float32)
        wd[0,0]=np.float32(.1)  # Self loop; off-diagonal pairs include cycles.
    p=blocks(wi,wd,wo,wi!=0,wd!=0,wo!=0)
    edge=from_blocks(p,max(1,edges+1))
    dense_state=jnp.zeros(len(p.node_mask),jnp.float32)
    edge_state=jnp.zeros(len(p.node_mask),jnp.float32)
    for x in rng.uniform(-1,1,(32,2)).astype(np.float32):
        dense_head,dense_state=block_step(p,dense_state,jnp.asarray(x))
        edge_head,edge_state=compiled_edge_step(edge,edge_state,jnp.asarray(x))
        np.testing.assert_allclose(edge_head,dense_head,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(edge_state,dense_state,atol=2e-5,rtol=2e-5)


def test_s1_duplicate_and_capacity_are_refused():
    with pytest.raises(ValueError,match='Duplicate'):
        checked_edges(np.array([0,0],np.int32),np.array([1,1],np.int32),
                      np.array([.2,.3],np.float32),2)
    p=blocks(np.ones((1,2),np.float32),np.eye(2,dtype=np.float32),
             np.ones((2,1),np.float32),np.ones((1,2),bool),
             np.eye(2,dtype=bool),np.ones((2,1),bool))
    with pytest.raises(OverflowError,match='capacity'):
        from_blocks(p,1)


def test_edge_layout_runs_through_shared_population_evaluator():
    wi=np.array([[1.], [0.]],np.float32)
    wd=np.array([[.5]],np.float32)
    wo=np.array([[1.]],np.float32)
    dense=blocks(wi,wd,wo,wi!=0,wd!=0,wo!=0)
    edge=from_blocks(dense,1)
    obs=jnp.zeros((2,17,2),jnp.float32).at[:,0,0].set(jnp.array([-1.,1.]))
    cues=jnp.array([-1,1],jnp.int32)
    delays=jnp.array([4,4],jnp.int32)
    times=jnp.arange(17)
    p=stack_plans([dense,dense])
    q=stack_plans([edge,edge])
    a=memory_chunk(p,memory_initial(p,2),obs,cues,delays,times,profile='layered')
    b=memory_chunk(q,memory_initial(q,2),obs,cues,delays,times,profile='layered')
    np.testing.assert_array_equal(a.correct,b.correct)
    np.testing.assert_allclose(a.policy,b.policy,atol=2e-5,rtol=2e-5)
