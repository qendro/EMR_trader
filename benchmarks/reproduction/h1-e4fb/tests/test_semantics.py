from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neuroevo.contracts import Graph, stack_plans
from neuroevo.planning import plan, to_graph
from neuroevo.reference import step as reference
from neuroevo.runtime import compiled_step
from neuroevo.environments import memory_oracle, memory_scenarios
from neuroevo.evaluation import memory_initial, memory_chunk
from neuroevo.cli import oracle


def graph(profile="dag", edges=((0, 2, 1., 0), (2, 3, .5, 0))):
    return Graph((0,1,2,3), (0,1), (3,), np.array([0,0,.1,-.2], np.float32),
                 np.ones(4, np.float32), np.array([0,0,1,1]), tuple(edges), profile)


@pytest.mark.parametrize("g", [graph(), graph("synchronous"),
    graph("synchronous", ((0,2,1.,0),(2,2,.8,0),(2,3,.5,0),(3,2,-.2,0))),
    graph(edges=())])
def test_float64_128_step_differential(g):
    p = plan(g)
    state, rs = jnp.zeros(32), np.zeros(4, np.float64)
    for t in range(128):
        x = np.array([np.sin(t)*.25, np.cos(t)*.1], np.float32)
        expected, rs = reference(g, rs, x)
        actual, state = compiled_step(p, state, x, profile=g.profile)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(state)[:4], rs, atol=1e-4, rtol=1e-4)
        assert np.all(np.asarray(state)[4:] == 0)


def test_mixed_delay_layered_reference():
    g = graph("layered", ((0,2,1.,0),(2,3,.5,0),(3,2,.75,1)))
    first, s = reference(g, np.zeros(4), np.array([1,0], np.float32))
    expected_h = np.tanh(1.1)
    np.testing.assert_allclose(first, [np.tanh(.5*expected_h-float(np.float32(.2)))], atol=1e-8)
    second, s2 = reference(g, s, np.array([0,0], np.float32))
    np.testing.assert_allclose(s2[2], np.tanh(float(np.float32(.1))+.75*s[3]))
    p = plan(g)
    assert p.mask_zero[2,3] and p.mask_delay[3,2]
    roundtrip = to_graph(p, "layered")
    np.testing.assert_allclose(reference(roundtrip, s, np.array([0,0]))[0], second)
    actual,_=compiled_step(p,jnp.zeros(32),jnp.array([1.,0.]),profile='layered')
    np.testing.assert_allclose(actual,first,atol=1e-6,rtol=1e-5)
    # Two distinct hidden layers: 0 -> 1 -> 2 -> 3, with delayed 2 -> 1.
    two = Graph((0,1,2,3),(0,),(3,),np.zeros(4,np.float32),np.ones(4,np.float32),
                np.array([0,1,1,0]),((0,1,1.,0),(1,2,.5,0),(2,3,2.,0),(2,1,.75,1)),"layered")
    y, first = reference(two,np.zeros(4),np.array([1.],np.float32))
    h1=np.tanh(1.)
    h2=np.tanh(.5*h1)
    np.testing.assert_allclose(y,[2*h2])
    y2, _ = reference(two,first,np.array([0.],np.float32))
    np.testing.assert_allclose(y2,[2*np.tanh(.5*np.tanh(.75*h2))])
    packed=plan(two)
    np.testing.assert_allclose(reference(to_graph(packed,"layered"),first,np.array([0.],np.float32))[0],y2)


def test_synchronous_reaction_latency():
    g = graph("synchronous")
    g = replace(g, bias=np.zeros(4, np.float32), activation=np.zeros(4, np.int32))
    p, s = plan(g), jnp.zeros(32)
    y0, s = compiled_step(p, s, jnp.array([1.,0.]), profile="synchronous")
    y1, s = compiled_step(p, s, jnp.array([0.,0.]), profile="synchronous")
    assert float(y0[0]) == 0 and float(y1[0]) == .5


@pytest.mark.parametrize("changes", [
    {"edges": ((0,99,1.,0),)}, {"edges": ((0,2,float("inf"),0),)},
    {"edges": ((0,2,1.,0),(0,2,2.,0))}, {"edges": ((2,0,1.,0),)},
    {"edges": ((2,3,1.,0),(3,2,1.,0))}, {"activation": np.array([0,0,9,0])},
    {"bias": np.array([0,0,np.nan,0])}, {"outputs": (99,)}, {"k": 2},
    {"node_ids": (0,1,2,2)}, {"edges": ((0,2,1.,1),)}])
def test_invalid_graph_rejected(changes):
    with pytest.raises(ValueError):
        plan(replace(graph(), **changes))


@pytest.mark.parametrize("profile",["dag","synchronous"])
def test_tanh_cannot_conceal_pre_activation_overflow(profile):
    g=Graph((0,1),(0,),(1,),np.zeros(2,np.float32),np.ones(2,np.float32),np.array([0,1]),
            ((0,1,float(np.finfo(np.float32).max),0),),profile)
    # The float64 oracle can represent the sum; FP32 execution must mark it invalid.
    expected,_=reference(g,np.zeros(2),np.array([2.],np.float32))
    assert expected[0]==1
    _,state=compiled_step(plan(g),jnp.zeros(32),jnp.array([2.],jnp.float32),profile=profile)
    assert not np.isfinite(np.asarray(state)[:2]).all()


def test_reference_rejects_nonfinite_weight_before_saturation():
    g=graph("synchronous",((0,2,float("inf"),0),))
    with pytest.raises(ValueError,match="reference edge"):
        reference(g,np.zeros(4),np.zeros(2,np.float32))


def test_caps_and_dynamic_topology_compilation():
    with pytest.raises(ValueError, match="Capacity"):
        plan(graph(), capacity=3)
    with pytest.raises(ValueError, match="Capacity"):
        plan(graph(), edge_capacity=1)
    p = plan(graph())
    compiled_step(p, jnp.zeros(32), jnp.zeros(2), profile="dag")
    before = compiled_step._cache_size()
    p2 = plan(graph(edges=((1,3,1.,0),)))
    compiled_step(p2, jnp.zeros(32), jnp.ones(2), profile="dag")
    assert compiled_step._cache_size() == before


def test_oracle_balance_permutations_chunks_isolation():
    assert oracle({})["oracle_accuracy"] == 1
    obs, cues, delays = memory_scenarios(7, "train")
    assert obs.shape == (32,17,2)
    for d in range(1,9):
        for cue in (-1,1):
            assert int(((delays == d) & (cues == cue)).sum()) == 2
    np.testing.assert_array_equal(obs, memory_scenarios(7, "train")[0])
    assert np.all(np.abs(np.asarray(obs)[:,1:,0]) <= .25)
    graphs = [memory_oracle(), replace(memory_oracle(), edges=((0,2,-1.,0),(2,2,2.,0)))]
    ps = stack_plans([plan(g) for g in graphs])
    initial = memory_initial(ps, 32)
    whole = memory_chunk(ps, initial, obs, cues, delays, jnp.arange(17))
    chunk = memory_chunk(ps, initial, obs[:,:3], cues, delays, jnp.arange(3))
    chunk = memory_chunk(ps, chunk, obs[:,3:], cues, delays, jnp.arange(3,17))
    for a,b in zip(whole, chunk):
        np.testing.assert_array_equal(a,b)
    assert np.mean(whole.correct[0]) == 1 and np.mean(whole.correct[1]) == 0
    permutation = np.random.default_rng(42).permutation(32)
    permuted = memory_chunk(ps, initial, obs[permutation], cues[permutation], delays[permutation], jnp.arange(17))
    np.testing.assert_array_equal(permuted.correct, whole.correct[:,permutation])
    swapped = memory_chunk(stack_plans([plan(g) for g in graphs[::-1]]), initial, obs, cues, delays, jnp.arange(17))
    np.testing.assert_array_equal(swapped.correct, whole.correct[::-1])
    count = memory_chunk._cache_size()
    memory_chunk(ps, initial, obs*.9, cues, delays, jnp.arange(17))
    assert memory_chunk._cache_size() == count


def test_padded_nan_masks_sanitized():
    p = plan(graph("synchronous"))
    bad = np.asarray(p.w_zero).copy()
    bad[~p.mask_zero] = np.nan
    p = p._replace(w_zero=bad)
    _, state = compiled_step(p, jnp.full(32, jnp.nan).at[:4].set(0), jnp.ones(2), profile="synchronous")
    assert np.isfinite(state).all()
