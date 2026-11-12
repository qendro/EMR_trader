"""A memory budget must reduce a rollout signature before allocation."""
import numpy as np
import pytest

from neuroevo.memory_schedule import choose_rollout_schedule, estimate_rollout_bytes
from neuroevo.substrate import blocks
from neuroevo.contracts import Graph
from neuroevo.planning import plan


def _plan(hidden):
    wi = np.zeros((4, hidden), np.float32)
    wd = np.zeros((hidden, hidden), np.float32)
    wo = np.zeros((hidden, 2), np.float32)
    return blocks(wi, wd, wo, np.zeros_like(wi, bool),
                  np.zeros_like(wd, bool), np.zeros_like(wo, bool))


def test_schedule_is_fixed_by_signature_and_budget():
    plan = _plan(32)
    one = estimate_rollout_bytes(plan, 20, 1, 'cartpole')
    two = estimate_rollout_bytes(plan, 20, 2, 'cartpole')
    schedule = choose_rollout_schedule(plan, 20, 16, 'cartpole', two)
    assert schedule.microbatch == 2
    assert schedule.estimated_bytes == two
    assert choose_rollout_schedule(plan, 20, 16, 'cartpole', one).microbatch == 1
    with pytest.raises(OverflowError):
        choose_rollout_schedule(plan, 20, 16, 'cartpole', one - 1)


def test_larger_bucket_and_scenarios_use_more_memory():
    small = _plan(8)
    large = _plan(64)
    assert estimate_rollout_bytes(large, 20, 4, 'memory') > estimate_rollout_bytes(small, 20, 4, 'memory')
    assert estimate_rollout_bytes(small, 100, 4, 'memory') > estimate_rollout_bytes(small, 20, 4, 'memory')


def test_direct_graph_plan_uses_same_budget_contract():
    graph=Graph((0,1,2),(0,1),(2,),np.zeros(3,np.float32),
                np.ones(3,np.float32),np.zeros(3,np.int32),((0,2,1.,0),))
    direct=plan(graph)
    schedule=choose_rollout_schedule(direct,2,4,'cartpole')
    assert schedule.microbatch==4
    assert schedule.estimated_bytes>0
