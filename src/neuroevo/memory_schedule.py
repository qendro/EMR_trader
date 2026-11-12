"""Declared upper estimates for a compiled population rollout's live arrays."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolloutSchedule:
    microbatch: int
    estimated_bytes: int
    budget_bytes: int | None


def plan_bytes(plan):
    return sum(np.asarray(value).nbytes for value in plan)


def estimate_role_plan_bytes(inputs, hidden, outputs):
    """Conservative host estimate for weights, masks and role metadata."""
    if min(inputs, hidden, outputs) < 0 or inputs < 1 or outputs < 1:
        raise ValueError('Invalid role plan dimensions')
    matrix=inputs*hidden + hidden*hidden + hidden*outputs
    return 5*matrix + 4*(hidden+outputs) + (inputs+hidden+outputs) + \
           4*(2*inputs+hidden+2*outputs)


def estimate_population_storage(active_counts, family):
    """Compact plus bucket-padded plans and reversible grid metadata."""
    return sum(estimate_role_plan_bytes(family.inputs,int(count),family.outputs) +
               estimate_role_plan_bytes(family.inputs,family.width(int(count)),family.outputs) +
               12*int(count) for count in active_counts)


def estimate_rollout_bytes(plan, episodes, microbatch, task):
    """Estimate plan, policy/environment carry, and JAX temporary arrays.

    The eightfold factor budgets intermediate state and scan buffers. This is
    a scheduling estimate, not a device allocation guarantee; measured peak
    memory remains the acceptance evidence.
    """
    if task not in ('cartpole', 'cartpole-partial', 'memory', 'visual') or episodes < 1 or microbatch < 1:
        raise ValueError('Invalid rollout estimate')
    nodes = int(np.asarray(plan.node_mask).size)
    inputs = (int(np.asarray(plan.w_in).shape[0]) if hasattr(plan,'w_in')
              else int(np.asarray(plan.inputs).size))
    per_candidate = plan_bytes(plan) + 8 * episodes * (4 * nodes + 4 * inputs + 64)
    return int(microbatch * per_candidate + episodes * inputs * 4)


def choose_rollout_schedule(plan, episodes, max_microbatch, task, budget_bytes=None):
    if type(max_microbatch) is not int or max_microbatch < 1:
        raise ValueError('Invalid maximum microbatch')
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 1):
        raise ValueError('Invalid rollout memory budget')
    for size in range(max_microbatch, 0, -1):
        estimate = estimate_rollout_bytes(plan, episodes, size, task)
        if budget_bytes is None or estimate <= budget_bytes:
            return RolloutSchedule(size, estimate, budget_bytes)
    raise OverflowError('One candidate exceeds the declared rollout memory budget')
