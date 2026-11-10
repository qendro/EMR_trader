"""Independent scalar float64 semantics; does not use the JAX planner/runtime."""
import math
import numpy as np
from .contracts import Graph


def step(graph: Graph, state, observation):
    ids = graph.node_ids
    if graph.profile not in ("dag", "layered", "synchronous") or graph.k != 1:
        raise ValueError("Unsupported reference execution profile")
    if np.shape(state) != (len(ids),) or np.shape(observation) != (len(graph.inputs),):
        raise ValueError("Invalid reference state/observation shape")
    if not np.isfinite(state).all() or not np.isfinite(observation).all():
        raise ValueError("Nonfinite reference input")
    if not set(graph.activation) <= {0,1}:
        raise ValueError("Unsupported reference activation")
    if len(set(ids)) != len(ids) or not set(graph.inputs+graph.outputs) <= set(ids):
        raise ValueError("Invalid reference node/interface mapping")
    if set(graph.inputs) & set(graph.outputs):
        raise ValueError("Overlapping reference interfaces")
    with np.errstate(over="ignore",invalid="ignore"):
        for values in (graph.bias,graph.response):
            if np.shape(values) != (len(ids),) or not np.isfinite(np.asarray(values,np.float32)).all():
                raise ValueError("Invalid reference node attributes")
        seen=set()
        for source,target,weight,delay in graph.edges:
            if source not in ids or target not in ids or target in graph.inputs or delay not in (0,1):
                raise ValueError("Invalid reference edge")
            if (source,target,delay) in seen or not np.isfinite(np.float32(weight)):
                raise ValueError("Duplicate/nonfinite reference edge")
            if delay and graph.profile != "layered":
                raise ValueError("Unsupported reference edge delay")
            seen.add((source,target,delay))
    old = {i: float(state[j]) for j, i in enumerate(ids)}
    current = {i: 0.0 for i in ids}
    for i, x in zip(graph.inputs, np.asarray(observation, np.float32), strict=True):
        current[i] = float(x)
        if graph.profile == "synchronous":
            old[i] = float(x)
    attrs = {i: j for j, i in enumerate(ids)}
    def evaluate(i, immediate):
        j = attrs[i]
        acc = 0.0
        for src, dst, weight, delay in graph.edges:
            if dst == i:
                source = old if delay or graph.profile == "synchronous" else immediate
                acc += float(np.float32(weight)) * source[src]
        z = float(np.float32(graph.bias[j])) + float(np.float32(graph.response[j])) * acc
        if not math.isfinite(z):
            raise ValueError("Reference pre-activation overflow")
        return math.tanh(z) if graph.activation[j] == 1 else z
    if graph.profile == "synchronous":
        for i in ids:
            if i not in graph.inputs:
                current[i] = evaluate(i, old)
    else:
        remaining = set(ids) - set(graph.inputs)
        completed = set(graph.inputs)
        while remaining:
            ready = sorted(i for i in remaining if all(
                src in completed for src, dst, _, delay in graph.edges if dst == i and not delay))
            if not ready:
                raise ValueError("Cycle in immediate graph")
            for i in ready:
                current[i] = evaluate(i, current)
            completed.update(ready)
            remaining.difference_update(ready)
    result = np.array([current[i] for i in ids], np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite policy state")
    return np.array([current[i] for i in graph.outputs]), result
