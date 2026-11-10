"""One capacity, explicit rejection; no gene identity is rewritten."""
import numpy as np
from .contracts import Graph, Plan, BlockPlan, PROFILES, ACTIVATIONS


def plan(graph: Graph, capacity=32, edge_capacity=128):
    ids = graph.node_ids
    if graph.profile not in PROFILES or graph.k != 1:
        raise ValueError("Unsupported profile or settling count")
    if not ids or len(ids) > capacity or len(graph.edges) > edge_capacity:
        raise ValueError("Capacity overflow or empty graph")
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 2**31 for i in ids):
        raise ValueError("Invalid/duplicate semantic node IDs")
    if not graph.inputs or not graph.outputs or set(graph.inputs) & set(graph.outputs):
        raise ValueError("Invalid input/output sets")
    for interface in (graph.inputs, graph.outputs):
        if len(set(interface)) != len(interface) or not set(interface) <= set(ids):
            raise ValueError("Invalid interface mapping")
    for a in (graph.bias, graph.response, graph.activation):
        if np.shape(a) != (len(ids),) or not np.isfinite(a).all():
            raise ValueError("Invalid active node attributes")
    if not set(graph.activation) <= set(ACTIVATIONS):
        raise ValueError("Unsupported activation")
    lookup = {v: i for i, v in enumerate(ids)}
    w = [np.zeros((capacity, capacity), np.float32) for _ in range(2)]
    masks = [np.zeros((capacity, capacity), bool) for _ in range(2)]
    seen = set()
    incoming = {i: set() for i in ids}
    for src, dst, weight, delay in graph.edges:
        if src not in lookup or dst not in lookup or dst in graph.inputs:
            raise ValueError("Invalid edge endpoint")
        if delay not in (0, 1) or not np.isfinite(np.float32(weight)):
            raise ValueError("Invalid edge delay/weight")
        if graph.profile != "layered" and delay:
            raise ValueError("Explicit delays require layered profile")
        if (src, dst, delay) in seen:
            raise ValueError("Duplicate edge")
        seen.add((src, dst, delay))
        w[delay][lookup[src], lookup[dst]] = weight
        masks[delay][lookup[src], lookup[dst]] = True
        if not delay:
            incoming[dst].add(src)
    order = []
    if graph.profile != "synchronous":
        pending = dict(incoming)
        while pending:
            ready = sorted(i for i, deps in pending.items() if not deps)
            if not ready:
                raise ValueError("Cycle in immediate graph")
            order.extend(lookup[i] for i in ready if i not in graph.inputs)
            pending = {i: deps - set(ready) for i, deps in pending.items() if i not in ready}
    else:
        order = [lookup[i] for i in ids if i not in graph.inputs]
    def pad(a, dtype):
        result = np.zeros(capacity, dtype)
        result[:len(a)] = a
        return result
    bias, response = pad(graph.bias, np.float32), pad(graph.response, np.float32)
    if not np.isfinite(bias).all() or not np.isfinite(response).all():
        raise ValueError("FP32 attribute overflow")
    return Plan(w[0], w[1], masks[0], masks[1], pad([True]*len(ids), bool),
                bias, response, pad(graph.activation, np.int32),
                np.array([lookup[i] for i in graph.inputs], np.int32),
                np.array([lookup[i] for i in graph.outputs], np.int32),
                pad(order, np.int32), np.int32(len(order)), pad(ids, np.int32))


def to_graph(p, profile, k=1):
    if isinstance(p, BlockPlan):
        from .substrate import to_graph as blocks_to_graph
        if profile != "layered" or k != 1:
            raise ValueError("Role blocks require layered K=1")
        return blocks_to_graph(p)
    n = int(np.sum(p.node_mask))
    ids = tuple(map(int, p.node_ids[:n]))
    edges = []
    for delay, (weights, mask) in enumerate(((p.w_zero, p.mask_zero), (p.w_delay, p.mask_delay))):
        for i, j in np.argwhere(mask):
            edges.append((ids[i], ids[j], float(weights[i, j]), delay))
    return Graph(ids, tuple(ids[i] for i in p.inputs), tuple(ids[i] for i in p.outputs),
                 np.asarray(p.bias[:n]), np.asarray(p.response[:n]),
                 np.asarray(p.activation[:n]), tuple(edges), profile, k)
