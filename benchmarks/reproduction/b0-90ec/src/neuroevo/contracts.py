"""V0: finite, versioned graph and execution contracts (source -> target weights)."""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

PROFILES = ("dag", "synchronous", "layered")
ACTIVATIONS = {0: "identity", 1: "tanh"}
PRECISION = {"dtype": "float32", "matmul": "float32", "reference": "float64"}


@dataclass(frozen=True)
class Graph:
    node_ids: tuple[int, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    bias: np.ndarray
    response: np.ndarray
    activation: np.ndarray
    # (semantic source ID, semantic target ID, FP32 weight, delay)
    edges: tuple[tuple[int, int, float, int], ...]
    profile: str = "dag"
    k: int = 1


class Plan(NamedTuple):
    w_zero: object
    w_delay: object
    mask_zero: object
    mask_delay: object
    node_mask: object
    bias: object
    response: object
    activation: object
    inputs: object
    outputs: object
    order: object
    count: object
    node_ids: object


class BlockPlan(NamedTuple):
    """One hidden population; immediate input/output, delayed hidden edges."""
    w_in: object
    w_delay: object
    w_out: object
    mask_in: object
    mask_delay: object
    mask_out: object
    bias_hidden: object
    bias_output: object
    node_mask: object
    inputs: object
    outputs: object
    node_ids: object


def stack_plans(plans):
    if not plans or any(type(p) is not type(plans[0]) for p in plans):
        raise ValueError("Cannot batch empty or mixed phenotype encodings")
    return type(plans[0])(*(np.stack(values) for values in zip(*plans)))
