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


def stack_plans(plans):
    return Plan(*(np.stack(values) for values in zip(*plans)))
