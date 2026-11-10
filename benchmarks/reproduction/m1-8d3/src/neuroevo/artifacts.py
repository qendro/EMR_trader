"""Atomic local checkpoints and backend-independent portable phenotypes."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
from .contracts import Plan, PRECISION
from .planning import plan, to_graph


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b"\n")


def save_checkpoint(path, state, config, provenance, history, champion=None, elapsed_seconds=0.):
    import jax
    payload = {"schema": 1, "state": jax.device_get(state), "config": config,
               "precision": PRECISION, "provenance": provenance, "history": history,
               "champion": champion, "elapsed_seconds": elapsed_seconds}
    raw = pickle.dumps(payload, protocol=5)
    atomic_write(path, b"NEUROEVO1\n"+hashlib.sha256(raw).digest()+raw)


def load_checkpoint(path, config, provenance):
    # Only load local trusted run checkpoints. Inference never deserializes pickle.
    data = Path(path).read_bytes()
    if data[:10] != b"NEUROEVO1\n" or hashlib.sha256(data[42:]).digest() != data[10:42]:
        raise ValueError("Corrupt checkpoint or unknown schema")
    result = pickle.loads(data[42:])
    if (result["schema"] != 1 or result["config"] != config or result["precision"] != PRECISION
            or result["provenance"] != provenance):
        raise ValueError("Checkpoint configuration/dependency/schema mismatch")
    import jax
    result["state"] = jax.tree.map(jax.device_put, result["state"])
    return result


def export(path, p, profile, metadata):
    import io
    arrays = {name: np.asarray(value) for name, value in zip(Plan._fields, p)}
    meta = {"schema": 1, "encoding": "dense-separate-delay-v1", "profile": profile, "k": 1,
            "precision": PRECISION, "activation_vocabulary": {"0": "identity", "1": "tanh"},
            "reset": "zero"}
    if set(meta) & set(metadata):
        raise ValueError("Metadata cannot override phenotype interpretation")
    meta.update(metadata)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays, metadata=np.frombuffer(canonical(meta), np.uint8))
    atomic_write(path, buf.getvalue())


def reload(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].tobytes())
        if (meta["schema"] != 1 or meta["encoding"] != "dense-separate-delay-v1"
                or meta["precision"] != PRECISION or meta["reset"] != "zero" or meta["k"] != 1
                or meta.get("activation_vocabulary") != {"0":"identity","1":"tanh"}
                or meta.get("action") not in ("argmax-first-tie","sign-positive-tie")):
            raise ValueError("Unsupported phenotype interpretation")
        p = Plan(*(data[name] for name in Plan._fields))
        capacity = len(p.node_mask)
        if not np.array_equal(p.node_mask, np.arange(capacity) < np.sum(p.node_mask)):
            raise ValueError("Invalid packed node mask")
        checked = plan(to_graph(p, meta["profile"]), capacity=capacity,
                       edge_capacity=int(np.sum(p.mask_zero)+np.sum(p.mask_delay)))
        for actual, expected in zip(p, checked):
            if np.asarray(actual).dtype != np.asarray(expected).dtype or not np.array_equal(actual, expected):
                raise ValueError("Invalid phenotype array or schedule")
    return p, meta
