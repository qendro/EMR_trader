"""Select and verify the backend BEFORE importing JAX anywhere else."""
import csv
import os
import platform
import subprocess


def initialize():
    requested = os.environ.get("NEUROEVO_BACKEND", "cpu")
    if requested not in ("cpu", "gpu"):
        raise ValueError("NEUROEVO_BACKEND must be cpu or gpu")
    selected = None
    if requested == "gpu":
        rows = list(csv.reader(subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version,memory.free",
            "--format=csv,noheader,nounits"], text=True).splitlines()))
        matches = [r for r in rows if r[1].strip() == os.environ["NEUROEVO_GPU_NAME"]
                   and int(r[2]) == int(os.environ["NEUROEVO_GPU_MIB"])]
        if len(matches) != 1:
            raise RuntimeError(f"GPU identity mismatch: {rows}")
        selected = [s.strip() for s in matches[0][:4]]
        available = (int(matches[0][4])-1024)/int(matches[0][2])
        fraction = min(.6, available)
        if fraction <= 0:
            raise RuntimeError("Insufficient free GPU memory after 1 GiB reserve")
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
        os.environ["CUDA_VISIBLE_DEVICES"] = selected[0]
    os.environ["JAX_PLATFORMS"] = "cuda" if requested == "gpu" else "cpu"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
    os.environ["JAX_ENABLE_X64"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    jax.config.update("jax_default_matmul_precision", "float32")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != requested:
        raise RuntimeError(f"Requested {requested}, found {devices}")
    if requested == "gpu" and devices[0].device_kind != selected[1]:
        raise RuntimeError("JAX device differs from selected NVIDIA GPU")
    return {"backend": requested, "devices": [str(d) for d in devices],
            "gpu": selected, "kernel": platform.release(), "python": platform.python_version(),
            "precision": "float32", "preallocate": False,
            "image_digest": os.environ.get("NEUROEVO_IMAGE_DIGEST", "not-recorded-development-run"),
            "allocator_fraction": float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0")),
            "development_mount": os.environ.get("PYTHONPATH") == "/app/src"}
