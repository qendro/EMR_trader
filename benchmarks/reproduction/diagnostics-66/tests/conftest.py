import os
# The CLI selects GPU visibility before pytest imports any JAX test modules.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")
