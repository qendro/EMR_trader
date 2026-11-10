FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg MUJOCO_GL=disable \
    JAX_DEFAULT_MATMUL_PRECISION=float32 JAX_ENABLE_X64=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/cache/jax-0.6.1
WORKDIR /app
COPY locks/requirements.txt /locks/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-build-isolation --require-hashes -r /locks/requirements.txt
COPY locks/compatibility.txt /locks/compatibility.txt
RUN pip install --no-build-isolation --require-hashes -r /locks/compatibility.txt
COPY vendor /vendor
COPY locks/sources.json /locks/sources.json
COPY scripts/verify_sources.py /locks/verify_sources.py
RUN python /locks/verify_sources.py
RUN pip install --no-deps --no-build-isolation /opt/tensorneat /opt/emr
COPY . /app
RUN pip install --no-deps --no-build-isolation . && pip check
ENTRYPOINT ["python", "-m", "neuroevo"]
CMD ["qualify", "--output", "/results/qualification.json"]
