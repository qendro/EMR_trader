"""Q1 probes and small upstream generation; evidence is profile-specific."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from .artifacts import write_json
from .runner import provenance, load_config
from .adapters.tensorneat import TensorNEATAdapter


@jax.jit
def primitives(key):
    x = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    scanned = jax.lax.scan(lambda c, a: (c+a, c+a), jnp.float32(0), x[0])[1]
    counted = jax.lax.fori_loop(0, 8, lambda i, v: v+i, jnp.int32(0))
    loop = jax.lax.while_loop(lambda v: v < 5, lambda v: v+1, jnp.int32(0))
    conditional = jax.lax.cond(counted > 0, lambda: x+1, lambda: x-1)
    gathered = x[:, jnp.array([1, 3, 5])]
    scatter = jnp.zeros(8).at[jnp.array([1, 1, 4])].add(jnp.array([2., 3., 7.]))
    return (x@x.T, scanned, counted, loop, conditional, gathered, scatter,
            jnp.sum(jnp.where(x > 10, x, 0), axis=1), jax.lax.top_k(x[0], 3),
            jnp.sort(x[0][::-1]), jax.random.normal(key, (32,)),
            jax.vmap(lambda a: a*a)(x))


def qualify(hardware, output):
    begin = time.perf_counter()
    values = jax.device_get(primitives(jax.random.PRNGKey(42)))
    cold = time.perf_counter()-begin
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    expected = (x@x.T, np.cumsum(x[0]), 28, 5, x+1, x[:, [1,3,5]],
                [0,5,0,0,7,0,0,0], np.where(x > 10, x, 0).sum(axis=1))
    for actual, reference in zip(values[:8], expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(values[8][0], [7,6,5])
    np.testing.assert_array_equal(values[8][1], [7,6,5])
    np.testing.assert_array_equal(values[9], np.arange(8))
    np.testing.assert_array_equal(values[11], x*x)
    np.testing.assert_array_equal(values[10], primitives(jax.random.PRNGKey(42))[10])
    assert np.isfinite(values[10]).all()
    config = load_config("configs/m1-smoke-v1.json")["adapter"]
    adapter = TensorNEATAdapter(config)
    state = adapter.initialize(913)
    batch = adapter.ask(state, "qualification")
    transformed = jax.jit(jax.vmap(adapter.algorithm.transform, in_axes=(None, 0)))(
        state, (state.pop_nodes, state.pop_conns))
    outputs = jax.jit(jax.vmap(adapter.algorithm.forward, in_axes=(None, 0, None)))(
        state, transformed, jnp.zeros(4))
    assert np.isfinite(outputs).all()
    next_state, _ = adapter.tell(state, batch, batch.ids, np.arange(config["population"], dtype=np.float32))
    assert int(next_state.generation) == 1
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        jax.block_until_ready(primitives(jax.random.PRNGKey(43)))
        timings.append(time.perf_counter()-start)
    result = {"schema": 1, "task": "Q1", "status": "pass", "hardware": hardware,
              "provenance": provenance(), "primitive_cold_seconds": cold,
              "primitive_warm_median_seconds": float(np.median(timings)),
              "primitives": ["fp32", "int32", "bool", "matmul", "vmap", "scan", "fori", "while",
                             "cond", "gather", "scatter-add", "masked-reduction", "top-k", "sort", "rng"],
              "tensorneat_generation": "pass", "wall_seconds": time.perf_counter()-begin}
    from .resources import memory_record
    result["memory"] = memory_record()
    write_json(output, result)
    return result
