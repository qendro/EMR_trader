"""Compare dense and edge recurrent execution on declared synthetic graph cells."""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.runtime import block_step
from neuroevo.sparse_runtime import edge_step, from_blocks
from neuroevo.substrate import blocks
from neuroevo.artifacts import reload as reload_artifact
from neuroevo.contracts import BlockPlan


def fixture(inputs, hidden, edges, seed):
    if edges > hidden * hidden:
        raise ValueError('More unique edges requested than hidden pairs')
    rng = np.random.default_rng(seed)
    wi = rng.normal(0, .1, (inputs, hidden)).astype(np.float32)
    wo = rng.normal(0, .1, (hidden, 1)).astype(np.float32)
    wd = np.zeros((hidden, hidden), np.float32)
    mask = np.zeros((hidden, hidden), bool)
    chosen = rng.choice(hidden * hidden, size=edges, replace=False)
    wd.flat[chosen] = rng.normal(0, .1, edges).astype(np.float32)
    mask.flat[chosen] = True
    return blocks(wi, wd, wo, np.ones_like(wi, bool), mask,
                  np.ones_like(wo, bool))


def time_scan(step, plan, observations, repeats):
    @jax.jit
    def run(p, x):
        def body(carry, obs):
            _, state = step(p, carry, obs)
            return state, state[-1]
        return jax.lax.scan(body, jnp.zeros(p.node_mask.shape, jnp.float32), x)[1]
    t0 = time.perf_counter()
    result = run(plan, observations)
    result.block_until_ready()
    compile_seconds = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(plan, observations).block_until_ready()
        times.append(time.perf_counter()-start)
    return np.asarray(result), compile_seconds, float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--artifact')
    args = parser.parse_args()
    cells = [(4, 20, 100), (25, 128, 1024), (400, 2048, 32768)]
    report = {'device': str(jax.devices()[0]),
              'device_kind': jax.devices()[0].device_kind,
              'visible_devices': len(jax.devices()), 'cells': []}
    for inputs, hidden, edges in cells:
        dense = fixture(inputs, hidden, edges, 91+hidden)
        sparse = from_blocks(dense, edges)
        observations = jnp.asarray(np.random.default_rng(hidden).normal(
            0, .1, (128, inputs)).astype(np.float32))
        a, ac, at = time_scan(block_step, dense, observations, args.repeats)
        b, bc, bt = time_scan(edge_step, sparse, observations, args.repeats)
        error=float(np.max(np.abs(a-b)))
        if error > 2e-4:
            raise AssertionError(f'Sparse recurrent parity failed: {error}')
        report['cells'].append({'source':'synthetic','inputs':inputs,'hidden':hidden,'edges':edges,
            'steps':128,'dense_host_bytes':sum(np.asarray(v).nbytes for v in dense),
            'sparse_host_bytes':sum(np.asarray(v).nbytes for v in sparse),
            'dense_compile_seconds':ac,'sparse_compile_seconds':bc,
            'dense_warm_seconds':at,'sparse_warm_seconds':bt,
            'sparse_speedup':at/bt,'max_abs_output_error':error})
        print(json.dumps(report['cells'][-1]),flush=True)
    if args.artifact:
        dense,_=reload_artifact(args.artifact)
        if not isinstance(dense,BlockPlan):
            raise ValueError('Crossover artifact must contain role blocks')
        inputs,hidden=dense.w_in.shape
        edges=int(np.asarray(dense.mask_delay).sum())
        sparse = from_blocks(dense, edges)
        observations = jnp.asarray(np.random.default_rng(hidden).normal(
            0, .1, (128, inputs)).astype(np.float32))
        a, ac, at = time_scan(block_step, dense, observations, args.repeats)
        b, bc, bt = time_scan(edge_step, sparse, observations, args.repeats)
        error=float(np.max(np.abs(a-b)))
        if error > 2e-4:
            raise AssertionError(f'Sparse recurrent parity failed: {error}')
        report['cells'].append({'source':'archived-emr','artifact':args.artifact,
            'inputs':inputs,'hidden':hidden,'edges':edges,
            'steps':128,'dense_host_bytes':sum(np.asarray(v).nbytes for v in dense),
            'sparse_host_bytes':sum(np.asarray(v).nbytes for v in sparse),
            'dense_compile_seconds':ac,'sparse_compile_seconds':bc,
            'dense_warm_seconds':at,'sparse_warm_seconds':bt,
            'sparse_speedup':at/bt,'max_abs_output_error':error})
        print(json.dumps(report['cells'][-1]),flush=True)
    with open(args.output,'w') as f:
        json.dump(report,f,indent=2)


if __name__ == '__main__':
    main()
