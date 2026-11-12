"""Synchronize live JAX values at pinned EMR generation timer boundaries.

This clock is installed only for a separate Q2 stage probe. It replaces the
``time`` name in EMR's generation module, not the process-wide time module.
The pinned feedforward path times stages in ``lap``; the general recurrent path
times them in ``run_generation_verbose``. Each call waits for every JAX array
reachable from that generation frame before returning a timestamp.
"""
import inspect
import time

import jax


class SynchronizedStageClock:
    def __init__(self):
        self.records = []

    def perf_counter(self):
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code.co_name == 'lap':
                frame = frame.f_back
            if frame.f_code.co_name in ('_feedforward_generation', 'run_generation_verbose'):
                begin = time.perf_counter()
                arrays = [leaf for leaf in jax.tree_util.tree_leaves(frame.f_locals)
                          if isinstance(leaf, jax.Array)]
                for array in arrays:
                    array.block_until_ready()
                self.records.append({'function': frame.f_code.co_name,
                                     'line': frame.f_lineno,
                                     'arrays_waited': len(arrays),
                                     'barrier_seconds': time.perf_counter() - begin})
            return time.perf_counter()
        finally:
            del frame

    def __getattr__(self, name):
        return getattr(time, name)
