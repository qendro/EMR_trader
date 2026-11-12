"""Q2 screening harness. Uses the upstream generation loop unchanged."""
import json
from pathlib import Path
import time
import os
import resource
import threading
import jax
import numpy as np
import psutil
from .artifacts import write_json


def _cgroup_value(name):
    path=Path('/sys/fs/cgroup')/name
    try:
        value=path.read_text().strip()
        return None if value == 'max' else int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _resources():
    # ru_maxrss is KiB on the pinned Linux image. Cgroup peak includes cache and
    # other processes in the container, so retain both rather than conflating them.
    result={'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'cgroup_memory_current_bytes':_cgroup_value('memory.current'),
            'cgroup_memory_peak_bytes':_cgroup_value('memory.peak'),
            'cgroup_swap_current_bytes':_cgroup_value('memory.swap.current')}
    try:
        lines=Path('/proc/meminfo').read_text().splitlines()
        fields={line.split(':',1)[0]:int(line.split(':',1)[1].split()[0])*1024 for line in lines}
        result['host_mem_available_bytes']=fields['MemAvailable']
        result['host_swap_used_bytes']=fields['SwapTotal']-fields['SwapFree']
    except (FileNotFoundError, KeyError, ValueError, IndexError):
        result['host_mem_available_bytes']=None
        result['host_swap_used_bytes']=None
    return result


def _population_sizes(algo, population, mode):
    # These caches are the exact substrate evaluated by the pinned upstream
    # generation. Refuse an API/layout change instead of reporting partial sizes.
    mask=getattr(algo,'_cached_masks_A',None)
    w1=getattr(algo,'_cached_W1',None)
    w2=getattr(algo,'_cached_W2',None)
    if mask is None or w1 is None or w2 is None:
        raise RuntimeError('Pinned EMR substrate cache unavailable for Q2 size audit')
    mask=np.asarray(jax.device_get(mask),bool)
    w1=np.asarray(jax.device_get(w1))
    w2=np.asarray(jax.device_get(w2))
    if (mask.shape[0]!=population or w1.shape[0]!=population or w2.shape[0]!=population or
            mask.ndim!=2 or w1.ndim!=3 or w2.ndim!=3 or
            w1.shape[2]!=mask.shape[1] or w2.shape[1]!=mask.shape[1]):
        raise RuntimeError('Pinned EMR substrate cache shape changed')
    nodes=mask.sum(axis=1).astype(np.int64)
    input_edges=np.count_nonzero(w1,axis=(1,2)).astype(np.int64)
    output_edges=np.count_nonzero(w2,axis=(1,2)).astype(np.int64)
    sparse=getattr(algo,'_cached_sparse_hh',None)
    if mode=='recurrent' and sparse is None:
        raise RuntimeError('Pinned EMR recurrent connection cache unavailable')
    recurrent_edges=(np.asarray(jax.device_get(sparse.num_valid),np.int64) if sparse is not None
                     else np.zeros(population,np.int64))
    if recurrent_edges.shape!=(population,):
        raise RuntimeError('Pinned EMR recurrent connection count shape changed')
    return {'candidate_positions':int(mask.shape[1]),
            'active_nodes_per_genome':nodes.tolist(),
            'input_edges_per_genome':input_edges.tolist(),
            'output_edges_per_genome':output_edges.tolist(),
            'recurrent_edges_per_genome':recurrent_edges.tolist(),
            'total_edges_per_genome':(input_edges+output_edges+recurrent_edges).tolist()}


def screen(manifest, seed, mode, output, hardware, smoke=False, stage_probe=False):
    from emr_hyperneat import EMRHyperNEAT
    from .runner import provenance
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"manifest.json", {"manifest": manifest, "hardware": hardware,
                                    "provenance": provenance(), "seed": seed, "mode": mode,
                                    "label": "stage-probe-smoke" if stage_probe and smoke else
                                             "Q2-stage-probe" if stage_probe else
                                             "smoke" if smoke else "Q2-screening"})
    ds = manifest["dataset"]
    inputs = np.random.Generator(np.random.PCG64(ds["seed"])).uniform(-1,1,(ds["samples"],4)).astype(np.float32)
    targets = (inputs[:,:2]+np.float32(1))/np.float32(2)
    np.savez(root/"dataset.npz",inputs=inputs,targets=targets)
    class Problem:
        input_shape=(4,)
        output_shape=(2,)
        jitable=True
        use_bias=False
        is_exhaustive_dataset=False
        def get_data(self):
            return list(zip(inputs.tolist(), targets.tolist()))
    m=manifest
    d=m["discovery"]
    emr={**{k:v for k,v in d.items() if k != "execution_mode"},
         "execution":{"mode":d["execution_mode"]},
         "dynamic_functions":{"mode":"disabled"},"aggregation":{"mode":"disabled"},
         "recurrence":({**m["recurrence"]} if mode == "recurrent" else
                       {"enabled":False,"allow_hidden_to_hidden":False,"hh_cache_enabled":False}),
         "verbose":False}
    if smoke:
        emr["max_depth"]=1
    c=m["cppn"]
    config={"algorithm_params":{"emrhyperneat":{
        "population_size":32 if smoke else m["population"],"emr_hyperneat":emr,
        "substrate":{"input_coords":m["input_coords"],"output_coords":m["output_coords"],
                     "activate_time":m["substrate"]["settling_steps"],
                     "hidden_activation":"tanh","output_activation":"sigmoid"},
        "neat_mutation":c["mutation"],"neat_species":c["species"],"neat_selection":c["selection"]}}}
    algo=EMRHyperNEAT()
    if stage_probe:
        from emr_hyperneat.engine import generation_mixin
        from .stage_probe import SynchronizedStageClock
        stage_clock=SynchronizedStageClock()
        original_generation_time=generation_mixin.time
    def timeout():
        write_json(root/"failure.json", {"status":"over-budget", "phase":"cold-initialize-first-generation",
                                        "maximum_seconds":m["maximum_cold_seconds"]})
        os._exit(124)
    watchdog=threading.Timer(m["maximum_cold_seconds"],timeout)
    watchdog.daemon=True
    watchdog.start()
    try:
        begin=time.perf_counter()
        state=algo.initialize(algo.create_config(config),Problem(),seed=seed)
        rows=[]
        cold_generations=1 if smoke else m["cold_generations"]
        for generation in range(2 if smoke else cold_generations+m["warm_generations"]):
            start=time.perf_counter()
            if stage_probe:
                stage_clock.records=[]
                generation_mixin.time=stage_clock
            try:
                state,metrics=algo.run_generation(state,Problem(),verbose=False)
            finally:
                if stage_probe:
                    generation_mixin.time=original_generation_time
            stage_sync_records=list(stage_clock.records) if stage_probe else None
            jax.block_until_ready(state)
            elapsed=time.perf_counter()-start
            cold_total=time.perf_counter()-begin if generation < cold_generations else cold_total
            if generation == cold_generations-1:
                watchdog.cancel()
            observation_start=time.perf_counter()
            sizes=_population_sizes(algo,32 if smoke else m['population'],mode)
            resources=_resources()
            device_memory=jax.devices()[0].memory_stats()
            metric_values=json.loads(json.dumps(metrics.__dict__,default=lambda x:np.asarray(x).tolist()))
            observation_seconds=time.perf_counter()-observation_start
            row={"generation":generation,"seconds":elapsed,"phase":"cold" if generation < cold_generations else "warm",
                 "rss_bytes":psutil.Process().memory_info().rss,
                 "device_memory":device_memory,
                 "resources":resources,"population_sizes":sizes,
                 "observation_seconds_excluding_persistence":observation_seconds,
                 "stage_timing_status":("synchronized-live-array-boundaries" if stage_probe else
                                        "upstream-dispatch-times-not-isolated-synchronized-stages"),
                 "stage_sync_records":stage_sync_records,
                 "metrics":metric_values}
            rows.append(row)
            write_json(root/"generations.json",rows)
            print(json.dumps({"Q2":mode,"generation":generation,"seconds":elapsed}),flush=True)
            if generation < cold_generations and time.perf_counter()-begin > m["maximum_cold_seconds"]:
                raise RuntimeError("Q2 cold compile ceiling exceeded")
            if max(row['rss_bytes'],resources['process_peak_rss_bytes'],
                   resources['cgroup_memory_peak_bytes'] or 0) > m["host_memory_budget_bytes"]:
                raise RuntimeError("Q2 host-memory budget exceeded")
        warm=float(np.median([r["seconds"] for r in rows if r["phase"] == "warm"]))
        result={"status":"smoke-completed" if smoke else "measured",
                "warm_median_seconds":warm,"timing_pass":warm <= m["maximum_warm_median_seconds"],
                "cold_initialize_generation_seconds":cold_total,
                "resource_gate":"requires-external-GPU-and-peak-memory-audit",
                "stage_gate":("synchronized-probe-measured-pending-audit" if stage_probe else
                              "requires-synchronized-isolated-stage-probe"),
                "peak_process_rss_bytes":max(r['resources']['process_peak_rss_bytes'] for r in rows),
                "peak_cgroup_memory_bytes":max(r['resources']['cgroup_memory_peak_bytes'] or 0 for r in rows),
                "peak_device_pool_bytes":max((r['device_memory'] or {}).get('peak_pool_bytes',0) for r in rows),
                "hardware":hardware}
        write_json(root/"result.json",result)
    except Exception as error:
        write_json(root/"failure.json",{"status":"failed","type":type(error).__name__,"message":str(error)})
        raise
    finally:
        watchdog.cancel()
