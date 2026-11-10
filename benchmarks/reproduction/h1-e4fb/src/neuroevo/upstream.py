"""Q2 screening harness. Uses the upstream generation loop unchanged."""
import json
from pathlib import Path
import time
import os
import threading
import jax
import numpy as np
import psutil
from .artifacts import write_json


def screen(manifest, seed, mode, output, hardware, smoke=False):
    from emr_hyperneat import EMRHyperNEAT
    from .runner import provenance
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"manifest.json", {"manifest": manifest, "hardware": hardware,
                                    "provenance": provenance(), "seed": seed, "mode": mode,
                                    "label": "smoke" if smoke else "Q2-screening"})
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
            state,metrics=algo.run_generation(state,Problem(),verbose=False)
            jax.block_until_ready(state)
            elapsed=time.perf_counter()-start
            cold_total=time.perf_counter()-begin if generation < cold_generations else cold_total
            if generation == cold_generations-1:
                watchdog.cancel()
            row={"generation":generation,"seconds":elapsed,"phase":"cold" if generation < cold_generations else "warm",
                 "rss_bytes":psutil.Process().memory_info().rss,
                 "device_memory":jax.devices()[0].memory_stats(),
                 "metrics":json.loads(json.dumps(metrics.__dict__,default=lambda x:np.asarray(x).tolist()))}
            rows.append(row)
            write_json(root/"generations.json",rows)
            print(json.dumps({"Q2":mode,"generation":generation,"seconds":elapsed}),flush=True)
            if generation < cold_generations and time.perf_counter()-begin > m["maximum_cold_seconds"]:
                raise RuntimeError("Q2 cold compile ceiling exceeded")
            if row["rss_bytes"] > m["host_memory_budget_bytes"]:
                raise RuntimeError("Q2 host-memory budget exceeded")
        warm=float(np.median([r["seconds"] for r in rows if r["phase"] == "warm"]))
        result={"status":"smoke-completed" if smoke else "measured",
                "warm_median_seconds":warm,"timing_pass":warm <= m["maximum_warm_median_seconds"],
                "cold_initialize_generation_seconds":cold_total,
                "resource_gate":"requires-profile-peak-memory-audit","hardware":hardware}
        write_json(root/"result.json",result)
    except Exception as error:
        write_json(root/"failure.json",{"status":"failed","type":type(error).__name__,"message":str(error)})
        raise
    finally:
        watchdog.cancel()
