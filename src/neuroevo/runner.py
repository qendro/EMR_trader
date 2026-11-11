"""V5: bounded generation controller, archive before tell, selection without test leakage."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from .artifacts import (canonical, write_json, export, save_checkpoint, load_checkpoint)
from .planning import plan
from .environments import cartpole, memory_scenarios
from .evaluation import (make_cartpole_rollout, evaluate_population, memory_chunk, memory_initial, visual_chunk)
from .contracts import stack_plans
from .randomness import key


def provenance():
    return {"sources": json.loads(Path("locks/sources.json").read_text()),
            "lock_sha256": hashlib.sha256(Path("locks/requirements.txt").read_bytes()).hexdigest(),
            "compatibility_lock_sha256": hashlib.sha256(Path("locks/compatibility.txt").read_bytes()).hexdigest(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("jax", "jaxlib", "numpy", "tensorneat", "emr-hyperneat", "gymnax")},
            "engine_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in
                sorted(Path(__file__).parent.rglob("*.py")))).hexdigest()}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if type(config.get("schema")) is not int or config["schema"] != 1 or config.get("task") not in ("cartpole", "memory", "visual"):
        raise ValueError("Unsupported experiment schema/task")
    if config.get("kind") not in ("acceptance","smoke","diagnostic"):
        raise ValueError("Unsupported experiment kind")
    expected={"cartpole":(4,2,"argmax-first-tie"),"memory":(2,1,"sign-positive-tie"),
              "visual":(25,1,"sign-positive-tie")}[config["task"]]
    baseline=config['adapter'].get('kind')=='fixed-weight-v1'
    if config['adapter'].get('kind') not in (None,'fixed-weight-v1'):
        raise ValueError('Unknown evolution adapter')
    if baseline:
        from .adapters.weights import settings as weight_settings
        weight_settings(config['adapter'])
        if config.get('decoder') is not None:
            raise ValueError('Weight-only baseline cannot also have a CPPN decoder')
    decoder=config.get('decoder')
    if decoder is not None:
        from .fixed import settings
        decoder=settings(decoder)
        if (config['adapter']['inputs'],config['adapter']['outputs'],config['adapter']['profile'])!=(4,1,'dag'):
            raise ValueError('Fixed decoder requires a four-input one-output DAG CPPN')
    interface=decoder if decoder is not None else config['adapter']
    if (interface['inputs'],interface['outputs'],config["action"]) != expected:
        raise ValueError("Unsupported task interface or action mapping")
    seeds=config.get("seeds",[])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds):
        raise ValueError("Invalid seed namespace")
    if type(config["target"]) not in (int,float) or not np.isfinite(config["target"]):
        raise ValueError("Invalid learning target")
    for field in (("population","inputs","outputs") if baseline else ("population","nodes","edges","inputs","outputs")):
        if type(config["adapter"][field]) is not int or config["adapter"][field] <= 0:
            raise ValueError(f"Invalid adapter {field}")
    for field in ("generations", "episodes", "horizon", "microbatch", "validation_episodes",
                  "heldout_episodes", "finalists", "checkpoint_every", "budget_seconds_per_seed"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"Invalid {field}")
    if not 1 <= config["finalists"] <= config["adapter"]["population"]:
        raise ValueError("Invalid finalist count")
    if config["scenario_policy"] not in ("fixed", "generation-specific"):
        raise ValueError("Unknown comparability policy")
    if config["task"] == "memory" and (config["episodes"], config["horizon"]) != (32, 17):
        raise ValueError("Shared memory protocol requires 32 episodes and horizon 17")
    if config["task"] == "visual":
        from .visual import input_coordinates
        if (config['kind']!='diagnostic' or config['horizon']!=1 or
                any(config[f] % n for f,n in (('episodes',10),('validation_episodes',10),('heldout_episodes',8)))):
            raise ValueError('Visual diagnostic requires one step and balanced scenarios')
        if decoder is not None and (decoder['kind']!='fixed-coordinate-v2' or
                                    decoder['input_coordinates']!=input_coordinates()):
            raise ValueError('Visual fixed decoder requires the frozen 5x5 coordinates')
    return config


def run(config, seed, directory, hardware, *, resume=False, stop_after=None, pilot=False):
    started = float(os.environ.get("NEUROEVO_PROCESS_STARTED", time.perf_counter()))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = provenance()
    config_hash = hashlib.sha256(canonical(config)).hexdigest()
    if config["kind"] == "acceptance" and not pilot:
        qualification = json.loads((directory.parent/"pilot.json").read_text())
        if (qualification["config_sha256"] != config_hash or qualification["provenance"] != source
                or qualification["hardware"] != hardware or not qualification["passes"]):
            raise ValueError("Acceptance requires a matching successful integrated G9 pilot")
    write_json(directory/"manifest.json", {"config": config, "seed": seed, "hardware": hardware,
                                          "provenance": source, "config_sha256": config_hash})
    cache_dir=os.environ.get("JAX_COMPILATION_CACHE_DIR")
    write_json(directory/"runtime.json", {"compilation_cache_dir":cache_dir,
        "cache_entries_at_start":len(list(Path(cache_dir).glob("*-cache"))) if cache_dir else 0,
        "jit_enabled":not jax.config.jax_disable_jit,
        "matmul_precision":jax.config.jax_default_matmul_precision,
        "jax_enable_x64":jax.config.jax_enable_x64,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "planning":"host-NumPy-once-per-generation","rollout":"compiled-JAX-scan"})
    baseline=config['adapter'].get('kind')=='fixed-weight-v1'
    if baseline:
        from .adapters.weights import WeightAdapter
        adapter=WeightAdapter(config['adapter'])
    else:
        from .adapters.tensorneat import TensorNEATAdapter
        adapter=TensorNEATAdapter(config["adapter"])
    decoder=None
    interpretation=adapter.interpretation() if baseline else {}
    def genes(batch,index):
        return {'weights':batch.weights[index]} if baseline else {'nodes':batch.nodes[index],'conns':batch.conns[index]}
    if config.get('decoder') is not None:
        from .adapters.cppn import CPPNQuery
        from .fixed import FixedDecoder
        decoder=FixedDecoder(CPPNQuery(adapter),config['decoder'])
        interpretation=decoder.interpretation()
        write_json(directory/'decoder.json',interpretation)
    def settings(obj):
        def value(v):
            if isinstance(v,(tuple,list)):
                return [value(x) for x in v]
            if callable(v):
                return getattr(v,"__name__",type(v).__name__)
            if isinstance(v,(str,bool,int,float)) or v is None:
                return v
            return np.asarray(v).tolist()
        return {k:value(v) for k,v in vars(obj).items() if not k.startswith("_")}
    write_json(directory/"backend-settings.json", interpretation if baseline else {
        "node_gene":settings(adapter.genome.node_gene),"conn_gene":settings(adapter.genome.conn_gene),
        "mutation":settings(adapter.genome.mutation),"input_transform":None,"output_transform":None,
        "network_type":adapter.genome.network_type,
        "gene_identity":"native-node-keys-and-connection-endpoint-pairs"})
    history, archive, previous_wall = [], None, 0.
    if resume:
        saved = load_checkpoint(directory/"checkpoint.bin", {**config, "run_seed": seed}, source)
        state, history, archive = saved["state"], saved["history"], saved["champion"]
        previous_wall = saved["elapsed_seconds"]
    else:
        state = adapter.initialize(seed)
    profile = 'layered' if decoder is not None else config["adapter"]["profile"]
    if config["task"] == "cartpole":
        env, params = cartpole()
        write_json(directory/"environment.json", {k:np.asarray(v).tolist() for k,v in vars(params).items()})
        evaluator = make_cartpole_rollout(env, params, config["horizon"], profile)
        shared = jnp.zeros((config["horizon"], 4), jnp.float32)
        def evaluate(plans, namespace, generation, episodes):
            keys = jnp.stack([key(seed, namespace, generation, e) for e in range(episodes)])
            result = evaluate_population(plans, evaluator, keys, shared, config["microbatch"])
            return result[0].mean(axis=1), result[4].all(axis=1), result[0]
    elif config["task"] == "memory":
        def evaluate(plans, namespace, generation, episodes):
            delays = range(9, 17) if namespace == "generalization" else range(1, 9)
            obs, cues, ds = memory_scenarios(seed, namespace, generation, delays=delays, repeats=episodes//16)
            summaries, validity = [], []
            for start in range(0, len(plans), config["microbatch"]):
                part = plans[start:start+config["microbatch"]]
                packed = stack_plans(part+[part[0]]*(config["microbatch"]-len(part)))
                final = memory_chunk(packed, memory_initial(packed, episodes), obs, cues, ds,
                                     jnp.arange(17), profile=profile)
                summaries.extend(np.asarray(final.correct[:len(part)]))
                validity.extend(np.asarray(final.valid[:len(part)]).all(axis=1))
            scores = np.asarray(summaries, np.float32)
            return scores.mean(axis=1), np.array(validity), scores
    else:
        from .visual import visual_scenarios, TRAIN_POSITIONS, HELDOUT_POSITIONS
        write_json(directory/'environment.json', {'grid':[5,5], 'stroke_length':3,
            'train_positions':TRAIN_POSITIONS, 'heldout_positions':HELDOUT_POSITIONS,
            'background_noise_uniform':[-.15,.15], 'target':'vertical=+1; horizontal=-1',
            'one_step':True})
        def evaluate(plans, namespace, generation, episodes):
            obs, labels, _ = visual_scenarios(seed,namespace,generation,episodes)
            summaries, validity = [], []
            for start in range(0,len(plans),config['microbatch']):
                part=plans[start:start+config['microbatch']]
                packed=stack_plans(part+[part[0]]*(config['microbatch']-len(part)))
                correct, valid=visual_chunk(packed,obs,labels,profile=profile)
                summaries.extend(np.asarray(correct[:len(part)]))
                validity.extend(np.asarray(valid[:len(part)]))
            scores=np.asarray(summaries,np.float32)
            return scores.mean(axis=1),np.asarray(validity,bool),scores
    limit = min(config["generations"], stop_after or config["generations"])
    for generation in range(int(state.generation), limit):
        gen_start = time.perf_counter()
        batch = adapter.ask(state, f"seed-{seed}")
        plans, decode_valid, errors, decoded_stats = [], [], [], []
        candidates=((w,None) for w in batch.weights) if baseline else zip(batch.nodes,batch.conns)
        for nodes, conns in candidates:
            try:
                p = (adapter.decode(nodes) if baseline else decoder.decode(nodes,conns) if decoder is not None else
                     plan(adapter.decode(nodes, conns), config["adapter"]["nodes"], config["adapter"]["edges"]))
                plans.append(p)
                decode_valid.append(True)
                errors.append(None)
                if decoder is not None:
                    decoded_stats.append(dict(decoder.last_stats))
            except ValueError as error:
                plans.append(None)
                decode_valid.append(False)
                errors.append(str(error))
        if not any(decode_valid):
            write_json(directory/"failure.json", {"generation": generation, "errors": errors})
            raise ValueError("Entire population failed decoding")
        fallback = next(p for p in plans if p is not None)
        plans = [fallback if p is None else p for p in plans]
        decode_seconds = time.perf_counter()-gen_start
        eval_start = time.perf_counter()
        scenario_generation = generation if config["scenario_policy"] == "generation-specific" else 0
        scores, valid, _ = evaluate(plans, "train", scenario_generation, config["episodes"])
        rollout_seconds = time.perf_counter()-eval_start
        valid &= np.array(decode_valid) & np.isfinite(scores)
        scores = np.where(valid, scores, -np.inf)
        if not valid.any():
            raise ValueError("All rollouts invalid")
        winner = int(np.argmax(scores))
        # Persist evaluated genes AND phenotype before reproduction can change slot contents.
        archive = {"id": batch.ids[winner], "score": float(scores[winner]),**genes(batch,winner)}
        export(directory/"evaluated-winner.npz", plans[winner], profile,
               {"candidate_id": archive["id"], "training_score": archive["score"],
                "action": config["action"], "provenance": source,**interpretation})
        atomic_genes(directory/"evaluated-winner-genes.npz", archive)
        finalists = np.argsort(-scores, kind="stable")[:config["finalists"]]
        # Final selection is evaluated on the final evaluated generation, before tell.
        selection_seconds = 0.
        if generation == limit-1:
            selection_start = time.perf_counter()
            atomic_genes(directory/"validation-finalists-genes.npz",genes(batch,finalists))
            vs, vv, _ = evaluate([plans[i] for i in finalists], "validation", 0, config["validation_episodes"])
            vs = np.where(vv & valid[finalists], vs, -np.inf)
            if not np.isfinite(vs).any():
                raise ValueError("All validation candidates invalid")
            champion = int(finalists[np.argmax(vs)])
            hs, hv, episode_scores = evaluate([plans[champion]], "interpolation", 0, config["heldout_episodes"])
            result = {"candidate_id": batch.ids[champion], "validation_score": float(np.max(vs)),
                      "finalist_ids":[batch.ids[i] for i in finalists],
                      "finalist_training_scores":[float(scores[i]) if valid[i] else None for i in finalists],
                      "finalist_validation_scores":[float(v) if np.isfinite(v) else None for v in vs],
                      "heldout_score": float(hs[0]), "heldout_valid": bool(hv[0]),
                      "episode_scores": episode_scores[0].tolist(), "kind": config["kind"],
                      "passes_learning": bool(hv[0] and hs[0] >= config["target"])}
            if config["task"] == "memory":
                gs, gv, ge = evaluate([plans[champion]], "generalization", 0, config["heldout_episodes"])
                result.update(generalization_score=float(gs[0]), generalization_valid=bool(gv[0]),
                              generalization_by_delay=ge[0].reshape(8, -1).mean(axis=1).tolist(),
                              interpolation_by_delay=episode_scores[0].reshape(8, -1).mean(axis=1).tolist())
            export(directory/"champion.npz", plans[champion], profile,
                   {"action": config["action"], "candidate_id": batch.ids[champion], "provenance": source,**interpretation})
            atomic_genes(directory/"champion-genes.npz",genes(batch,champion))
            write_json(directory/"selection.json", result)
            selection_seconds = time.perf_counter()-selection_start
        tell_start = time.perf_counter()
        state, removed = adapter.tell(state, batch, batch.ids, scores)
        evolve_seconds = time.perf_counter()-tell_start
        row = {"generation": generation, "best": float(scores[valid].max()),
               "mean": float(scores[valid].mean()), "valid": int(valid.sum()),
               "species": None if baseline else int(np.isfinite(np.asarray(state.species.species_keys)).sum()),
               "stagnation_removed": removed,
               "nodes_max": adapter.i+adapter.h+adapter.o if baseline else int((~np.isnan(batch.nodes[:, :, 0])).sum(axis=1).max()),
               "edges_max": sum(adapter.sizes[:3]) if baseline else int((~np.isnan(batch.conns[:, :, 0])).sum(axis=1).max()),
               "decode_plan_seconds": decode_seconds, "rollout_seconds": rollout_seconds,
               "evolve_seconds": evolve_seconds, "selection_seconds": selection_seconds,
               "rollout_signatures": (evaluator._cache_size() if config["task"] == "cartpole" else
                   memory_chunk._cache_size() if config["task"] == "memory" else visual_chunk._cache_size()),
               "tell_signatures": adapter._tell._cache_size(),
               "observer_signatures": None if baseline else adapter._observe._cache_size(),
               "rss_bytes": psutil.Process().memory_info().rss,
               "device_memory": jax.devices()[0].memory_stats(),
               "decode_errors": {str(i): e for i, e in enumerate(errors) if e}}
        if baseline:
            row.update(counts_kind='fixed-phenotype',genotype_parameters=adapter.parameters,
                       phenotype_nodes_max=adapter.i+adapter.h+adapter.o,
                       phenotype_edges_max=sum(adapter.sizes[:3]))
        if decoded_stats:
            row.update(phenotype_nodes_max=max(s['phenotype_nodes'] for s in decoded_stats),
                       phenotype_edges_max=max(s['expressed_edges'] for s in decoded_stats),
                       cppn_queries_total=sum(s['queries'] for s in decoded_stats),
                       phenotype_host_bytes_total=sum(s['host_array_bytes'] for s in decoded_stats))
        history.append(row)
        row["total_seconds"] = time.perf_counter()-gen_start
        if int(state.generation) % config["checkpoint_every"] == 0 or generation == limit-1:
            save_checkpoint(directory/"checkpoint.bin", state, {**config, "run_seed": seed}, source, history, archive,
                            previous_wall+time.perf_counter()-started)
        row["total_seconds"] = time.perf_counter()-gen_start
        # Atomic bounded history also makes interrupted generations reviewable.
        write_json(directory/"history.json", history)
        print(json.dumps({k: row[k] for k in ("generation", "best", "mean", "nodes_max", "total_seconds")}), flush=True)
        row["total_seconds"] = time.perf_counter()-gen_start
        if previous_wall+time.perf_counter()-started > config["budget_seconds_per_seed"]:
            write_json(directory/"budget-failure.json", {"status": "over-budget", "generation": generation})
            raise RuntimeError("Research budget exceeded")
    write_json(directory/"history.json", history)
    from .resources import memory_record
    summary = {"seed": seed, "generations": int(state.generation), "kind": config["kind"],
               "wall_seconds": previous_wall+time.perf_counter()-started, "provenance": source,
               "hardware": hardware, "config_sha256": config_hash, "memory": memory_record(),
               "structural_mutation_observed": False if baseline else any(r["nodes_max"] > config["adapter"]["inputs"]+config["adapter"]["outputs"] for r in history)}
    write_json(directory/"summary.json", summary)
    return summary, history


def atomic_genes(path, archive):
    import io
    from .artifacts import atomic_write
    buf = io.BytesIO()
    arrays=({'weights':archive['weights']} if 'weights' in archive else
            {'nodes':archive['nodes'],'conns':archive['conns']})
    np.savez_compressed(buf,**arrays)
    atomic_write(path, buf.getvalue())
