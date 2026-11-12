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
from .artifacts import (canonical, atomic_write, write_json, export, save_checkpoint, load_checkpoint)
from .planning import plan
from .environments import cartpole, memory_scenarios
from .evaluation import (make_cartpole_rollout, evaluate_population, memory_chunk, memory_initial, visual_chunk)
from .contracts import stack_plans
from .packing import route as route_buckets
from .memory_schedule import choose_rollout_schedule
from .sparse_runtime import from_blocks
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
    if type(config.get("schema")) is not int or config["schema"] != 1 or config.get("task") not in ("cartpole", "cartpole-partial", "memory", "visual"):
        raise ValueError("Unsupported experiment schema/task")
    if config.get("kind") not in ("acceptance","smoke","diagnostic"):
        raise ValueError("Unsupported experiment kind")
    if config.get('history_policy') not in (None,'jsonl'):
        raise ValueError('Unsupported history persistence policy')
    memory_budget=config.get('rollout_memory_budget_bytes')
    if memory_budget is not None and (type(memory_budget) is not int or memory_budget < 1):
        raise ValueError('Invalid rollout memory budget')
    storage_budget=config.get('phenotype_storage_budget_bytes')
    if storage_budget is not None and (type(storage_budget) is not int or storage_budget < 1):
        raise ValueError('Invalid phenotype storage budget')
    discovery_microbatch=config.get('discovery_microbatch')
    if discovery_microbatch is not None and (type(discovery_microbatch) is not int or
            not 1 <= discovery_microbatch <= config['adapter']['population'] or
            config['adapter'].get('kind')!='adaptive-emr-v1'):
        raise ValueError('Invalid adaptive discovery microbatch')
    if config.get('runtime_layout','dense') not in ('dense','edge'):
        raise ValueError('Invalid runtime layout')
    if config.get('runtime_layout') == 'edge' and config['adapter'].get('kind') != 'adaptive-emr-v1':
        raise ValueError('Edge runtime requires the adaptive layered adapter')
    elapsed_stop=config.get('elapsed_stop_seconds')
    if elapsed_stop is not None and (type(elapsed_stop) not in (int,float) or
            not np.isfinite(elapsed_stop) or not 0 < elapsed_stop < config['budget_seconds_per_seed']):
        raise ValueError('Elapsed stop must be within the research budget')
    expected={"cartpole":(4,2,"argmax-first-tie"),"cartpole-partial":(4,2,"argmax-first-tie"),"memory":(2,1,"sign-positive-tie"),
              "visual":(25,1,"sign-positive-tie")}[config["task"]]
    baseline=config['adapter'].get('kind')=='fixed-weight-v1'
    adaptive=config['adapter'].get('kind')=='adaptive-emr-v1'
    if config['adapter'].get('kind') not in (None,'fixed-weight-v1','adaptive-emr-v1'):
        raise ValueError('Unknown evolution adapter')
    if baseline:
        from .adapters.weights import settings as weight_settings
        weight_settings(config['adapter'])
        if config.get('decoder') is not None:
            raise ValueError('Weight-only baseline cannot also have a CPPN decoder')
    decoder=config.get('decoder')
    if adaptive:
        from .adapters.emr import settings as adaptive_settings
        a=adaptive_settings(config['adaptive'])
        if (decoder is not None or config['adapter']['profile']!='layered' or
                config['adapter']['population']!=a['population'] or
                config['adapter']['inputs']!=len(a['input_coords']) or
                config['adapter']['outputs']!=len(a['output_coords'])):
            raise ValueError('Adaptive EMR task interface or decoder conflicts with A0')
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
        if adaptive and config['adaptive']['kind']!='adaptive-emr-visual-v2':
            raise ValueError('Visual adaptive decoder requires its 5x5 profile')
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
    adaptive=config['adapter'].get('kind')=='adaptive-emr-v1'
    if baseline:
        from .adapters.weights import WeightAdapter
        adapter=WeightAdapter(config['adapter'])
    elif adaptive:
        from .adapters.emr_evolution import EMREvolution
        task_inputs,task_outputs=config['adapter']['inputs'],config['adapter']['outputs']
        class Problem:
            input_shape=(task_inputs,)
            output_shape=(task_outputs,)
            jitable=True
            use_bias=False
            is_exhaustive_dataset=False
            def get_data(self):
                return [(np.zeros(task_inputs,np.float32).tolist(),
                         np.zeros(task_outputs,np.float32).tolist()),
                        (np.ones(task_inputs,np.float32).tolist(),
                         np.ones(task_outputs,np.float32).tolist())]
        adapter=EMREvolution(config['adaptive'],
            json.loads(Path('configs/q2-capacity-v2.json').read_text()),Problem(),
            node_capacity=config['adapter']['nodes'],edge_capacity=config['adapter']['edges'],
            phenotype_storage_budget_bytes=config.get('phenotype_storage_budget_bytes'),
            discovery_microbatch=config.get('discovery_microbatch'))
    else:
        from .adapters.tensorneat import TensorNEATAdapter
        adapter=TensorNEATAdapter(config["adapter"])
    decoder=None
    interpretation=(adapter.interpretation() if baseline else
                    {'adaptive_config':config['adaptive'],'source':'public-EMR-population-substrates'}
                    if adaptive else {})
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
    write_json(directory/"backend-settings.json", interpretation if baseline or adaptive else {
        "node_gene":settings(adapter.genome.node_gene),"conn_gene":settings(adapter.genome.conn_gene),
        "mutation":settings(adapter.genome.mutation),"input_transform":None,"output_transform":None,
        "network_type":adapter.genome.network_type,
        "gene_identity":"native-node-keys-and-connection-endpoint-pairs"})
    history, archive, previous_wall = [], None, 0.
    if resume:
        if adaptive:
            adapter.initialize(seed)  # Rebuild native compiled ask/tell before loading state.
        saved = load_checkpoint(directory/"checkpoint.bin", {**config, "run_seed": seed}, source)
        state, history, archive = saved["state"], saved["history"], saved["champion"]
        previous_wall = saved["elapsed_seconds"]
    else:
        state = adapter.initialize(seed)
    streaming_history=config.get('history_policy')=='jsonl'
    if streaming_history:
        # A checkpoint is the committed state. Discard any later JSONL rows on resume.
        atomic_write(directory/'history.jsonl',b''.join(canonical(row)+b'\n' for row in history))
    profile = 'layered' if decoder is not None or adaptive else config["adapter"]["profile"]
    rollout_schedules = {}
    def partitions(plans):
        if adaptive:
            if config.get('runtime_layout')=='edge':
                converted=[from_blocks(p,config['adapter']['edges']) for p in plans]
                groups={}
                for index,p in enumerate(converted):
                    sig=(p.w_in.shape,p.w_out.shape,len(p.weight))
                    groups.setdefault(sig,[]).append(index)
                return [(tuple(indices),[converted[i] for i in indices])
                        for _,indices in sorted(groups.items())]
            return [(indices,[plans[i] for i in indices])
                    for _,indices in route_buckets(plans)]
        return [(tuple(range(len(plans))),plans)]
    def batch_size(group, namespace, episodes):
        schedule=choose_rollout_schedule(group[0], episodes, config['microbatch'],
            config['task'],config.get('rollout_memory_budget_bytes'))
        signature=(str(group[0].w_in.shape[1]) if hasattr(group[0],'w_in')
                   else f'graph-{len(group[0].node_mask)}')
        if hasattr(group[0],'weight'):
            signature+=f'-e{len(group[0].weight)}'
        rollout_schedules.setdefault(namespace,{})[signature]={
            'microbatch':schedule.microbatch,'estimated_bytes':schedule.estimated_bytes,
            'budget_bytes':schedule.budget_bytes}
        return schedule.microbatch
    if config["task"] in ("cartpole", "cartpole-partial"):
        env, params = cartpole()
        observation_mode = 'position-angle' if config['task']=='cartpole-partial' else 'full'
        write_json(directory/"environment.json", {**{k:np.asarray(v).tolist() for k,v in vars(params).items()},
                    'observation_mode':observation_mode})
        evaluator = make_cartpole_rollout(env, params, config["horizon"], profile, observation_mode)
        shared = jnp.zeros((config["horizon"], 4), jnp.float32)
        def evaluate(plans, namespace, generation, episodes):
            keys = jnp.stack([key(seed, namespace, generation, e) for e in range(episodes)])
            scores=np.empty((len(plans),episodes),np.float32)
            valid=np.empty(len(plans),bool)
            for indices,part in partitions(plans):
                result=evaluate_population(part,evaluator,keys,shared,batch_size(part,namespace,episodes))
                scores[list(indices)]=result[0]
                valid[list(indices)]=result[4].all(axis=1)
            return scores.mean(axis=1), valid, scores
    elif config["task"] == "memory":
        def evaluate(plans, namespace, generation, episodes):
            delays = range(9, 17) if namespace == "generalization" else range(1, 9)
            obs, cues, ds = memory_scenarios(seed, namespace, generation, delays=delays, repeats=episodes//16)
            scores=np.empty((len(plans),episodes),np.float32)
            valid=np.empty(len(plans),bool)
            for indices,group in partitions(plans):
                microbatch=batch_size(group,namespace,episodes)
                for start in range(0,len(group),microbatch):
                    part=group[start:start+microbatch]
                    packed=stack_plans(part+[part[0]]*(microbatch-len(part)))
                    final=memory_chunk(packed,memory_initial(packed,episodes),obs,cues,ds,
                                       jnp.arange(17),profile=profile)
                    selected=list(indices[start:start+len(part)])
                    scores[selected]=np.asarray(final.correct[:len(part)])
                    valid[selected]=np.asarray(final.valid[:len(part)]).all(axis=1)
            return scores.mean(axis=1),valid,scores
    else:
        from .visual import visual_scenarios, TRAIN_POSITIONS, HELDOUT_POSITIONS
        write_json(directory/'environment.json', {'grid':[5,5], 'stroke_length':3,
            'train_positions':TRAIN_POSITIONS, 'heldout_positions':HELDOUT_POSITIONS,
            'background_noise_uniform':[-.15,.15], 'target':'vertical=+1; horizontal=-1',
            'one_step':True})
        def evaluate(plans, namespace, generation, episodes):
            obs, labels, _ = visual_scenarios(seed,namespace,generation,episodes)
            scores=np.empty((len(plans),episodes),np.float32)
            validity=np.empty(len(plans),bool)
            for indices,group in partitions(plans):
                microbatch=batch_size(group,namespace,episodes)
                for start in range(0,len(group),microbatch):
                    part=group[start:start+microbatch]
                    packed=stack_plans(part+[part[0]]*(microbatch-len(part)))
                    correct,valid=visual_chunk(packed,obs,labels,profile=profile)
                    selected=list(indices[start:start+len(part)])
                    scores[selected]=np.asarray(correct[:len(part)])
                    validity[selected]=np.asarray(valid[:len(part)])
            return scores.mean(axis=1),validity,scores
    limit = min(config["generations"], stop_after or config["generations"])
    stopped_for_elapsed = False
    for generation in range(int(state.generation), limit):
        rollout_schedules.clear()
        gen_start = time.perf_counter()
        batch = adapter.ask(state, f"seed-{seed}")
        plans, decode_valid, errors, decoded_stats = [], [], [], []
        if adaptive:
            plans=list(batch.plans)
            decode_valid=[True]*len(plans)
            errors=[None]*len(plans)
            positions=4*(4**(config['adaptive']['max_depth']+1)-1)//3
            def adaptive_stats(p):
                graph=p.graph()
                digest=hashlib.sha256(canonical({'grid_indices':p.grid_indices.tolist(),
                    'edges':graph.edges})).hexdigest()
                return {'phenotype_nodes':len(p.plan.node_mask),
                        'active_hidden_nodes':len(p.grid_indices),
                        'expressed_edges':len(graph.edges),
                        'recurrent_edges':int(p.plan.mask_delay.sum()),
                        'input_edges':int(p.plan.mask_in.sum()),
                        'output_edges':int(p.plan.mask_out.sum()),
                        'graph_sha256':digest,
                        'queries':positions*(config['adapter']['inputs']+config['adapter']['outputs'])+
                                  (len(p.grid_indices)**2 if config['adaptive']['recurrent'] else 0),
                        'host_array_bytes':sum(np.asarray(v).nbytes for v in p.plan[:8])}
            decoded_stats=[adaptive_stats(p) for p in batch.phenotypes]
        else:
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
        export(directory/"evaluated-winner.npz", batch.phenotypes[winner].plan if adaptive else plans[winner], profile,
               {"candidate_id": archive["id"], "training_score": archive["score"],
                "action": config["action"], "provenance": source,**interpretation})
        atomic_genes(directory/"evaluated-winner-genes.npz", archive)
        finalists = np.argsort(-scores, kind="stable")[:config["finalists"]]
        # Final selection is evaluated on the final evaluated generation, before tell.
        selection_seconds = 0.
        stopped_for_elapsed = bool(config.get('elapsed_stop_seconds') is not None and
            previous_wall+time.perf_counter()-started >= config['elapsed_stop_seconds'])
        final_generation = generation == limit-1 or stopped_for_elapsed
        if final_generation:
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
            export(directory/"champion.npz", batch.phenotypes[champion].plan if adaptive else plans[champion], profile,
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
               "rollout_signatures": (evaluator._cache_size() if config["task"] in ("cartpole", "cartpole-partial") else
                   memory_chunk._cache_size() if config["task"] == "memory" else visual_chunk._cache_size()),
               "tell_signatures": adapter._tell._cache_size(),
               "observer_signatures": None if baseline else adapter._observe._cache_size(),
               "rss_bytes": psutil.Process().memory_info().rss,
               "device_memory": jax.devices()[0].memory_stats(),
               "decode_errors": {str(i): e for i, e in enumerate(errors) if e}}
        row['rollout_schedules']={name:dict(schedules)
                                  for name,schedules in rollout_schedules.items()}
        if baseline:
            row.update(counts_kind='fixed-phenotype',genotype_parameters=adapter.parameters,
                       phenotype_nodes_max=adapter.i+adapter.h+adapter.o,
                       phenotype_edges_max=sum(adapter.sizes[:3]))
        if decoded_stats:
            row.update(phenotype_nodes_max=max(s['phenotype_nodes'] for s in decoded_stats),
                       phenotype_edges_max=max(s['expressed_edges'] for s in decoded_stats),
                       cppn_queries_total=sum(s['queries'] for s in decoded_stats),
                       phenotype_host_bytes_total=sum(s['host_array_bytes'] for s in decoded_stats))
        if adaptive:
            row['phenotype_host_storage_bytes']=batch.host_storage_bytes
            row['adaptive_graphs']=[{k:s[k] for k in ('active_hidden_nodes','input_edges',
                'output_edges','recurrent_edges','expressed_edges','graph_sha256')}
                for s in decoded_stats]
            row['bucket_signatures']=[{'inputs':sig[0],'hidden_width':sig[1],
                'outputs':sig[2],'candidates':len(indices)}
                for sig,indices in route_buckets(plans)]
        history.append(row)
        row["total_seconds"] = time.perf_counter()-gen_start
        if int(state.generation) % config["checkpoint_every"] == 0 or final_generation:
            save_checkpoint(directory/"checkpoint.bin", state, {**config, "run_seed": seed}, source, history, archive,
                            previous_wall+time.perf_counter()-started)
        row["total_seconds"] = time.perf_counter()-gen_start
        if streaming_history:
            # Append one bounded record; avoid rewriting the whole history on long runs.
            with (directory/'history.jsonl').open('ab') as stream:
                stream.write(canonical(row)+b'\n')
                stream.flush()
                os.fsync(stream.fileno())
        else:
            # Atomic bounded history makes interrupted short runs reviewable.
            write_json(directory/"history.json", history)
        print(json.dumps({k: row[k] for k in ("generation", "best", "mean", "nodes_max", "total_seconds")}), flush=True)
        if not streaming_history:
            row["total_seconds"] = time.perf_counter()-gen_start
        if previous_wall+time.perf_counter()-started > config["budget_seconds_per_seed"]:
            write_json(directory/"budget-failure.json", {"status": "over-budget", "generation": generation})
            raise RuntimeError("Research budget exceeded")
        if final_generation:
            break
    write_json(directory/"history.json", history)
    from .resources import memory_record
    summary = {"seed": seed, "generations": int(state.generation), "kind": config["kind"],
               "wall_seconds": previous_wall+time.perf_counter()-started, "provenance": source,
               "hardware": hardware, "config_sha256": config_hash, "memory": memory_record(),
               "stop_reason": "elapsed" if stopped_for_elapsed else "generation-cap",
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
