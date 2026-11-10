import argparse
import json
import os
from pathlib import Path
import sys
import time


def main(hardware=None):
    if hardware is None:
        os.environ["NEUROEVO_PROCESS_STARTED"] = str(time.perf_counter())
        from .backend import initialize
        hardware = initialize()
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import pytest
        raise SystemExit(pytest.main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description="Container-only neuroevolution entry points")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("--output", required=True)
    t = sub.add_parser("test")
    t.add_argument("args", nargs=argparse.REMAINDER)
    for name in ("run", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--stop-after", type=int)
    p = sub.add_parser("oracle")
    p.add_argument("--output", required=True)
    p = sub.add_parser("upstream")
    p.add_argument("--manifest", default="configs/q2-v1.json")
    p.add_argument("--mode", choices=["feedforward", "recurrent"], required=True)
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--phenotype", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    from .artifacts import write_json
    if args.command == "test":
        import pytest
        raise SystemExit(pytest.main(args.args))
    if args.command == "qualify":
        from .qualification import qualify
        print(json.dumps(qualify(hardware, args.output), indent=2))
    elif args.command == "oracle":
        write_json(args.output, oracle(hardware))
    elif args.command == "replay":
        write_json(args.output, replay(args.phenotype, hardware))
    elif args.command == "upstream":
        from .upstream import screen
        screen(json.loads(Path(args.manifest).read_text()),args.seed,args.mode,args.output,hardware,args.smoke)
    else:
        from .runner import run, load_config
        config = load_config(args.config)
        if args.command == "benchmark":
            start = time.perf_counter()
            summary, history = run(config, args.seed if args.seed is not None else 900,
                                   Path(args.output)/"pilot-run", hardware, pilot=True, stop_after=12)
            import numpy as np
            warm = float(np.median([r["total_seconds"]-r["selection_seconds"] for r in history[2:]]))
            # Includes measured startup/compile, checkpoints and final validation/held-out selection.
            fixed = max(0., summary["wall_seconds"]-max(0,len(history)-2)*warm)
            allowable = .9*(config["budget_seconds_per_seed"]-fixed)/warm
            result = {**summary, "fixed_overhead_seconds": fixed, "warm_generation_seconds": warm,
                      "allowable_generations": allowable, "passes": bool(config["generations"] <= allowable),
                      "label": "G9-pilot-not-learning-acceptance", "pilot_seed": summary["seed"]}
            write_json(Path(args.output)/"pilot.json", result)
        else:
            summaries = []
            if args.seed is None and len(config["seeds"]) > 1:
                import subprocess
                for seed in config["seeds"]:
                    command = [sys.executable, "-m", "neuroevo", "run", "--config", args.config,
                               "--output", args.output, "--seed", str(seed)]
                    if args.resume:
                        command.append("--resume")
                    if args.stop_after is not None:
                        command.extend(["--stop-after",str(args.stop_after)])
                    subprocess.run(command, check=True)
                    directory = Path(args.output)/f"seed-{seed}"
                    summaries.append({**json.loads((directory/"summary.json").read_text()),
                                      "selection":json.loads((directory/"selection.json").read_text())})
                write_json(Path(args.output)/"experiment.json", {"config":config,"runs":summaries,
                    "passed_seeds":sum(r["selection"]["passes_learning"] for r in summaries),
                    "complete_protocol":args.stop_after is None})
                return
            for seed in ([args.seed] if args.seed is not None else config["seeds"]):
                result, _ = run(config, seed, Path(args.output)/f"seed-{seed}", hardware,
                                resume=args.resume, stop_after=args.stop_after)
                selection = json.loads((Path(args.output)/f"seed-{seed}"/"selection.json").read_text())
                summaries.append({**result, "selection": selection})
            write_json(Path(args.output)/"experiment.json", {"config": config, "runs": summaries,
                       "passed_seeds": sum(r["selection"]["passes_learning"] for r in summaries),
                       "complete_protocol": args.stop_after is None and len(summaries) == len(config["seeds"])})


def oracle(hardware):
    import jax.numpy as jnp
    import numpy as np
    from .environments import memory_oracle, memory_scenarios
    from .planning import plan
    from .contracts import stack_plans
    from .evaluation import memory_initial, memory_chunk
    p = stack_plans([plan(memory_oracle())])
    obs, cues, delays = memory_scenarios(0, "oracle", delays=[1,2,4,8,16], repeats=1, distractors=False)
    initial = memory_initial(p, len(cues))
    carried = memory_chunk(p, initial, obs, cues, delays, jnp.arange(17))
    reset = memory_chunk(p, initial, obs, cues, delays, jnp.arange(17), reset_each_step=True)
    accuracy, control = float(np.mean(carried.correct)), float(np.mean(reset.correct))
    if accuracy != 1 or control != .5:
        raise AssertionError(f"Oracle failure: {accuracy}, {control}")
    return {"status": "pass", "oracle_accuracy": accuracy, "reset_accuracy": control, "hardware": hardware,
            "episodes": len(cues), "delays": [1,2,4,8,16], "label": "runtime-correctness-only"}


def replay(path, hardware):
    # Portable inference: no evolution imports.
    import jax.numpy as jnp
    import numpy as np
    from .artifacts import reload
    from .runtime import compiled_step
    from .reference import step as reference_step
    from .planning import to_graph
    p, meta = reload(path)
    graph = to_graph(p, meta["profile"])
    state = jnp.zeros(len(p.node_mask), jnp.float32)
    rs = np.zeros(len(graph.node_ids), np.float64)
    actions, maximum_error = [], 0.
    # Fixed FP32 observation sequence exercises every input and 128-step state carry.
    for t in range(128):
        obs = np.full(len(p.inputs), np.float32(np.sin(t)*.25), np.float32)
        head, state = compiled_step(p, state, obs, profile=meta["profile"])
        rh, rs = reference_step(graph, rs, obs)
        maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(head)-rh))))
        np.testing.assert_allclose(head, rh, atol=1e-4, rtol=1e-4)
        action = int(np.argmax(head)) if meta["action"] == "argmax-first-tie" else int(head[0] >= 0)
        actions.append(action)
    if "tensorneat" in sys.modules or "emr_hyperneat" in sys.modules:
        raise AssertionError("Inference imported an evolution backend")
    return {"status": "pass", "steps": 128, "max_abs_error": maximum_error,
            "actions": actions, "hardware": hardware, "candidate_id": meta.get("candidate_id")}
