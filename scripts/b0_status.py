"""Verify completed B0 cells and write a compact measured status file."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import numpy as np

from neuroevo.artifacts import canonical
from neuroevo.artifacts import reload as reload_phenotype
from neuroevo.adapters.weights import WeightAdapter


def read(path):
    return json.loads(path.read_text())


def spread(values):
    return {"median": median(values), "range": [min(values), max(values)]}


def cell(root, profile, task):
    directory = root / f"{profile}-{task}"
    pilot = read(directory / "pilot.json")
    experiment = read(directory / "experiment.json")
    replay = read(directory / "portable-replay.json")
    config = experiment["config"]
    digest = hashlib.sha256(canonical(config)).hexdigest()
    assert config["id"] == f"b0-{task}-acceptance-v1"
    assert pilot["passes"] and pilot["config_sha256"] == digest
    assert experiment["complete_protocol"] and len(experiment["runs"]) == 5
    assert [run["seed"] for run in experiment["runs"]] == config["seeds"] == [0, 1, 2, 3, 4]
    assert replay["status"] == "pass" and replay["steps"] == 128
    scores, walls, generalization = [], [], []
    rss_peaks, cgroup_peaks, swap_peaks, allocator_peaks = [], [], [], []
    rollout_signatures, tell_signatures = [], []
    adapter = WeightAdapter(config["adapter"])
    for run in experiment["runs"]:
        seed = run["seed"]
        seed_dir = directory / f"seed-{seed}"
        manifest = read(seed_dir / "manifest.json")
        history = read(seed_dir / "history.json")
        selection = read(seed_dir / "selection.json")
        assert run["generations"] == config["generations"]
        assert run["config_sha256"] == digest
        assert run["hardware"] == pilot["hardware"]
        assert run["provenance"] == pilot["provenance"]
        assert manifest == {"config": config, "seed": seed, "hardware": run["hardware"],
                            "provenance": run["provenance"], "config_sha256": digest}
        assert selection == run["selection"]
        assert [row["generation"] for row in history] == list(range(config["generations"]))
        assert all(row["valid"] == config["adapter"]["population"] for row in history)
        rollout_signatures.append(max(row["rollout_signatures"] for row in history))
        tell_signatures.append(max(row["tell_signatures"] for row in history))
        assert len(selection["episode_scores"]) == config["heldout_episodes"]
        assert np.isclose(selection["heldout_score"], np.mean(selection["episode_scores"]))
        assert selection["candidate_id"] in selection["finalist_ids"]
        assert len(selection["finalist_ids"]) == config["finalists"]
        assert all(candidate.startswith(f"seed-{seed}:{config['generations']-1}:")
                   for candidate in selection["finalist_ids"])
        for name in ("checkpoint.bin", "champion.npz", "champion-genes.npz",
                     "evaluated-winner.npz", "evaluated-winner-genes.npz",
                     "validation-finalists-genes.npz"):
            assert (seed_dir / name).stat().st_size > 0
        phenotype, metadata = reload_phenotype(seed_dir / "champion.npz")
        assert metadata["candidate_id"] == selection["candidate_id"]
        with np.load(seed_dir / "champion-genes.npz", allow_pickle=False) as genes:
            assert genes.files == ["weights"]
            decoded = adapter.decode(genes["weights"])
        for actual, expected in zip(phenotype, decoded):
            np.testing.assert_array_equal(actual, expected)
        assert run["selection"]["heldout_valid"]
        scores.append(run["selection"]["heldout_score"])
        walls.append(run["wall_seconds"])
        rss_peaks.append(run["memory"]["process_peak_rss_bytes"])
        cgroup_peaks.append(run["memory"]["cgroup_peak_bytes"])
        swap_peaks.append(run["memory"]["cgroup_swap_bytes"])
        if run["memory"]["device_allocator"] is not None:
            allocator_peaks.append(run["memory"]["device_allocator"]["peak_bytes_in_use"])
        if task == "memory":
            assert run["selection"]["generalization_valid"]
            generalization.append(run["selection"]["generalization_score"])
    assert experiment["passed_seeds"] == sum(s >= config["target"] for s in scores)
    assert replay["hardware"] == pilot["hardware"]
    assert replay["candidate_id"] == experiment["runs"][0]["selection"]["candidate_id"]
    return {
        "status": "measured", "config_sha256": digest,
        "image_digest": pilot["hardware"]["image_digest"],
        "engine_sha256": pilot["provenance"]["engine_sha256"],
        "pilot_passes": True, "pilot_warm_generation_seconds": pilot["warm_generation_seconds"],
        "pilot_allowable_generations": pilot["allowable_generations"],
        "generations": config["generations"], "population": config["adapter"]["population"],
        "heldout_score": spread(scores), "passed_seeds": experiment["passed_seeds"],
        "wall_seconds_per_seed": spread(walls), "portable_replay": "pass",
        "peak_process_rss_bytes": max(rss_peaks),
        "peak_cgroup_bytes": max(cgroup_peaks), "max_cgroup_swap_bytes": max(swap_peaks),
        "peak_device_allocator_bytes": max(allocator_peaks) if allocator_peaks else None,
        "max_rollout_signatures": max(rollout_signatures),
        "max_tell_signatures": max(tell_signatures),
        **({"generalization_score": spread(generalization)} if generalization else {}),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="/results/b0-v1")
parser.add_argument("--output", default="/reports/b0-status.json")
args = parser.parse_args()
root = Path(args.source)
cells = {profile: {task: cell(root, profile, task) for task in ("cartpole", "memory")}
         for profile in ("cpu", "rtx3070", "rtx3060")}
status = {"schema": 1, "method": "fixed-topology-weight-only", "cells": cells,
          "all_six_cells_measured": True,
          "all_learning_targets_met": all(c["passed_seeds"] >= 3 for p in cells.values() for c in p.values())}
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps({"status": "pass", "output": str(target), "all_learning_targets_met": status["all_learning_targets_met"]}))
