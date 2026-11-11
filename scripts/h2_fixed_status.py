"""Verify and summarize the frozen five-seed fixed-HyperNEAT acceptance cells."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from neuroevo.backend import initialize
hardware = initialize()

import numpy as np
from neuroevo.adapters.cppn import CPPNQuery
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.artifacts import canonical, reload
from neuroevo.fixed import FixedDecoder


def read(path):
    return json.loads(path.read_text())


def spread(values):
    return {"median": median(values), "range": [min(values), max(values)]}


def cell(root, task):
    directory = root / f"rtx3070-{task}"
    pilot = read(directory / "pilot.json")
    experiment = read(directory / "experiment.json")
    replay = read(directory / "portable-replay.json")
    config = experiment["config"]
    digest = hashlib.sha256(canonical(config)).hexdigest()
    assert config["id"] == f"h2-fixed-{task}-acceptance-v1"
    assert pilot["passes"] and pilot["config_sha256"] == digest
    assert pilot["hardware"] == hardware
    assert experiment["complete_protocol"] and len(experiment["runs"]) == 5
    assert [r["seed"] for r in experiment["runs"]] == config["seeds"] == [0, 1, 2, 3, 4]
    assert replay["status"] == "pass" and replay["steps"] == 128
    decoder = FixedDecoder(CPPNQuery(TensorNEATAdapter(config["adapter"])), config["decoder"])
    scores, walls, generalization, valid_counts = [], [], [], []
    for run in experiment["runs"]:
        seed = run["seed"]
        seed_dir = directory / f"seed-{seed}"
        manifest = read(seed_dir / "manifest.json")
        history = read(seed_dir / "history.json")
        selection = read(seed_dir / "selection.json")
        assert run["config_sha256"] == digest and run["generations"] == config["generations"]
        assert run["hardware"] == pilot["hardware"] and run["provenance"] == pilot["provenance"]
        assert manifest == {"config": config, "seed": seed, "hardware": run["hardware"],
                            "provenance": run["provenance"], "config_sha256": digest}
        assert selection == run["selection"]
        assert [row["generation"] for row in history] == list(range(config["generations"]))
        assert all(row["valid"] > 0 and row["phenotype_nodes_max"] ==
                   sum(config["decoder"][k] for k in ("inputs", "hidden", "outputs"))
                   and row["cppn_queries_total"] > 0 for row in history)
        valid_counts.extend(row["valid"] for row in history)
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
        phenotype, metadata = reload(seed_dir / "champion.npz")
        assert metadata["candidate_id"] == selection["candidate_id"]
        with np.load(seed_dir / "champion-genes.npz", allow_pickle=False) as genes:
            assert set(genes.files) == {"nodes", "conns"}
            decoded = decoder.decode(genes["nodes"], genes["conns"])
        for actual, expected in zip(phenotype, decoded):
            np.testing.assert_array_equal(actual, expected)
        assert selection["heldout_valid"]
        scores.append(selection["heldout_score"])
        walls.append(run["wall_seconds"])
        if task == "memory":
            assert selection["generalization_valid"]
            generalization.append(selection["generalization_score"])
    assert experiment["passed_seeds"] == sum(s >= config["target"] for s in scores)
    assert replay["candidate_id"] == experiment["runs"][0]["selection"]["candidate_id"]
    return {"status": "measured", "config_sha256": digest,
            "image_digest": pilot["hardware"]["image_digest"],
            "engine_sha256": pilot["provenance"]["engine_sha256"],
            "pilot_passes": True, "generations": config["generations"],
            "population": config["adapter"]["population"],
            "heldout_score": spread(scores), "seed_scores": scores,
            "passed_seeds": experiment["passed_seeds"],
            "wall_seconds_per_seed": spread(walls),
            "minimum_valid_candidates": min(valid_counts),
            "portable_replay": "pass",
            **({"generalization_score": spread(generalization)} if generalization else {})}


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="/results/h2-fixed-acceptance-v1")
parser.add_argument("--output", default="/reports/h2-fixed-status.json")
args = parser.parse_args()
cells = {task: cell(Path(args.source), task) for task in ("cartpole", "memory")}
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({"schema": 1, "method": "fixed-coordinate-CPPN",
                              "hardware": hardware, "cells": cells}, indent=2) + "\n")
print(json.dumps({"status": "pass", "output": str(target),
                  "passed_seeds": {task: row["passed_seeds"] for task, row in cells.items()}}))
