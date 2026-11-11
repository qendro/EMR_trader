"""Check the complete direct recurrent memory system-comparison cell."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from neuroevo.artifacts import canonical, reload


def read(path):
    return json.loads(path.read_text())


def spread(values):
    return {"median": median(values), "range": [min(values), max(values)]}


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="/results/h2-direct-memory-acceptance-v1/rtx3070-memory")
parser.add_argument("--output", default="/reports/h2-direct-status.json")
args = parser.parse_args()
root = Path(args.source)
pilot = read(root / "pilot.json")
experiment = read(root / "experiment.json")
replay = read(root / "task-portable-replay.json")
synthetic_replay = read(root / "synthetic-replay-audit.json")
config = experiment["config"]
digest = hashlib.sha256(canonical(config)).hexdigest()
assert config["id"] == "h2-direct-memory-acceptance-v1"
assert pilot["passes"] and pilot["config_sha256"] == digest
assert experiment["complete_protocol"] and len(experiment["runs"]) == 5
assert [r["seed"] for r in experiment["runs"]] == config["seeds"] == [0, 1, 2, 3, 4]
assert replay["status"] == "pass" and len(replay["runs"]) == 5
assert [row["seed"] for row in synthetic_replay["rows"]] == [0, 1, 2, 3, 4]
scores, generalization, walls = [], [], []
for run, check in zip(experiment["runs"], replay["runs"]):
    seed = run["seed"]
    directory = root / f"seed-{seed}"
    manifest, history, selection = (read(directory / name) for name in
                                    ("manifest.json", "history.json", "selection.json"))
    assert run["config_sha256"] == digest and run["generations"] == config["generations"]
    assert run["provenance"] == pilot["provenance"] and run["hardware"] == pilot["hardware"]
    assert manifest == {"config": config, "seed": seed, "hardware": run["hardware"],
                        "provenance": run["provenance"], "config_sha256": digest}
    assert selection == run["selection"]
    assert [row["generation"] for row in history] == list(range(config["generations"]))
    assert all(row["valid"] > 0 for row in history)
    assert len(selection["episode_scores"]) == config["heldout_episodes"]
    assert np.isclose(selection["heldout_score"], np.mean(selection["episode_scores"]))
    assert selection["candidate_id"] in selection["finalist_ids"]
    assert all(x.startswith(f"seed-{seed}:{config['generations']-1}:")
               for x in selection["finalist_ids"])
    assert selection["heldout_valid"] and selection["generalization_valid"]
    assert check["seed"] == seed and check["candidate_id"] == selection["candidate_id"]
    assert check["interpolation"]["score"] == selection["heldout_score"]
    assert check["generalization"]["score"] == selection["generalization_score"]
    phenotype, metadata = reload(directory / "champion.npz")
    assert metadata["candidate_id"] == selection["candidate_id"]
    with np.load(directory / "champion-genes.npz", allow_pickle=False) as genes:
        assert set(genes.files) == {"nodes", "conns"}
    for name in ("checkpoint.bin", "evaluated-winner.npz", "evaluated-winner-genes.npz",
                 "validation-finalists-genes.npz"):
        assert (directory / name).stat().st_size > 0
    scores.append(selection["heldout_score"])
    generalization.append(selection["generalization_score"])
    walls.append(run["wall_seconds"])
assert experiment["passed_seeds"] == sum(x >= config["target"] for x in scores)
value = {"schema": 1, "status": "measured", "method": "direct-recurrent-genome",
         "comparison": "system-only-synchronous-profile", "config_sha256": digest,
         "image_digest": pilot["hardware"]["image_digest"],
         "engine_sha256": pilot["provenance"]["engine_sha256"],
         "pilot_passes": True, "heldout_score": spread(scores), "seed_scores": scores,
         "generalization_score": spread(generalization),
         "seed_generalization_scores": generalization,
         "passed_seeds": experiment["passed_seeds"],
         "wall_seconds_per_seed": spread(walls),
         "task_portable_replay": "pass-all-five",
         "generic_synthetic_replay": {
             "passed_seeds": [row["seed"] for row in synthetic_replay["rows"]
                              if row["status"] == "pass"],
             "failed_seeds": [row["seed"] for row in synthetic_replay["rows"]
                              if row["status"] != "pass"]}}
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps({"status": "pass", "passed_seeds": value["passed_seeds"]}))
