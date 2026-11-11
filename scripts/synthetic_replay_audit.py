"""Retain per-seed results of the generic synthetic portable replay probe."""
import argparse
import json
from pathlib import Path

from neuroevo.backend import initialize
hardware = initialize()

from neuroevo.artifacts import write_json
from neuroevo.cli import replay


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.experiment)
rows = []
for seed in range(5):
    path = root / f"seed-{seed}" / "champion.npz"
    try:
        result = replay(path, hardware)
        rows.append({"seed": seed, "status": "pass", "max_abs_error": result["max_abs_error"]})
    except AssertionError as error:
        rows.append({"seed": seed, "status": "failed", "error_type": type(error).__name__,
                     "message": str(error)[:1200]})
write_json(args.output, {"schema": 1, "hardware": hardware,
                         "probe": "generic-128-step-synthetic-observation",
                         "rows": rows})
print(json.dumps({"passes": sum(row["status"] == "pass" for row in rows),
                  "failures": [row["seed"] for row in rows if row["status"] != "pass"]}))
