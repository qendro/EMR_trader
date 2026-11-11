"""Verify all three frozen W8 contract-result files."""
import argparse
import hashlib
import json
from pathlib import Path

from neuroevo.artifacts import canonical


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="/results/w8-v2")
parser.add_argument("--manifest", default="configs/w8-tracking-v2.json")
parser.add_argument("--output", default="/reports/w8-status.json")
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text())
digest = hashlib.sha256(canonical(manifest)).hexdigest()
cells = {}
engine, image, policy = None, None, None
for profile in ("cpu", "rtx3070", "rtx3060"):
    data = json.loads((Path(args.source) / f"{profile}.json").read_text())
    assert data["status"] == "pass" and data["manifest_sha256"] == digest
    assert data["steps"] == 10000 and data["longer_steps"] == 20000
    assert data["chunk_steps"] == manifest["chunk_steps"]
    assert data["signature_count_after_250_and_500_shapes"] == 2
    assert all(data[k] is True for k in ("dynamic_values_no_recompile",
                                         "longer_horizon_no_recompile",
                                         "causal_actions", "chunk_continuation"))
    assert data["memory"]["cgroup_swap_bytes"] == 0
    assert -1 <= data["min_reward"] <= data["max_reward"] <= 1
    assert abs(data["score"] - data["reference_score"]) <= .001 * (1 + abs(data["reference_score"]))
    assert data["hardware"]["backend"] == ("cpu" if profile == "cpu" else "gpu")
    if profile != "cpu":
        assert profile[3:] in data["hardware"]["gpu"][1]
    engine = engine or data["provenance"]["engine_sha256"]
    image = image or data["hardware"]["image_digest"]
    policy = policy or data["policy_candidate_id"]
    assert data["provenance"]["engine_sha256"] == engine
    assert data["hardware"]["image_digest"] == image
    assert data["policy_candidate_id"] == policy
    cells[profile] = {"status": "pass", "score": data["score"],
                      "reference_score": data["reference_score"],
                      "max_cgroup_bytes": data["memory"]["cgroup_peak_bytes"],
                      "max_device_allocator_bytes": (None if data["memory"]["device_allocator"] is None
                                                     else data["memory"]["device_allocator"]["peak_bytes_in_use"])}
assert max(c["score"] for c in cells.values()) - min(c["score"] for c in cells.values()) < 1e-4
output = {"schema": 1, "status": "pass", "task": "W8-tracking-contract",
          "manifest_sha256": digest, "engine_sha256": engine,
          "image_digest": image, "policy_candidate_id": policy, "cells": cells}
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"status": "pass", "profiles": list(cells)}))
