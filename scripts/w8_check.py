"""Run the frozen W8 tracking contract against one portable layered policy."""
import argparse
import hashlib
import json
from pathlib import Path

from neuroevo.backend import initialize
hardware = initialize()

import numpy as np
from neuroevo.artifacts import canonical, reload, write_json
from neuroevo.contracts import stack_plans
from neuroevo.resources import memory_record
from neuroevo.runner import provenance
from neuroevo.w8 import chunk, evaluate, sequence


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="configs/w8-tracking-v2.json")
parser.add_argument("--output", required=True)
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text())
assert manifest["schema"] == 1 and manifest["stage"] == "M2-contract-check"
phenotype, metadata = reload(manifest["policy_artifact"])
assert metadata["profile"] == manifest["profile"] == "layered"
plans = stack_plans([phenotype])
values = sequence(manifest["seed"], manifest["horizon"])
result = evaluate(plans, values, chunk_steps=manifest["chunk_steps"])
assert result["valid"].all() and int(result["carry"].moments.count[0, 0]) == 10000
assert result["rewards"].min() >= -1 and result["rewards"].max() <= 1
rewards = result["rewards"].astype(np.float64)
reference_score = float(rewards.mean() / (rewards.std() + .01))
np.testing.assert_allclose(result["score"], reference_score, atol=1e-3, rtol=1e-3)
assert chunk._cache_size() == 1

changed = values.copy()
changed[501:] *= -.5
perturbed = evaluate(plans, changed, chunk_steps=250)
np.testing.assert_array_equal(result["actions"][:500], perturbed["actions"][:500])
assert not np.array_equal(result["actions"][501:], perturbed["actions"][501:])
assert chunk._cache_size() == 1

repartitioned = evaluate(plans, values, chunk_steps=500)
np.testing.assert_array_equal(result["actions"], repartitioned["actions"])
np.testing.assert_allclose(result["score"], repartitioned["score"], atol=1e-5)
assert chunk._cache_size() == 2
longer = evaluate(plans, sequence(701, 20000), chunk_steps=250)
assert longer["valid"].all() and chunk._cache_size() == 2

write_json(args.output, {
    "status": "pass", "label": "W8-contract-not-learning-acceptance",
    "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
    "policy_candidate_id": metadata["candidate_id"],
    "hardware": hardware, "provenance": provenance(), "memory": memory_record(),
    "steps": 10000, "longer_steps": 20000, "chunk_steps": 250,
    "score": float(result["score"][0, 0]), "reference_score": reference_score,
    "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
    "min_reward": float(rewards.min()), "max_reward": float(rewards.max()),
    "signature_count_after_250_and_500_shapes": chunk._cache_size(),
    "dynamic_values_no_recompile": True, "longer_horizon_no_recompile": True,
    "causal_actions": True, "chunk_continuation": True,
})
print(json.dumps({"status": "pass", "score": float(result["score"][0, 0])}))
