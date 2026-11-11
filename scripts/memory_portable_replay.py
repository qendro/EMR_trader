"""Replay exported memory champions on the recorded held-out scenarios."""
import argparse
import json
from pathlib import Path

from neuroevo.backend import initialize
hardware = initialize()

import jax.numpy as jnp
import numpy as np
from neuroevo.artifacts import reload, write_json
from neuroevo.contracts import stack_plans
from neuroevo.environments import memory_scenarios
from neuroevo.evaluation import memory_chunk, memory_initial
from neuroevo.planning import to_graph
from neuroevo.reference import step as reference_step


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.experiment)
experiment = json.loads((root / "experiment.json").read_text())
config = experiment["config"]
assert config["task"] == "memory" and experiment["complete_protocol"]
rows = []
for run in experiment["runs"]:
    seed = run["seed"]
    phenotype, metadata = reload(root / f"seed-{seed}" / "champion.npz")
    selection = run["selection"]
    assert metadata["candidate_id"] == selection["candidate_id"]
    graph = to_graph(phenotype, metadata["profile"])
    by_namespace = {}
    for namespace, delays in (("interpolation", range(1, 9)),
                              ("generalization", range(9, 17))):
        observations, cues, delay_ids = memory_scenarios(
            seed, namespace, 0, delays=delays,
            repeats=config["heldout_episodes"] // 16)
        packed = stack_plans([phenotype])
        carried = memory_chunk(packed, memory_initial(packed, len(cues)),
                               observations, cues, delay_ids, jnp.arange(17),
                               profile=metadata["profile"])
        correct = np.asarray(carried.correct[0], dtype=np.int32)
        valid = np.asarray(carried.valid[0], dtype=bool)
        assert valid.all()
        expected = (selection["episode_scores"] if namespace == "interpolation"
                    else None)
        if expected is not None:
            np.testing.assert_array_equal(correct, expected)
        else:
            np.testing.assert_allclose(correct.reshape(8, -1).mean(axis=1),
                                       selection["generalization_by_delay"])
        expected_score = (selection["heldout_score"] if namespace == "interpolation"
                          else selection["generalization_score"])
        assert float(correct.mean()) == expected_score
        # Independent scalar replay checks the exported phenotype's action on
        # every held-out trajectory, without importing an evolution backend.
        obs_host = np.asarray(observations)
        cue_host = np.asarray(cues)
        delay_host = np.asarray(delay_ids)
        reference_correct = []
        for observation, cue, delay in zip(obs_host, cue_host, delay_host):
            state = np.zeros(len(graph.node_ids), np.float64)
            action = None
            for time in range(int(delay) + 1):
                head, state = reference_step(graph, state, observation[time])
                if time == delay:
                    action = 1 if head[0] >= 0 else -1
            reference_correct.append(action == cue)
        np.testing.assert_array_equal(correct, reference_correct)
        by_namespace[namespace] = {"score": expected_score,
                                    "episodes": len(correct),
                                    "scalar_action_agreement": True}
    rows.append({"seed": seed, "candidate_id": metadata["candidate_id"],
                 **by_namespace})
write_json(args.output, {"status": "pass", "hardware": hardware,
                         "profile": config["adapter"]["profile"], "runs": rows,
                         "label": "task-heldout-portable-and-scalar-replay"})
print(json.dumps({"status": "pass", "seeds": len(rows)}))
