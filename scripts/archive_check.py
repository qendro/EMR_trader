"""Re-evaluate the persisted pre-tell winner on its original semantic scenarios."""
import argparse
import json
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
import jax.numpy as jnp
import numpy as np
from neuroevo.runner import load_config
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.artifacts import reload,write_json
from neuroevo.environments import cartpole
from neuroevo.evaluation import make_cartpole_rollout,evaluate_population
from neuroevo.planning import plan
from neuroevo.randomness import key

p=argparse.ArgumentParser()
p.add_argument("directory")
args=p.parse_args()
directory=Path(args.directory)
manifest=json.loads((directory/"manifest.json").read_text())
config=manifest["config"]
adapter=TensorNEATAdapter(config["adapter"])
portable,meta=reload(directory/"evaluated-winner.npz")
with np.load(directory/"evaluated-winner-genes.npz",allow_pickle=False) as data:
    decoded=plan(adapter.decode(data["nodes"],data["conns"]))
for a,b in zip(decoded,portable):
    np.testing.assert_array_equal(a,b)
generation=int(meta["candidate_id"].split(":")[-2])
scenario_generation=generation if config["scenario_policy"] == "generation-specific" else 0
keys=jnp.stack([key(manifest["seed"],"train",scenario_generation,e) for e in range(config["episodes"])])
env,params=cartpole()
result=evaluate_population([portable],make_cartpole_rollout(env,params,config["horizon"],config["adapter"]["profile"]),
                           keys,jnp.zeros((config["horizon"],4),jnp.float32),1)
score=float(result[0].mean())
np.testing.assert_equal(score,meta["training_score"])
write_json(directory/"archive-check.json",{"status":"pass","candidate_id":meta["candidate_id"],
                                         "original_score":meta["training_score"],"replayed_score":score,
                                         "hardware":hardware})
