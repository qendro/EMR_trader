"""Compare full native state and quality summaries after separate-container resume."""
import argparse
import json
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
import jax
import numpy as np
from neuroevo.artifacts import load_checkpoint,write_json
from neuroevo.runner import provenance

p=argparse.ArgumentParser()
p.add_argument("uninterrupted")
p.add_argument("resumed")
p.add_argument("--output",required=True)
args=p.parse_args()
a,b=Path(args.uninterrupted),Path(args.resumed)
manifest=json.loads((a/"manifest.json").read_text())
config={**manifest["config"],"run_seed":manifest["seed"]}
x=load_checkpoint(a/"checkpoint.bin",config,provenance())
y=load_checkpoint(b/"checkpoint.bin",config,provenance())
assert jax.tree.structure(x["state"])==jax.tree.structure(y["state"])
for one,two in zip(jax.tree.leaves(x["state"]),jax.tree.leaves(y["state"])):
    np.testing.assert_array_equal(one,two)
assert len(x["history"])==len(y["history"])
for one,two in zip(x["history"],y["history"]):
    for field in ("generation","best","mean","valid","species","stagnation_removed","nodes_max","edges_max"):
        assert one[field]==two[field],field
assert x["champion"]["id"]==y["champion"]["id"]
assert x["champion"]["score"]==y["champion"]["score"]
assert set(x["champion"])==set(y["champion"])
for field in ("nodes","conns","weights"):
    if field in x["champion"]:
        np.testing.assert_array_equal(x["champion"][field],y["champion"][field])
write_json(args.output,{"status":"pass","comparison":"bitwise-all-native-state-leaves-and-evaluated-archive",
                       "split":40,"generations":int(x["state"].generation),"hardware":hardware,
                       "history_quality_fields":"exact","timing_fields":"excluded"})
