"""Freeze derived M1 diagnostics before running; no acceptance thresholds changed."""
import copy
import json
from pathlib import Path

root=Path("configs")
base=json.loads((root/"m1-acceptance-v1.json").read_text())
for name in ("dag-control","synchronous-cartpole","fixed-scenario","memory"):
    c=copy.deepcopy(base)
    c.update(id=f"m1-{name}-diagnostic-v1",kind="diagnostic",seeds=list(range(100,105)),
             budget_status="diagnostic-proposed-budget")
    if name == "fixed-scenario":
        c["scenario_policy"]="fixed"
    if name == "synchronous-cartpole":
        c["adapter"]["profile"]="synchronous"
    if name == "memory":
        c.update(task="memory",workload="W4-memory",generations=250,episodes=32,horizon=17,
                 validation_episodes=64,heldout_episodes=256,target=.9,action="sign-positive-tie",
                 shared_protocol="configs/memory-v1.json")
        c["adapter"].update(inputs=2,outputs=1,profile="synchronous")
    (root/f"m1-{name}-diagnostic-v1.json").write_text(json.dumps(c,indent=2)+"\n")
