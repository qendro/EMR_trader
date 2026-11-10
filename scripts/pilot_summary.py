import json
from pathlib import Path
import statistics
for profile in ("cpu","rtx3070","rtx3060"):
    root=Path("/results")/f"m1-{profile}"
    pilot=json.loads((root/"pilot.json").read_text())
    rows=json.loads((root/"pilot-run/history.json").read_text())[2:-1]
    print(json.dumps({"profile":profile,"passes":pilot["passes"],"warm_seconds":pilot["warm_generation_seconds"],
                      "fixed_seconds":pilot["fixed_overhead_seconds"],"allowable_generations":pilot["allowable_generations"],
                      "stages":{k:statistics.median(r[k] for r in rows) for k in ("decode_plan_seconds","rollout_seconds","evolve_seconds")},
                      "rollout_signatures":[r["rollout_signatures"] for r in rows],
                      "tell_signatures":[r["tell_signatures"] for r in rows]}))
