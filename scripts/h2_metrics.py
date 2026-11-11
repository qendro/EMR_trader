"""Print compact resource and compilation metrics from measured H2 cells."""
import json
from pathlib import Path


roots = {
    "fixed-cartpole": Path("/results/h2-fixed-acceptance-v1/rtx3070-cartpole"),
    "fixed-memory": Path("/results/h2-fixed-acceptance-v1/rtx3070-memory"),
    "direct-memory": Path("/results/h2-direct-memory-acceptance-v1/rtx3070-memory"),
}
for name, root in roots.items():
    experiment = json.loads((root / "experiment.json").read_text())
    pilot = json.loads((root / "pilot.json").read_text())
    histories = [json.loads((root / f"seed-{r['seed']}" / "history.json").read_text())
                 for r in experiment["runs"]]
    memory = [r["memory"] for r in experiment["runs"]]
    print(json.dumps({
        "cell": name,
        "pilot_warm_generation_seconds": pilot["warm_generation_seconds"],
        "max_process_rss_bytes": max(m["process_peak_rss_bytes"] for m in memory),
        "max_cgroup_bytes": max(m["cgroup_peak_bytes"] for m in memory),
        "max_swap_bytes": max(m["cgroup_swap_bytes"] for m in memory),
        "max_device_allocator_peak_bytes": max(m["device_allocator"]["peak_bytes_in_use"]
                                               for m in memory),
        "max_rollout_signatures": max(row["rollout_signatures"] for history in histories
                                      for row in history),
        "max_tell_signatures": max(row["tell_signatures"] for history in histories
                                   for row in history),
        "minimum_valid_candidates": min(row["valid"] for history in histories
                                        for row in history),
    }))
