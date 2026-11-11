# Q2 capacity promotion preparation

The frozen [capacity manifest](../configs/q2-capacity-v2.json) keeps depth 4,
population 128, canonical discovery thresholds and `sparse_cap_policy=refuse`.
It raises the recurrent sparse capacity from 10,000 to 32,768 after the
original seed-702 overflow at 13,424 connections. This is a separately
versioned upstream screen, not an adaptive platform acceptance run.

The [launcher](../scripts/q2_capacity.py) now prepares 12 sequential cells:
RTX 3070 and RTX 3060, feedforward and recurrent, seeds 700–702. Each cell
uses a fresh process and a distinct initially empty JAX cache. An external
watchdog stops a process if its two cold generations do not finish within
300 seconds, even if the compiler blocks the in-process timer. It records
container exit/OOM state, console output and physical GPU memory samples.
The M2 elapsed comparison must finish before this launcher will start cells.

The updated upstream wrapper records complete per-genome active-node and
input/output/recurrent edge counts from the substrate actually evaluated,
along with process RSS, cgroup peak and swap, JAX allocator peak, and host
available memory. Observation time is recorded separately from the frozen
`run_generation` timing boundary. The [audit](../scripts/q2_capacity_audit.py)
checks all cells against the manifest, timing ceilings, full distributions,
resource limits, dataset hashes and recorded failures. It deliberately leaves
the synchronized isolated-stage timing requirement open; upstream dispatch
timers alone do not meet that requirement.

No Q2 capacity measurement has been run. After M2 finishes, build an image
from this source, then run the matrix and copy its evidence:

```powershell
docker build -t neuroevo:q2-capacity-v2 .
python scripts/q2_capacity.py --dry-run
python scripts/q2_capacity.py
```

The measurements will land in `/results/q2-capacity-v2` in the
`neuroevo-results` Docker volume. Copy them through `scripts/collect_results.py`,
then run `python scripts/q2_capacity_audit.py` against the copied evidence.
Only measured capacity, resource and separately synchronized stage evidence
can support the dated Q2 decision before A0.
