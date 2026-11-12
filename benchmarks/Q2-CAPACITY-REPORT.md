# Q2 capacity and resource screen

Measured November 12, 2026, on the RTX 3070 and RTX 3060 with driver 610.88.
The frozen [capacity manifest](../configs/q2-capacity-v2.json) was run for
feedforward and recurrent modes, seeds 700–702, in 12 separate fresh processes
with distinct cold compilation caches. The image digest was
`sha256:03c1f0dcd1638ad0ccad2f0a7693ca5db190bc3827560be640f30aa38713f19a`;
the engine hash was
`c759c0027b19fccd63b3d775fbe2f572036a47fde02cd1356dac2b6249db07de`.
The [audit](q2-capacity-v2-audit.json) reports
`capacity-and-resource-screen-pass-stage-audit-open`: all 12 capacity and
resource cells pass. Its per-cell evidence is in
[`results/evidence/q2-capacity-v2`](results/evidence/q2-capacity-v2/).

| GPU and mode | Cells | Slowest cold phase (s) | Slowest warm median (s) | Max active nodes | Max recurrent edges | Peak cgroup (MiB) | Peak physical GPU used (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3070 feedforward | 3/3 | 48.83 | 0.152 | 108 | 0 | 1,414 | 435 |
| RTX 3070 recurrent | 3/3 | 68.67 | 0.343 | 128 | 15,104 | 1,830 | 1,207 |
| RTX 3060 feedforward | 3/3 | 48.27 | 0.168 | 108 | 0 | 1,397 | 1,063 |
| RTX 3060 recurrent | 3/3 | 72.23 | 0.575 | 128 | 15,104 | 1,812 | 1,868 |

Each cell completed two cold and ten warm generations. The external 300-second
cold watchdog did not fire, and every warm median was below the frozen
28-second ceiling. No container exited with an error or was OOM killed. The
audit checked each generation's full 128-genome active-node and edge-count
distributions, upstream count agreement, the 32,768 recurrent connection cap,
the 6 GiB host budget, JAX allocator and physical GPU memory, and a byte-identical
dataset across cells. Recurrent seed 702 reached 15,104 hidden connections on
both cards, exceeding the original 10,000 cap without exceeding the separately
frozen 32,768 cap.

The upstream `step_timings` are dispatch intervals and do not isolate and
synchronize each stage. **The stage-timing requirement remains open.** This
screen also uses independent samples and does not demonstrate an adaptive
platform rollout or close the incomplete M2 fixed-time comparison.
