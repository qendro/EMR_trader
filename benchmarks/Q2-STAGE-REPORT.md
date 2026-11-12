# Q2 synchronized stage probe

Measured November 12, 2026, on both GPUs with the frozen
[Q2 capacity manifest](../configs/q2-capacity-v2.json), seeds 700–702 in
feedforward and recurrent modes. The stage-probe image was
`sha256:27f068364304686b5b1879e288dc3a1daa61ab854ff9315e04fe54b916cfbf87`.
Each of the 12 cells used a fresh process and cold cache, two cold generations,
and ten warm generations. The separate
[audit](q2-stage-v1-audit.json) passes 12/12 cells, with per-cell warm median
and p95 values. Raw records are in
[`results/evidence/q2-stage-v1`](results/evidence/q2-stage-v1/).

The probe replaces only the pinned EMR generation module's clock during
`run_generation`. At each of its feedforward or recurrent timer boundaries, it
waits for every live JAX array in that generation frame before returning the
timestamp. Each row records the timer source line, array count and barrier
duration. The audit verifies all nine feedforward or 15 recurrent timer sites
per generation, nonempty array waits, ten warm samples per stage, and matching
reported fitness summaries, graph sizes and dataset bytes with the
corresponding capacity run.

| GPU and mode | CPPN query median (s) | Hidden recurrence discovery median (s) | Evaluation median (s) | Evolution median (s) | Unattributed generation median (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3070 feedforward | 0.0846 | 0 | 0.0052 | 0.0038 | 0.0272 |
| RTX 3070 recurrent | 0.1096 | 0.1403 | 0.0191 | 0.0042 | 0.0363 |
| RTX 3060 feedforward | 0.1060 | 0 | 0.0055 | 0.0040 | 0.0254 |
| RTX 3060 recurrent | 0.1234 | 0.1472 | 0.0235 | 0.0045 | 0.0366 |

Each table value is the median of the three per-seed warm medians. The audit
retains all seven stage medians and p95 values per cell. Feedforward has no
hidden-to-hidden discovery, so its zero is a protocol fact. Unattributed time
includes work outside the upstream stage timers; these stage medians must not
be summed as if they were an exact end-to-end generation time. The separate
[capacity and resource screen](Q2-CAPACITY-REPORT.md) remains the end-to-end
timing and memory evidence.

Together, the capacity/resource audit and this stage audit provide a passed
Q2 upstream feasibility screen for the declared depth-4, P=128 envelope and
32,768-slot recurrent cap. They do not establish an adaptive platform rollout
or close M2's incomplete fixed-time comparison.
