# W7 adaptive stability on the RTX 3070

The [audit](w7-stability-v1-audit.json) passes the declared W7 and G7
stability cell. The source-mounted adaptive EMR run used the
[frozen diagnostic manifest](../configs/s2-w7-adaptive-stability-v1.json):
population 128, changing discovered topology, discovery slices of eight,
bounded converted-phenotype storage and a 64 MiB rollout estimate limit.
The run stopped after 380 measured generations and 966.95 seconds, exceeding
both the 100-generation and 15-minute minima. Its
[compressed history](results/s2-w7-adaptive-v1/seed-900/history.json.gz)
and [summary](results/s2-w7-adaptive-v1/seed-900/summary.json) are retained.

| Measurement after 38-generation warmup | Result |
| --- | ---: |
| Training rollout signatures, warmup to last training generation | 4 to 4 |
| Additional final validation/held-out signatures | 4 |
| Early/late warm RSS median | 1,570.6 / 1,586.0 MB |
| Warm RSS growth | 0.98% |
| Device live memory growth | 0.00% |
| Median warm generation time, early/late | 2.437 / 2.421 s |
| Cgroup swap | 0 bytes |

The final-selection signature increase is expected because validation and
held-out scenario shapes differ from training; it occurred after the last
training generation. The G7 limit applies to the fixed warm training
signature set. No per-sensor thermal series was captured, so this report
does not assert a temperature trend. G5's separately matched complete
generation ratio remains a distinct gate.
