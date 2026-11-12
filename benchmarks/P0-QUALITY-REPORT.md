# P0 partially observed CartPole: first frozen adaptive result

The immutable adaptive image
`sha256:5232a86eebe1aaa7b6c71e5f69626290939dfbcf5a36801b0ed51d2f2cdaf8f8`
passed its exact-manifest G9 pilot and completed all five 100-generation
seeds. Each final policy was selected using 32 validation episodes and scored
once on 100 paired held-out episodes. The [experiment](results/p0-partial-acceptance-v1/experiment.json),
[quality audit](p0-partial-adaptive-quality-v1.json), frozen
[random baseline](p0-random-action-baseline-v1.json), and compressed per-seed
histories retain the evidence.

| Seed | Random mean steps | Adaptive held-out mean | Improvement / 500 |
| ---: | ---: | ---: | ---: |
| 0 | 22.64 | 47.00 | 4.87% |
| 1 | 23.67 | 57.99 | 6.86% |
| 2 | 21.14 | 85.31 | 12.83% |
| 3 | 21.44 | 59.10 | 7.53% |
| 4 | 19.61 | 52.17 | 6.51% |

Median paired normalized improvement was **6.86%**, below the frozen 20%
target. The seed-level bootstrap 95% interval was **[4.87%, 12.83%]**;
the positive interval establishes improvement over random action in this
sample but does not satisfy the required magnitude. Zero of five seeds met
the separate raw 120-step target. The P0 harder-task gate therefore remains
open. This failed result stays attached to the frozen protocol; it is not
reinterpreted as a pass because the policies beat random action.

The direct, fixed-substrate and fixed-topology weight comparisons were
registered before these adaptive scores were available and are queued on
the same five seeds and held-out protocol. Their results will indicate
whether the shortfall is specific to adaptive discovery or shared by the
task and search budget. No tuning or task change is claimed from the
unfinished comparisons.
