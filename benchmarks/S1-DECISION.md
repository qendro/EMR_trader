# S1 recurrent edge layout: development decision

The edge layout preserves dense input/output role blocks and replaces only the
delayed hidden matrix with unique `(source, target, weight)` entries. Tests
cover carried cycles, a self-loop, zero hidden nodes, duplicate refusal, and
the shared population evaluator. The layout is opt-in with
`"runtime_layout": "edge"`; the default remains `dense`.

The [crossover probe](s1-crossover-gpu-development.json) was exploratory:
both CUDA cards were visible, so its `cuda:0` label is not a qualified
single-card result. A 128-step single-policy scan gave these ratios:

| Inputs / hidden / recurrent edges | Sparse speedup | Dense / edge host bytes |
| --- | ---: | ---: |
| 4 / 20 / 100 | 1.76x | 2,729 / 1,909 |
| 25 / 128 / 1,024 | 0.95x | 99,950 / 30,622 |
| 400 / 2,048 / 32,768 | 1.64x | 25,099,805 / 4,542,869 |

The [CPU probe](s1-crossover-cpu-development.json) also includes an exported
EMR champion: two inputs, eight hidden nodes and 64 recurrent edges. For that
actual dense graph, the edge representation used 999 bytes versus 543 bytes
and its isolated step was slower. The synthetic large-cell benefit therefore
does not describe the current evolved graph distribution.

Matched 12-generation dense and edge runs used seed 900 and the same memory
smoke configuration. Generations 2–10 had equal best/mean scores in every
generation. Median rollout time was 0.181 s dense versus 0.192 s edge;
median complete generation was 0.574 s versus 0.589 s. Those runs also had
both GPUs visible and are development evidence, not a G5 or G8 gate cell.
The [single-device crossover map](s1-map-v1-audit.json) has now measured 27
cells per profile: hidden widths 20, 128 and 512; uniform, hub and local
degree patterns; and one, four and 16 episodes. Maximum absolute output error
was 5.96e-7. Median edge speedup by hidden width was:

| Profile | 20 hidden | 128 hidden | 512 hidden |
| --- | ---: | ---: | ---: |
| CPU | 1.16x | 0.83x | 1.28x |
| RTX 3070 | 0.97x | 1.01x | 1.19x |
| RTX 3060 | 1.10x | 0.92x | 1.16x |

These isolated stages do not establish a full-generation G8 gain on evolved
graphs. The actual exported eight-hidden-node champion still favors dense;
the production default therefore remains dense. The map and per-profile raw
cells are retained for future density-based dispatch work.
