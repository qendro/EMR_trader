# W6 irregular and dense-substrate scale fixture

The [machine audit](w6-v1-audit.json) passes both independently selected
cards. Each card first completed a population-8 pilot at microbatch 4 with
zero cgroup swap, then executed the full population-256 fixture with that
same batch size. The active total per graph was exactly 2,048 nodes,
including input and output roles. Each logical agent ran four scenarios for
128 carried steps. One JAX signature compiled per layout; the remaining
chunks introduced no new signature.

| Card | Layout | Inputs / hidden / output | Active edges per graph | Warm 4-agent chunk | Process peak RSS | Swap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RTX 3070 | Edge recurrent | 4 / 2,043 / 1 | 32,768 | 24.75 ms | 851 MB | 0 |
| RTX 3070 | Dense substrate | 400 / 1,647 / 1 | 693,215 | 90.39 ms | 1,312 MB | 0 |
| RTX 3060 | Edge recurrent | 4 / 2,043 / 1 | 32,768 | 30.33 ms | 851 MB | 0 |
| RTX 3060 | Dense substrate | 400 / 1,647 / 1 | 693,215 | 104.93 ms | 1,317 MB | 0 |

The same-shape [CPU/GPU stage comparison](w6-cpu-gpu-ratio-v1.json) used
population-8 pilots and warm four-agent chunks. GPU/CPU speedups were 4.12x
(edge) and 3.45x (dense) on the RTX 3070, and 3.44x (edge) and 4.74x
(dense) on the RTX 3060. These ratios cover rollout execution only; they
do not satisfy the full-generation M4 G5 policy.

The dense fixture's edge count exceeds 32,768 because its fully connected
400-wide input and output blocks are deliberately outside the sparse
irregular cap. These are declared synthetic execution fixtures, not evolved
EMR phenotypes. The evolved feasible envelope and full-generation CPU/GPU
speed ratio are measured separately in M4 G5.
