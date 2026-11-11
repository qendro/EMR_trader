# W8 shared-sequence contract check

Measured November 10, 2026 on CPU, RTX 3070 and RTX 3060. The versioned
[`w8-tracking-v2.json`](../configs/w8-tracking-v2.json) defines a two-channel
uniform FP32 sequence, 10,000 scored steps, a causal current-row observation,
the next row's first channel as target, bounded tanh action, and reward
`clip(1 - |action-target| - 0.5|action-previous_action|, -1, 1)`. One exported
B0 layered memory policy is used as a fixed fixture. This is a runtime and
evaluator contract check, not an evolved W8 learning score.
The full current suite passed 69 tests on each of the three profiles.

The compiled scan consumes 250 steps per call and carries policy state,
previous action and Welford mean/M2/count across 40 chunks. The shared
sequence is a dynamic argument outside the population axis. Each profile
passed these checks:

| Profile | 10,000-step score | Causal prefix | Changed values recompile | 20,000-step new signature | Chunk continuation | Swap |
|---|---:|---|---|---|---|---:|
| CPU | 0.7142630 | pass | no | no | pass | 0 |
| RTX 3070 | 0.7142629 | pass | no | no | pass | 0 |
| RTX 3060 | 0.7142629 | pass | no | no | pass | 0 |

The reducer score agrees with a float64 calculation from the same FP32 reward
sequence within 1e-3 absolute and relative tolerance. All rewards stayed in
[-1, 1], every run counted exactly 10,000 steps, and 250-step versus 500-step
partitioning produced identical actions and matching scores. Two JIT
signatures correspond to those two chunk shapes; extending the horizon from
10,000 to 20,000 steps with the 250-step shape added none. Perturbing future
sequence values left earlier actions unchanged. The prior Welford/Chan fixture
suite separately covers constant, nearly constant and full-range 10,000-step
reward sequences with four partitions.

Raw per-profile results and resource measurements are under
[`w8-v2`](results/evidence/w8-v2), indexed by the
[evidence bundle](results/evidence/index.json). The consolidated
[status](w8-status.json) verifies matching manifest, policy, engine and image
identities across all three profiles. The source image digest and
manifest hash are recorded in each result. The W8 engine hash is
`54b32a9f470a556e28c6e537a4b1a5d342ddd15a0a58b993a864392057f1d0e7`.
W8 closes its M2 contract check,
while the geometric diagnostic, fixed-time comparison and M2 performance gates
remain open.
