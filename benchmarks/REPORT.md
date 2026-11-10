# M1 implementation evidence

This is the historical M1 source snapshot. Subsequent H0/H1 implementation and its separate qualification are recorded in [H1-REPORT.md](H1-REPORT.md); this report's later-work status refers to the M1 handoff.

Measured on this Windows/WSL2 host in Linux Docker containers. RTX 3070 gates progression. Every lane uses the same hash-locked dependencies; smoke and pilot runs are excluded from learning acceptance.

| Profile | Tests | Held-out means, seeds 0–4 | Wall seconds per seed | G9 warm seconds |
|---|---:|---|---|---:|
| cpu | 44 | 500, 500, 500, 500, 500 | 41.0, 39.5, 46.7, 46.0, 51.2 | 0.2049 |
| rtx3070 | 44 | 500, 500, 500, 500, 500 | 298.2, 260.6, 240.2, 261.2, 222.4 | 1.6149 |
| rtx3060 | 44 | 500, 500, 500, 500, 500 | 331.4, 292.2, 283.7, 286.3, 210.2 | 1.8254 |

The target is ≥450 in at least three of five seeds. Each seed has a 7,200-second budget. G9 pilots use seed 900. Learning runs shared host CPUs with other diagnostic/card runs; their timings establish feasibility, not an isolated GPU speed comparison.

## Correctness, compilation and controller cost

| Profile | Differential max error/threshold | G2 100 post-warm generations | G4 median ratio |
|---|---:|---|---:|
| cpu | 0.01752706730962038 | True | 0.989 |
| rtx3070 | 0.021638256058457168 | True | 1.033 |
| rtx3060 | 0.021638256058457168 | True | 0.950 |

Complete native state, evaluated archive and history quality fields are compared after separate-container 40/100 CPU resume. Standalone closed-loop replay compares all actions, policy states, environment states and rewards against the independent NumPy reference. See the qualification artifacts for status and numerical errors.

G4 uses three fresh-process pairs, the same seed 900, dimensions, diagnostics and checkpoint interval. It compares generations 2–10 inclusive in both paths. Driver trial 0 continues to 102 generations for G2; the other two stop at 12. Numerical/quality fields must match exactly before a pair is accepted.

G4 compares the median of the three controller medians with the median of the three driver medians. All pairs, including timing outliers, remain in the raw report.

| Profile | Trial | Controller median seconds | Driver median seconds | Pair ratio |
|---|---:|---:|---:|---:|
| cpu | 0 | 0.2012 | 0.1993 | 1.010 |
| cpu | 1 | 0.3153 | 0.2122 | 1.486 |
| cpu | 2 | 0.1989 | 0.2034 | 0.977 |
| rtx3070 | 0 | 1.4673 | 1.3619 | 1.077 |
| rtx3070 | 1 | 1.3938 | 1.3754 | 1.013 |
| rtx3070 | 2 | 1.4209 | 1.4874 | 0.955 |
| rtx3060 | 0 | 1.5485 | 1.6788 | 0.922 |
| rtx3060 | 1 | 1.4907 | 1.6004 | 0.931 |
| rtx3060 | 2 | 1.5204 | 1.5876 | 0.958 |

## Ungated diagnostics

- memory: seed 100 = 1, seed 101 = 1, seed 102 = 1, seed 103 = 1, seed 104 = 1.
- dag-control: seed 100 = 500, seed 101 = 500, seed 102 = 500, seed 103 = 500, seed 104 = 500.
- synchronous-cartpole: seed 100 = 500, seed 101 = 500, seed 102 = 500, seed 103 = 500, seed 104 = 500.
- fixed-scenario: seed 100 = 500, seed 101 = 500, seed 102 = 500, seed 103 = 500, seed 104 = 500.

Memory interpolation delays 1–8 and generalization delays 9–16 are reported separately per delay. Only final champions receive held-out evaluation: the recorded first training threshold is a training proxy, not a measured held-out solve generation. Synchronous hidden hops add one observation of latency per hidden node; the latency fixtures test one and two hops. Comparisons use diagnostic seeds 100–104, separate from acceptance.

| Seed | DAG return | Synchronous minus DAG return | Varying-scenario stagnation removals | Fixed-scenario removals | Memory interpolation / generalization | First training ≥90% generation (zero based) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 500 | 0 | 1 | 1 | 1 / 1 | 54 |
| 101 | 500 | 0 | 0 | 3 | 1 / 1 | 8 |
| 102 | 500 | 0 | 3 | 3 | 1 / 1 | 22 |
| 103 | 500 | 0 | 2 | 1 | 1 / 1 | 11 |
| 104 | 500 | 0 | 0 | 3 | 1 / 1 | 11 |

The synchronous comparison changes both propagation timing and permitted genome operators. It is a direct-genome system comparison; it isolates no encoding advantage. Ceiling CartPole scores do not establish that latency has no cost on other tasks. Stagnation removal differences are descriptive over five matched seeds, with no significance claim.

## Q2 upstream screen

| Card | Mode | Seed | Result | Warm seconds/generation |
|---|---|---:|---|---:|
| rtx3060 | feedforward | 700 | timing pass; resource audit open | 0.1633 |
| rtx3060 | feedforward | 701 | timing pass; resource audit open | 0.1545 |
| rtx3060 | feedforward | 702 | timing pass; resource audit open | 0.1627 |
| rtx3060 | recurrent | 700 | timing pass; resource audit open | 0.3444 |
| rtx3060 | recurrent | 701 | timing pass; resource audit open | 0.3541 |
| rtx3060 | recurrent | 702 | SparseCapExceeded | — |
| rtx3070 | feedforward | 700 | timing pass; resource audit open | 0.1406 |
| rtx3070 | feedforward | 701 | timing pass; resource audit open | 0.1770 |
| rtx3070 | feedforward | 702 | timing pass; resource audit open | 0.1461 |
| rtx3070 | recurrent | 700 | timing pass; resource audit open | 0.4849 |
| rtx3070 | recurrent | 701 | timing pass; resource audit open | 0.3259 |
| rtx3070 | recurrent | 702 | SparseCapExceeded | — |

Both recurrent seed-702 runs refuse a 13,424-connection discovery at the frozen 10,000 cap. `q2-capacity-v2.json` declares 32,768 slots with refusal on overflow; it has not run. Q2 remains open: profile peak-resource audit, complete active-size distributions and synchronized isolated stage timing are still required. Native upstream stage timers are dispatch/incidental-wait measurements. Successful warm medians are not sequential-rollout budgets or complete Q2 passes. Full CPU EMR and max-depth-5 cells have not run; neither is inferred from the GPU depth-4 screen.

### Raw upstream direct/fixed baselines

These use 64 independent samples, native DAG direct genomes or native fixed HyperNEAT with reset per sample and K=3 settling. They share the frozen dataset with EMR but differ in capacity, functions and propagation timing. This is a system comparison, not evidence about encoding alone. V1 passed raw scores to native evolution; nonfinite direct fitness caused strict JSON failures. V2 freezes candidate rejection for nonfinite outputs/fitness, -1e30 invalid fitness, counts per generation and all-invalid abort. It changes no native operators or workload capacities. V1 failed cells remain intact. V2 uses one initially empty persistent cache per profile/mode: seed 700 is empty-cache, 701/702 retain it in fresh processes. Initial entry counts are recorded; their first-generation times are not empty-cache compilation measurements.

| Version | Profile | Mode | Completed / failed seeds | Warm median seconds (range across completed seeds) | Invalid candidate evaluations |
|---|---|---|---|---:|---:|
| v1 | cpu | direct | 0 / 3 of 3 | no completed cell | not recorded |
| v1 | cpu | fixed | 3 / 0 of 3 | 0.0388 (0.0307–0.0466) | not recorded |
| v1 | rtx3070 | direct | 0 / 3 of 3 | no completed cell | not recorded |
| v1 | rtx3070 | fixed | 3 / 0 of 3 | 0.0174 (0.0153–0.0204) | not recorded |
| v1 | rtx3060 | direct | 0 / 3 of 3 | no completed cell | not recorded |
| v1 | rtx3060 | fixed | 3 / 0 of 3 | 0.0253 (0.0251–0.0259) | not recorded |
| v2 | cpu | direct | 3 / 0 of 3 | 0.0344 (0.0334–0.0358) | 26, 21, 38 |
| v2 | cpu | fixed | 3 / 0 of 3 | 0.0391 (0.0323–0.0448) | 0, 0, 0 |
| v2 | rtx3070 | direct | 3 / 0 of 3 | 0.0222 (0.0212–0.0235) | 26, 21, 38 |
| v2 | rtx3070 | fixed | 3 / 0 of 3 | 0.0226 (0.0214–0.0228) | 0, 0, 0 |
| v2 | rtx3060 | direct | 3 / 0 of 3 | 0.0313 (0.0301–0.0318) | 26, 21, 38 |
| v2 | rtx3060 | fixed | 3 / 0 of 3 | 0.0330 (0.0300–0.0334) | 0, 0, 0 |

The first-invalid-genome replay reproduces 64 nonfinite outputs across 64 samples. Active attributes are finite; isolated non-input node IDs are [4]. The fixture, genes, inputs, outputs and replay result are retained under `q2-native-v2/cpu-direct-700`.

49 persisted Q2 datasets have byte-identical FP32 input/target arrays (`results/q2-dataset-audit.json`); the audit also checks completed v2 histories, manifest/script identity and the declared cache policy.

## Scope and provenance

Implemented: Q0/Q1 and V0–V5, the M1 one-capacity DAG/synchronous runtime, scalar layered reference, strict portable import/export, native checkpoints, memory/CartPole evaluation and reducer fixtures. M2–M5, platform fixed/adaptive decoders, large-fan-in calibration, W8 sequence environment, scale optimizations and sustained W7 stability are not implemented/qualified. No trading functionality is included.

Current results live under `results/evidence/*current*` and `g4-*`; every run records its engine and image hash. Earlier source `e03c0613…` results and checkpoints are retained under `m1-final-*`, with exact source in `reproduction/m1-e03`. They do not supply acceptance evidence for the final overflow guard. Other retained development results are identified by their own manifests. The file index contains SHA-256 hashes, including failed cells and portable/checkpoint artifacts.

Measured W3 GPU/CPU throughput ratios: {"rtx3070": 0.141595578423241, "rtx3060": 0.1323281345276122}. M1 reports these ratios without gating; no useful-acceleration claim follows from CUDA placement.

Three-run controller medians and ranges (seconds): {"cpu": {"median_seconds": 0.20119082700148283, "range_seconds": [0.19885442899976624, 0.31529132700052287]}, "rtx3070": {"median_seconds": 1.4208835419995012, "range_seconds": [1.3937903549995099, 1.467340596998838]}, "rtx3060": {"median_seconds": 1.5203934349992778, "range_seconds": [1.490665433999311, 1.548492351999812]}}.

## Gate disposition

closed-first-vertical-slice; H0 is next; M2–M5 not implemented.
Per-profile G3: {"cpu": true, "rtx3070": true, "rtx3060": true}. Host/process/cgroup and allocator peaks are recorded separately. Physical GPU and VM observations use the recorded sampling window; pre-existing VM swap use is retained and swap-out growth is checked separately. See JSON for maxima, minimum free memory, thermals, sample counts and window boundaries. The RTX 3060 drives the host display; the RTX 3070 display is disabled (`results/display-state.csv`). Physical telemetry includes background display use.
