# H2 geometric visual diagnostic

Measured November 10, 2026 on the RTX 3070 with image digest
`sha256:1a0c2014dbf15aca9a24259d3fe643b1d2c75ebde19981df92c915b81962eed3`
and engine hash `c01f6ca585f9cd66cac3adf160a0f839a1d939bd99a3f714d8726d6f12a5460d`.
This is an **ungated diagnostic**, separate from H2's CartPole and memory
acceptance scores.

The frozen [protocol](../configs/visual-orientation-v1.json) presents a
5×5 grid with a three-pixel horizontal or vertical stroke. Training and
validation use five positions; held-out evaluation uses four different
positions. Every position and orientation is balanced. Each image receives
independent uniform background noise in [-0.15, 0.15], and the policy makes
one decision. Validation selects a finalist before held-out positions are
evaluated. The seeds 700–702 are disjoint from acceptance seeds. The
[run script](../scripts/h2_visual_runs.ps1) executes all three methods with
the same one-step population evaluator, 96 candidates, 60 generations,
40 training and validation images, and 160 held-out images per seed.

| Method | Held-out scores, seeds 700/701/702 | Median [range] | Wall seconds per seed, median [range] |
|---|---:|---:|---:|
| Fixed-topology B0, 12 hidden | 0.694 / 0.550 / 0.525 | 0.550 [0.525, 0.694] | 16.7 [16.6, 18.4] |
| Direct feedforward genomes | 0.744 / 0.794 / 0.706 | 0.744 [0.706, 0.794] | 27.5 [27.5, 58.1] |
| Fixed-coordinate CPPN, 12 hidden | 0.750 / 0.788 / 0.800 | 0.788 [0.750, 0.800] | 62.2 [60.3, 87.0] |

The fixed CPPN uses the new explicit 2D input coordinates in
`fixed-coordinate-v2`; B0 has the same 25-input, 12-hidden, one-output
feedforward substrate shape. Direct genomes evolve their own sparse topology
and use the DAG profile. All three produce an output on the same input images
through the same evaluator. The fixed CPPN had the highest median held-out
score at this candidate-episode budget, while B0 was fastest. Fixed CPPN
scores were still uneven by position: its per-position scores span 0.5–1.0
within a seed. Three seeds do not establish a general encoding advantage.
The direct-genome result also differs in architecture family; it is a system
comparison. No elapsed-time-matched quality claim follows from these runs.

All 96 candidates were valid in every generation. The [status file](h2-visual-status.json)
contains every seed score, wall time, config hash, per-position accuracy and
the audit outcome. The audit reconstructed all nine champion phenotypes from
archived genes and replayed all 160 held-out decisions per seed. The 72-test
CPU suite passed; the three new visual protocol tests passed on both RTX GPUs.
Raw manifests, histories, selections, checkpoints and exported phenotypes
are retained under [evidence](results/evidence/h2-visual-diagnostic-v1/).
