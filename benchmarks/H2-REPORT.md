# H2 small-substrate learning and comparison

Measured November 10, 2026 on the RTX 3070 gating card. The fixed-coordinate
CPPN used the frozen CartPole and memory manifests at population 128, with one
8-node or 4-node hidden population. Each experiment ran seeds 0–4 after a fresh
matching integrated G9 pilot. B0's previously measured, substrate-shaped
weight baseline is included as the layered control. The direct recurrent memory
cell uses the same seeds and memory scenarios under its synchronous profile.

| Method and task | Held-out median [range] | Target passes | Wall seconds/seed, median [range] |
|---|---:|---:|---:|
| B0 CartPole, layered | 500 [500, 500] | 5/5 | 22.9 [22.8, 24.0] |
| Fixed CPPN CartPole, layered | 500 [439.9, 500] | 4/5 | 137.1 [126.6, 142.0] |
| B0 memory, layered | 1.00 [1.00, 1.00] | 5/5 | 59.0 [58.5, 59.2] |
| Fixed CPPN memory, layered | 1.00 [1.00, 1.00] | 5/5 | 447.5 [422.6, 472.1] |
| Direct recurrent memory, synchronous | 1.00 [0.742, 1.00] | 4/5 | 75.2 [74.8, 75.6] |

The fixed CartPole seed scores were 439.9, 500, 476.25, 500 and 500; the
required score is at least 450. All five fixed memory scores were 1.00 against
the 0.90 target. The direct memory seed scores were 1.00, 0.7421875, 1.00,
1.00 and 1.00. Longer-delay memory generalization was reported separately:
fixed CPPN and B0 were 1.00 on all five seeds; direct recurrent was 1.00 in
four seeds and 0.73828125 in seed 1. These are held-out scores after
validation-only champion selection. All methods used 128 candidates per
generation, 100 CartPole or 250 memory generations, and the same episode counts
per task. Their candidate-episode evaluation budgets therefore match within
each task. The measured 2-hour fixed-elapsed-time comparison remains open.

The fixed encoding and B0 share the layered profile, one hidden population and
the same hidden size, so their within-task rows are the encoding-isolating
control at the declared evaluation budget. B0 matched or exceeded fixed CPPN
quality and completed each task faster on this small substrate. The direct
memory row is a system comparison only because its synchronous update timing
differs. Existing M1 direct CartPole learning on older source is historical
system context; it is not silently combined with these image-matched timings.

The fresh fixed G9 pilots measured warm generations at 1.159 seconds for
CartPole and 1.161 for memory; the direct memory pilot measured 0.234 seconds.
All pilots passed their declared 2-hour per-seed budget, and all acceptance
seeds finished well below it. All 128 candidates were valid in every measured
generation of these three cells; cgroup swap was zero. Peak cgroup usage was
at most 1.21 GB for fixed CartPole, 1.16 GB for fixed memory and 1.16 GB for
direct memory. The raw histories retain per-stage timing, phenotype counts,
compilation signatures and allocator statistics. These small learning cells do
not satisfy M2's separate 1,024-node G5 benefit gate.

The fixed verifier checked matching pilot/config/source/hardware records,
all five histories, validation selections, exact reconstruction of champion
phenotypes from archived CPPN genes, and the 128-step portable replay on both
tasks. The direct verifier checked its histories, selections and task-specific
portable replay. That replay reloaded each exported phenotype and reproduced
all 256 held-out interpolation and 256 generalization decisions per seed,
including agreement with the independent scalar reference. The generic
128-step synthetic replay passed direct seeds 1, 3 and 4; seeds 0 and 2 failed
because their evolved recurrent values overflowed FP32 on that out-of-task
observation sequence. The failure is retained in
[`synthetic-replay-audit.json`](results/evidence/h2-direct-memory-acceptance-v1/rtx3070-memory/synthetic-replay-audit.json).
The learned champions passed the actual task replay, but this overflow limits
the claim of unrestricted portable robustness.

Machine-readable results are [fixed status](h2-fixed-status.json) and
[direct status](h2-direct-status.json). The [evidence index](results/evidence/index.json)
records 2,447 raw artifact hashes; they were verified after collection, and
both status checks passed again against the copied evidence. Fixed cells used
image digest `adb5659f…`; the direct cell used `1570d9ac…` to include its new
manifest. Both share engine hash `90ec21d50ecf68fef4e0c13e526bde8a469262fde40b32fd6c2d4e8b7cb934be`.
The image-matched source and frozen manifests are retained in
[`reproduction/h2-90ec`](reproduction/h2-90ec/source.json).

The [W8 shared-sequence contract check](W8-REPORT.md) subsequently passed
on CPU and both GPUs with a new source image. The [geometric visual diagnostic](H2-VISUAL-REPORT.md)
has now been measured separately on a newer image. H2/M2 remain open for the
fixed-elapsed-time comparison and integrated G2–G5 performance evidence,
including the named 1,024-node G5 cell. Q2's capacity/resource decision remains
a separate prerequisite for adaptive A0.
