# M4 evolved EMR full-generation GPU comparison

The [v1 manifest](../configs/s2-m4-g5-evolved-v1.json) fixes adaptive EMR CartPole at population 128, eight episodes, 500 steps and a physical rollout microbatch of 16. Three fresh-process replicas per profile measured the same 12-generation run; the warm-generation statistic excludes startup and final selection. CPU and each GPU ran in separate single-device profiles from the same source and dependency locks. The [v1 audit](m4-g5-evolved-v1-audit.json) verifies the shared manifest and source hashes.

| Cell | CPU median | RTX 3070 median / speedup | RTX 3060 median / speedup | G5 threshold |
| --- | ---: | ---: | ---: | --- |
| V1, microbatch 16 | 2.909 s | 2.722 s / 1.069x | 2.775 s / 1.048x | Fail on both |
| V2, microbatch 64 | 3.010 s | 2.757 s / 1.092x | 2.730 s / 1.102x | Fail on both |
| V3, bulk public-substrate transfer | 2.972 s | 1.767 s / 1.682x | 1.579 s / 1.882x | **Pass on both** |

V2 is a [versioned follow-up](../configs/s2-m4-g5-evolved-v2.json) measured after v1, with only the physical microbatch and protocol identifier changed. Its independent [audit](m4-g5-evolved-v2-audit.json) retains all nine measurements. Both versions show a repeatable GPU gain, but neither reaches the frozen 1.20x complete-generation requirement.

V3 keeps the v2 workload and microbatch. It changes public EMR phenotype extraction to transfer each substrate chunk to host once before extracting individual candidates. The [v3 audit](m4-g5-evolved-v3-audit.json) confirms one shared source hash and manifest across nine fresh-process replicas. CPU warm-generation times range from 2.961 to 3.000 s; RTX 3070 from 1.763 to 1.872 s; RTX 3060 from 1.573 to 1.654 s. Both GPU ranges are separated from CPU and both medians exceed the frozen 1.20x threshold. The **M4 G5 performance cell passes on both cards** on the v3 source.

The v1 CPU stage medians were approximately 2.621 s decode, 0.225 s rollout and 0.022 s evolution; RTX 3070 was 2.297 s decode, 0.314 s rollout and 0.008 s evolution. Increasing the rollout batch in v2 did not remove that bottleneck. In v3, CPU stage medians were 2.459 s decode and 0.406 s rollout; RTX 3070 was 1.575 s decode and 0.137 s rollout; RTX 3060 was 1.383 s decode and 0.145 s rollout. The [W6 scale report](W6-SCALE-REPORT.md) shows a separate 3.4–4.7x GPU benefit for a 2,048-node **rollout stage**. That stage-only fixture is reported alongside G5, not substituted for the evolved generation.

Each replicate's `pilot.json` and compressed history under `results/m4-g5-*-rep*/pilot-run/` or `results/m4-g5-v2-*-rep*/pilot-run/` or `results/m4-g5-v3-*-rep*/pilot-run/` retain the original timing and topology trace. These are diagnostic, source-mounted runs with no immutable image digest; they establish the stated development comparison and do not claim learning acceptance.
