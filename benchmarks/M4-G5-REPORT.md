# M4 evolved EMR full-generation GPU comparison

The [v1 manifest](../configs/s2-m4-g5-evolved-v1.json) fixes adaptive EMR CartPole at population 128, eight episodes, 500 steps and a physical rollout microbatch of 16. Three fresh-process replicas per profile measured the same 12-generation run; the warm-generation statistic excludes startup and final selection. CPU and each GPU ran in separate single-device profiles from the same source and dependency locks. The [v1 audit](m4-g5-evolved-v1-audit.json) verifies the shared manifest and source hashes.

| Cell | CPU median | RTX 3070 median / speedup | RTX 3060 median / speedup | G5 threshold |
| --- | ---: | ---: | ---: | --- |
| V1, microbatch 16 | 2.909 s | 2.722 s / 1.069x | 2.775 s / 1.048x | Fail on both |
| V2, microbatch 64 | 3.010 s | 2.757 s / 1.092x | 2.730 s / 1.102x | Fail on both |

V2 is a [versioned follow-up](../configs/s2-m4-g5-evolved-v2.json) measured after v1, with only the physical microbatch and protocol identifier changed. Its independent [audit](m4-g5-evolved-v2-audit.json) retains all nine measurements. Both versions show a repeatable GPU gain, but neither reaches the frozen 1.20x complete-generation requirement. M4 G5 and the dependent M5 gate remain open.

The v1 CPU stage medians were approximately 2.621 s decode, 0.225 s rollout and 0.022 s evolution; RTX 3070 was 2.297 s decode, 0.314 s rollout and 0.008 s evolution. Host-side discovery and conversion dominate this small evolved envelope. Increasing the rollout batch did not change that bottleneck. The [W6 scale report](W6-SCALE-REPORT.md) shows 3.4–4.7x GPU benefit for a separate 2,048-node **rollout stage**; that fixture is not the evolved EMR generation required by G5.

Each replicate's `pilot.json` and compressed history under `results/m4-g5-*-rep*/pilot-run/` or `results/m4-g5-v2-*-rep*/pilot-run/` retain the original timing and topology trace. These are diagnostic, source-mounted runs with no immutable image digest; they establish the stated development comparison and do not claim learning acceptance.
