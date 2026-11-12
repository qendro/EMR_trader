# Standalone platform acceptance ledger — November 12, 2026

This ledger separates implemented behavior from milestone gates. The current
source image `neuroevo:m4-p0-comparison-v1` passed 116 CPU tests. Its source
and the older immutable A2/P0 acceptance images are separate provenance
lines; their measurements are never combined into a single speed ratio.

| Requirement | Evidence now | Status |
| --- | --- | --- |
| Direct, fixed CPPN, adaptive EMR and weight-only evaluation | Shared runner and existing M1/H2/B0/A2 reports | Implemented |
| Persistent memory learning | A2 RTX 3070 memory audit 5/5; M1/H2/B0 reports | Passed on reported images |
| CartPole learning | A2 RTX 3070 audit 4/5; earlier M1/H2/B0 reports | Passed on reported images |
| Adaptive topology packing | [S0 development](S0-DEVELOPMENT-REPORT.md) and lossless tests | Implemented; long compilation gate open |
| Sparse alternate runtime | [S1 decision](S1-DECISION.md), parity tests and opt-in execution | Implemented; dense remains default |
| Memory scheduling | [S2 development](S2-DEVELOPMENT-REPORT.md), explicit budget refusal | Implemented; measured scale gate open |
| 2,048-node, P=256 W6 fixture | [W6 script](../scripts/w6_fixture.py), small CPU check | Per-card full fixture queued |
| W7 stability | No 15-minute single-process fixed-shape audit | Open |
| Harder partially observed control | [P0 protocol](../docs/p0.md), paired random baseline, exact G9 pilot | Five-seed acceptance and three comparison methods running/queued |
| M4 G5 and both-card gates | No isolated matched end-to-end CPU/GPU M4 cell | Open |
| Artifact portability and resume | M1/A2 smoke evidence | Passed for recorded cases; M5 replacement check open |
| Full cross-backend current-source regression | 116 CPU tests on image | Both GPU profiles open |

The final M5 target is intentionally not declared complete while W6/W7,
per-card performance, the harder-task paired quality interval and replacement
checks are outstanding. The P0 report script preserves failed research
results, including any method that does not meet the frozen target.
