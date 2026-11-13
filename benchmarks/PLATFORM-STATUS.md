# Standalone platform acceptance ledger — November 12, 2026

This ledger separates implemented behavior from milestone gates. The current
source image `neuroevo:m4-p0-comparison-v1` passed 116 CPU tests. The later
source-mounted P1 v2 validation passed 117 tests on each of CPU and both GPUs. Its source
and the older immutable A2/P0 acceptance images are separate provenance
lines; their measurements are never combined into a single speed ratio.

| Requirement | Evidence now | Status |
| --- | --- | --- |
| Direct, fixed CPPN, adaptive EMR and weight-only evaluation | Shared runner and existing M1/H2/B0/A2 reports | Implemented |
| Persistent memory learning | A2 RTX 3070 memory audit 5/5; M1/H2/B0 reports | Passed on reported images |
| CartPole learning | A2 RTX 3070 audit 4/5; earlier M1/H2/B0 reports | Passed on reported images |
| Adaptive topology packing | [S0 development](S0-DEVELOPMENT-REPORT.md), lossless tests and [W7](W7-STABILITY-REPORT.md) warm-signature stability | Implemented; W7 observed no new warm training signatures |
| Sparse alternate runtime | [S1 decision](S1-DECISION.md), parity tests and 27-cell single-device crossover map on each profile | Implemented; dense remains default |
| Memory scheduling | [S2 development](S2-DEVELOPMENT-REPORT.md), explicit budget refusal, [W6](W6-SCALE-REPORT.md) zero-swap scale and [W7](W7-STABILITY-REPORT.md) stability | Implemented with measured W6/W7 evidence |
| 2,048-node, P=256 W6 fixture | [W6 scale report](W6-SCALE-REPORT.md): both GPUs passed piloted sparse and 400-input dense fixtures without swap | Synthetic scale gate passed on both cards; evolved envelope remains separate |
| W7 stability | [W7 report](W7-STABILITY-REPORT.md): 380 generations, 966.95 s, 0.98% warm RSS growth, no swap | Passed on RTX 3070 |
| Harder partially observed control | [P0 four-method comparison](P0-COMPARISON-REPORT.md): weight-only median paired gain 95.27%, adaptive 6.86%, direct 4.34%, fixed 3.20% | Harder-task quality target passed via weight-only; decoder-specific adaptive/direct/fixed limitations retained |
| M4 G5 and both-card gates | [M4 G5 report](M4-G5-REPORT.md): v1/v2 failed; v3 1.682x/1.882x on RTX 3070/3060 with separated replicate ranges | V3 passed on both cards |
| Artifact portability and resume | [P1 v2 audit](p1-crossbackend-v2-audit.json): replay actions identical on three profiles; 3-generation checkpoint continuation after container replacement | Passed for tested artifact and v3 source |
| Full cross-backend current-source regression | [P1 v2 audit](p1-crossbackend-v2-audit.json): 117/117 tests on CPU, RTX 3070 and RTX 3060 | Passed on v3 source |

The final M5 target remains open because M2 fixed-time comparisons are
incomplete and earlier milestone dependencies are not inherited from later
diagnostic runs. The P0
report preserves failed research results for adaptive, direct and fixed methods.
