# A2 adaptive integration: development evidence

**Status:** integration smoke passed; acceptance open. The four final-source
three-generation smokes used population 8, depth 1 and the shared persistent
CartPole/memory evaluators. The [audit](a2-smoke-v1-audit.json) checks complete
histories and exact CPU/RTX 3070 quality, selected policy and per-genome
topology counts. Weighted graph hashes differ across backends, consistent with
small FP32 decode differences; exact weighted-graph equality is not claimed.

| Task | Distinct active hidden counts | Expressed edge range | Distinct weighted graphs | Smoke held-out score | Acceptance target |
| --- | --- | ---: | ---: | ---: | ---: |
| Memory | 4, 8, 12, 16, 20 | 28–450 | 8 | 1.000 | 0.900 in ≥3/5 seeds |
| CartPole | 4, 8, 12, 16, 20 | 39–510 | 8 | 42.375 | 450 in ≥3/5 seeds |

The first memory smoke, `a2-memory-smoke-v1`, stopped on an honest recurrent
capacity refusal: one genome had 390 connections against a 256-slot cap. Its
manifest and partial artifacts are retained. Version 2 raised the cap to 512
and node capacity to 24 before the completed smokes. No connection was
silently truncated. The runner's generation rows include actual input,
output and recurrent edge counts, active hidden counts and graph hashes.

The CPU memory run resumed in a separate process after generation 0. Against
an uninterrupted run it matched all 12 native state leaves, the quality
history and final selection exactly. The selected compact memory phenotype
also passed the backend-independent 128-step portable replay. The full CPU
suite passed 98 tests; the native ask/decode/persistent-rollout/tell test is
included. The development GPU smokes passed on the RTX 3070. All raw bounded
JSON, NPZ and checkpoint artifacts are hashed in the
[evidence index](results/evidence/a2-development-v1/index.json).

## Frozen small-substrate pilot

The [memory](../configs/a2-memory-acceptance-v1.json) and
[CartPole](../configs/a2-cartpole-acceptance-v1.json) manifests were written
before their G9 pilots. Both use population 128, depth 1, five acceptance
seeds and the existing 7,200-second per-seed budget. They differ in task
interface and capacity: memory has at most 23 nodes/512 edges, CartPole at
most 26 nodes/1,024 edges. The 512-slot recurrent discovery cap refuses
overflow. This small-substrate envelope is a deliberate comparison cell;
depth 4 remains a later scale cell.

The immutable `neuroevo:a2-v1` image is
`sha256:89a3b14f9695910f70f9f060a4013ebca188e890b53ec05365d875649a4fc569`.
It ran on the selected RTX 3070 with no source mount. Its integrated 12-generation
pilots include cold setup, production checkpointing and final validation:

| Pilot | Warm-generation median | Pilot wall time | G9 budget result |
| --- | ---: | ---: | --- |
| Memory | 1.623 s | 49.136 s | Pass; 250-generation manifest fits with reserve |
| CartPole | 1.590 s | 50.235 s | Pass; 100-generation manifest fits with reserve |

The pilots are workload/budget evidence, not learning acceptance. The initial
source-mounted pilot files are retained separately; their hardware identity
does not satisfy the acceptance gate. The immutable pilot manifests, raw
histories, selections and checkpoints are in the evidence index. Acceptance
must use this exact image/config/hardware provenance or rerun the pilot.

## Remaining A2 work

Run and audit all five memory and CartPole acceptance seeds on the gating
card, validate the required CPU and second-card cells, and repeat the 5×5
geometric diagnostic for adaptive discovery. Then produce fixed-evaluation
and fixed-elapsed comparisons with B0 and fixed HyperNEAT, including the
stopped M2 fixed-time cells as missing evidence until completed or explicitly
decided. Measure end-to-end discovery, conversion, rollout and memory gates
at the declared adaptive envelope. The CartPole smoke score is far below the
acceptance threshold; the 12-generation pilot reaches a training best of
415/500, but held-out acceptance is still unknown.
