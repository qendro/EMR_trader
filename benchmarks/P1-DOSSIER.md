# M4/M5 acceptance dossier

This report reads the retained machine-readable audits. A failed research or performance gate stays open; implementation and passing tests do not convert it into a pass.

| Gate | Result |
| --- | --- |
| A2 adaptive memory learning | pass |
| A2 adaptive CartPole learning | pass |
| P0 partially observed control learning | pass |
| W6 2,048-node fixture on both cards | pass |
| W7 100-generation, 15-minute stability | pass |
| M4 G5 end-to-end speedup on both cards | open |
| P1 cross-backend replay and resume | pass |

## Harder-task methods

| Method | Research target | Raw 120-step seeds (diagnostic) |
| --- | --- | ---: |
| adaptive | open | 0/5 |
| direct | open | 0/5 |
| fixed | open | 1/5 |
| weight | pass | 5/5 |

The [P0 comparison](P0-COMPARISON-REPORT.md) contains paired scores and bootstrap intervals. The [W6](w6-v1-audit.json), [W7](w7-stability-v1-audit.json), [G5 v1](m4-g5-evolved-v1-audit.json), [G5 v2](m4-g5-evolved-v2-audit.json) and [P1](p1-crossbackend-v1-audit.json) audits retain measured limits. The [G5 report](M4-G5-REPORT.md) explains the versioned follow-up.

All listed gates: open.
M5 also depends on historical M2/M3 comparison and per-card gates tracked in the implementation record; this dossier does not close those by inheritance.
