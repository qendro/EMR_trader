# A2 gated learning results and visual diagnostic

The RTX 3070 runs used immutable A2 image
`sha256:89a3b14f9695910f70f9f060a4013ebca188e890b53ec05365d875649a4fc569`
and matching integrated G9 pilots. The retained
[memory audit](a2-memory-acceptance-v1-audit.json) reports five of five
held-out scores at 1.0 (required: three). The
[CartPole audit](a2-cartpole-acceptance-v1-audit.json) reports four of five
held-out scores at or above 450 (required: three): 500, 500, 500, 355.95
and 472.55. Both A2 learning targets passed on the gating card. These runs
pre-date S0–S2 and do not validate the later layouts or memory scheduling.

The separate [60-generation adaptive visual diagnostic](a2-visual-diagnostic-v1-experiment.json) used pinned visual
image `sha256:b3d48ce6f35587811c670d3e79d8898813390eb9b02f56acd0d133c81c03de9b`.
Held-out orientation accuracies on seeds 700–702 were 0.7375, 0.7625 and
0.74375. One exceeded its 0.75 exploratory target. Its manifest declares
the experiment ungated; all three scores are retained without a pass claim.
CPU and GPU wiring smokes completed before the three-seed diagnostic.

A2's source-specific acceptance results establish the learning integration
on one GPU. M3's separate G5, second-card and comparison evidence remains
open in the implementation record.
