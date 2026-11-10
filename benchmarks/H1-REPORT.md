# H0/H1 correctness and wiring report

Fixed-coordinate CPPN decoding, persistent layered inference, role-shaped dense blocks, and portable artifacts. This report does not close M2 or qualify its learning/performance gates. The M1 report is historical evidence for its separately retained source and image.

| Profile | Tests passed | 256 / 1024 nodes | Frozen calibration |
|---|---:|---|---|
| cpu | 54 | pass | validation-pass |
| rtx3070 | 54 | pass | validation-pass |
| rtx3060 | 54 | pass | validation-pass |

## Arithmetic and failed calibration coverage

V1 c=30 failed the CPU 1024-node decoded output preactivation (ratio 3.451 of its threshold). The initial audit had omitted this output check and is superseded. The failure remains in raw evidence. Before further qualification, v2 added single-candidate hidden products and repeated-term two-output reductions. Its frozen rule derived c=456 from CPU and RTX 3070 only; RTX 3060 takes no part in choosing it. All c-dependent calibration/large-fixture checks were rerun on every profile. Small-fan-in and stable-trajectory tolerances, FP32/HIGHEST policy and learning targets were unchanged. See `docs/h1.md` for the dated rationale and exact rerun scope.

| Profile | Nodes | Contractive trajectory max absolute error | Decoded trajectory max absolute error | Hidden / output preactivation error-to-threshold |
|---|---:|---:|---:|---:|
| cpu | 256 | 9.88e-08 | 0.000841 | 0.0256 / 0.0275 |
| cpu | 1024 | 2.22e-07 | 0.0189 | 0.0583 / 0.227 |
| rtx3070 | 256 | 2.43e-08 | 0.000841 | 0.00267 / 0.0275 |
| rtx3070 | 1024 | 2.67e-08 | 5.74e-05 | 0.00407 / 0.000571 |
| rtx3060 | 256 | 2.43e-08 | 0.000841 | 0.00267 / 0.0275 |
| rtx3060 | 1024 | 2.67e-08 | 5.74e-05 | 0.00407 / 0.000571 |

Trajectories are checked at atol=1e-4 and rtol=1e-4. The constant-score decoder fixture has large identity outputs, so its absolute error must be interpreted with the relative term. The separate contractive fixture checks unsaturated dynamics. Both run 128-step compiled scans with connected recurrent phenotypes. Small mixed-delay cases also match the independent scalar FP64 graph oracle.

At 1024 total nodes (4 inputs, 1018 hidden, 2 outputs), the decoder evaluates 1,043,450 coordinate queries in 4096-row slabs, producing 1,042,432 edges plus 1018 bias queries. Role-block arrays occupy 5,221,384 host bytes per phenotype. These are synthetic P=1/B=1 correctness fixtures, not evolved agents or the M2 G5 cell. Export/reload preserves arrays exactly. Changing weight/mask/observation values adds no compiled trajectory signature within each fixed shape. This is not the 100-generation M2 G2 gate.

## Wiring demonstrations

| Profile | CartPole smoke held-out | Memory smoke interpolation / generalization | Portable replay |
|---|---:|---:|---|
| cpu | 35.5 | 1.0 / 1.0 | pass |
| rtx3070 | 35.5 | 1.0 / 1.0 | pass |
| rtx3060 | 35.5 | 1.0 / 1.0 | pass |

All wiring runs use seed 900 and four generations at P=8. CartPole uses its smoke horizon 64; memory uses the shared 32-training-episode/17-step protocol. They archive evaluated CPPN genes and decoded phenotypes before tell. The full controller split-resume/reconstruction is also covered in the 54-test suite. No smoke score substitutes for five-seed H2 acceptance.

The next work is B0, then H2 with frozen learning manifests, integrated G9 pilots, the geometric diagnostic, W8 sequence environment and the system/encoding-isolating reports. Q2’s resource/capacity work remains a separate prerequisite for A0. No adaptive platform decoder, scale optimizer or M2 completion is claimed.
