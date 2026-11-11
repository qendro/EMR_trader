# M2 integrated performance gates

Measured November 10, 2026 on one Windows/WSL2 host with CPU, RTX 3070
(8 GiB gating card) and RTX 3060 (12 GiB). The [G5 manifest](../configs/m2-g5-fixed-1024-v2.json)
fixes CartPole P=128, B=8, T=500, a 1,024-node recurrent fixed substrate,
and 16-candidate rollout microbatches. Each fresh-process trial warms one
complete generation, then measures one complete generation including CPPN
decoding, planning, rollout, reduction and `tell`. Three seeds are measured
per profile. The [G5 status](m2-g5-v2-status.json) verifies every manifest,
profile, generation, recurrent edge count, memory record and timing.
All performance cells use engine hash
`c01f6ca585f9cd66cac3adf160a0f839a1d939bd99a3f714d8726d6f12a5460d`;
the status files record each image digest and config hash.

| G5 configuration | CPU median seconds [range] | RTX 3070 median [range] | GPU/CPU throughput | Gate |
|---|---:|---:|---:|---|
| V1, 4,096-row query slab | 45.12 [44.79, 51.00] | 87.59 [87.55, 88.46] | 0.52× | Failed |
| V2, 65,536-row query slab | 36.88 [36.87, 37.19] | 23.89 [23.79, 24.14] | **1.54×** | **Passed** |

V1's end-to-end generation was dominated by approximately 82 seconds of
GPU CPPN decoding, compared with 24 seconds on CPU. The
[slab probe](results/evidence/m2-g5-slab-probe-v1/) checked exact decoded-array
equality at 4,096, 16,384 and 65,536 query rows on CPU and RTX 3070 before
V2 was frozen. A single phenotype's warm GPU decode fell from 0.634 to
0.177 seconds at the largest slab; CPU went from 0.190 to 0.126 seconds.
Only the query slab changed; the tested seeded phenotype matched exactly.
V2's 1.54× gain exceeds
the 1.20× G5 threshold, and the three CPU/GPU timing ranges do not overlap.
The failed [V1 status](m2-g5-v1-status.json) remains in the evidence.
The exact run scripts and their hashes are retained in the
[V1](reproduction/m2-g5-v1/source.json) and
[V2](reproduction/m2-g5-v2/source.json) reproduction snapshots. The current
runner also validates the frozen source-manifest hash across Windows and
Linux line endings.

The RTX 3060's V2 median is 26.50 seconds [26.47, 40.46], or 1.39× CPU by
median, but its range overlaps CPU. One trial spent approximately 15.6
seconds in `tell`, versus about two seconds in the other trials. Its G5 noise
criterion is therefore **inconclusive**. The RTX 3070 is the M2 gating card;
the second card's open G5 observation is retained separately. All G5 trials
had 128 valid candidates, 1,043,450 CPPN queries per candidate, and expressed
recurrent edges.

## G2–G4

| Gate | Measured result | Status |
|---|---|---|
| G2 compilation | Zero backend compilations in 100 post-warm fixed-CPPN memory generations on CPU, RTX 3070 and RTX 3060 | Pass |
| G3 memory | Monitored physical peak 397 MiB on RTX 3070, 787 MiB on RTX 3060; each kept >1 GiB free; cgroup and allocator peaks within caps; zero swap-out growth | Pass |
| G4 controller overhead | Three matched RTX 3070 controller/driver pairs; median latency ratio 0.997 | Pass |

The [G2 audit](m2-g2-status.json) checks low-level backend compilations,
not just top-level JIT cache sizes. Final validation/selection made four
additional compilations on every profile; these are recorded separately
after the 100-generation timing window. An initial instrument run counted
them in the G2 window and was marked failed; its raw record is retained under
`m2-g2-v1`. The revised instrument marks the start of final selection and
passes on all three profiles. This is compilation stability, not another
learning acceptance run.

For G3, one additional full G5 trial on each GPU ran under the physical
[GPU and WSL telemetry](results/evidence/m2-g3-telemetry/) monitor. WSL had
at least 5.71 GB available; swap-out pages and used swap did not grow during
153 samples. The [G3 audit](m2-g3-status.json) also checks the trial's cgroup
peak, cgroup swap and allocator pool against the 6 GB host and 60% VRAM
limits. The three-trial G5 status checks those allocator/cgroup limits too.

The [G4 audit](m2-g4-status.json) compares generations 2–10 of three
fresh-process pairs at the same P=128, B=8, T=500 small-substrate CartPole
cell. It verifies matching generation scores, validity, species, topology
counts and CPPN query counts before computing the ratio. The terminal
selection generation is excluded from both warm-time samples.

Raw G2–G5 manifests, histories, status files, monitor samples and failures
are indexed under [evidence](results/evidence/index.json). M2 remains open
for the separately required fixed-elapsed-time quality comparison; these
performance gates do not substitute for that experiment.
