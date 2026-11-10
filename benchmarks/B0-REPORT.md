# B0 fixed-topology weight baseline

Measured November 10, 2026. B0 evolves dense input-hidden, delayed hidden-hidden,
hidden-output and hidden-bias weights with truncation selection and Gaussian
mutation. It uses the same layered evaluator and hidden sizes as the fixed
substrates: 8 on CartPole and 4 on memory. The population is 128. The frozen
manifests allow 100 CartPole and 250 memory generations on seeds 0–4.

All 67 tests passed on CPU, RTX 3070 and RTX 3060. A matching integrated G9
pilot passed before each acceptance experiment. All 30 acceptance seeds completed
within the two-hour per-seed limit. The CPU CartPole experiment was already in
the results volume at the start of this review; its pilot, config, hardware and
source identities were verified, and its portable replay was completed here.

| Profile | CartPole held-out, seeds passing | CartPole wall seconds/seed, median [range] | Memory interpolation, seeds passing | Memory generalization | Memory wall seconds/seed, median [range] |
|---|---:|---:|---:|---:|---:|
| CPU | 500/500, 5/5 | 29.6 [28.6, 40.7] | 1.00, 5/5 | 1.00 | 31.3 [31.0, 31.5] |
| RTX 3070 | 500/500, 5/5 | 22.9 [22.8, 24.0] | 1.00, 5/5 | 1.00 | 59.0 [58.5, 59.2] |
| RTX 3060 | 500/500, 5/5 | 23.4 [23.3, 23.5] | 1.00, 5/5 | 1.00 | 59.6 [59.5, 60.2] |

Memory interpolation and longer-delay generalization each use 256 held-out
episodes. The reported score is the median across five seeds; every score in
each cell equals the displayed value. CartPole uses 20 held-out episodes per
seed. No B0 parameter sweep is recorded; one frozen setting per task was run.

The six G9 pilot warm-generation medians were 0.225/0.117 seconds on CPU,
0.165/0.195 on RTX 3070 and 0.161/0.190 on RTX 3060, ordered CartPole/memory.
End-to-end wall times include startup, compilation or cache loading, evolution,
checkpointing and final selection. The memory workload is small enough that the
GPU runs took longer than CPU; these measurements do not establish a general
throughput ranking. Peak cgroup memory across seeds was at most 916 MiB and
recorded cgroup swap was zero. The raw summaries retain per-seed process RSS
and GPU allocator statistics.

The verifier checked five ordered seeds, all declared generations, 128 valid
candidates per generation, pilot/run config and provenance equality, champion
selection, complete checkpoints and exported weights, exact reconstruction of
each champion phenotype from its archived genome, and 128-step portable replay
on every profile/task. Source engine SHA-256 is
`90ec21d50ecf68fef4e0c13e526bde8a469262fde40b32fd6c2d4e8b7cb934be`;
the image digest is recorded in [b0-status.json](b0-status.json). Config hashes,
per-cell resource maxima and compilation signature counts are there too. Raw
pilots, manifests, histories, selections, checkpoints and phenotypes are in
[indexed evidence](results/evidence/index.json), whose 2,197 file hashes passed
verification.
The exact image-matched source and configs are retained under
[`reproduction/b0-90ec`](reproduction/b0-90ec/source.json).

A separate CPU CartPole controller run stopped after generation 40, resumed in
a new container and completed generation 100. The [40/100 comparison](results/evidence/b0-resume-v1/cpu-cartpole/compare-40-100.json)
matched every native state leaf, the evaluated archive and all per-generation
quality fields against uninterrupted seed 0. Timing fields were excluded.

B0 is now a measured layered baseline. The fixed-HyperNEAT H2 acceptance runs,
W8 sequence environment, geometric diagnostic and system/encoding comparisons
remain open. These B0 scores do not establish an encoding advantage. M2 is not
accepted by this baseline alone.
