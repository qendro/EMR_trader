# Standalone neuroevolution platform

The first stateful implementation is in `src/neuroevo`, with a pinned Linux Docker environment, contract/differential tests, a TensorNEAT adapter, memory and CartPole evaluation, checkpoints and portable inference. Milestone acceptance remains separate from implementation. See the [implementation record](docs/implementation.md) for measured results and open gates.

M1's required gates passed on its retained source: 44 tests on CPU and each GPU, five of five CartPole seeds at 500/500 on every profile, exact native CPU resume, and portable replay. The [M1 benchmark report](benchmarks/REPORT.md) retains failures and timing variation.

H0/H1 add fixed-coordinate CPPN development, persistent layered inference and portable role-shaped blocks. Their connected 256/1,024-node execution fixtures passed on CPU and each GPU; see the [H1 report](benchmarks/H1-REPORT.md). B0 adds a fixed-topology weight baseline. Its five CartPole and five memory seeds per profile meet their held-out targets with matching G9 pilots; see the [B0 report](benchmarks/B0-REPORT.md). H2's small-substrate CartPole and memory learning passed the gating-card seed targets, with a direct-memory comparison; see the [H2 report](benchmarks/H2-REPORT.md). The [W8 contract check](benchmarks/W8-REPORT.md) passed on CPU and both GPUs. The [geometric visual diagnostic](benchmarks/H2-VISUAL-REPORT.md) is measured on the RTX 3070; the 72-test CPU suite and new visual tests on both GPUs passed. [M2's G2–G5 performance checks](benchmarks/M2-PERFORMANCE-REPORT.md) pass on the gating card after a bounded query-slab revision. M2 remains open for the fixed-elapsed-time quality comparison. Q2's capacity/resource work and M3–M5 remain open.

The project will develop a general-purpose, accelerator-oriented neuroevolution platform, progressing from TensorNEAT to fixed-substrate HyperNEAT and adaptive EMR-HyperNEAT. Persistent temporal memory is part of the first working milestone. Financial trading is a later application, outside the standalone platform.

## Development blueprint

Read the documents in this order:

1. [Refined architecture](docs/architecture.md): component boundaries, contracts, graph semantics, dependencies, and one-generation lifecycle.
2. [Implementation roadmap](docs/roadmap.md): phased deliverables, dependencies, gates, and definition of done.
3. [Engineering tasks](docs/tasks.md): implementation-sized work packages and dependency ordering.
4. [Testing and benchmarking](docs/benchmarks.md): correctness, environments, workload manifests, the comparison protocol, hardware qualification, profiling, and measurable performance gates.
5. [Risk register](docs/risks.md): failure modes, mitigations, and decisions requiring experiments.
6. [First development milestone](docs/milestone-1.md): exact scope and demonstration requirements.
7. [Windows and Docker execution](docs/execution-environment.md): host prerequisites, container boundaries, CUDA dependencies, and GPU profiles.

## Current target

The deployment targets are NVIDIA RTX 3060 and RTX 3070 GPUs on Windows. All project execution runs in Linux Docker containers through Docker Desktop’s WSL 2 backend. NVIDIA GPU passthrough is available on that backend. [Docker GPU support](https://docs.docker.com/desktop/features/gpu/)

Both cards have been inventoried and qualified independently on this host: RTX 3060 12 GiB and RTX 3070 8 GiB, driver 591.74. The RTX 3070 gates progression. Each container selects exactly one verified card before importing JAX. Combined multi-GPU execution remains a later extension. See the [measured report](benchmarks/REPORT.md) and [machine-readable status](benchmarks/status.json) for learning results and open gates.

## Run the implementation

```powershell
docker compose build cpu
$env:NEUROEVO_IMAGE_DIGEST=(docker image inspect neuroevo:release --format '{{.Id}}').Trim()
docker compose run --rm cpu test -q
docker compose run --rm rtx3070 qualify --output /results/q1/rtx3070.json
docker compose run --rm cpu oracle --output /results/oracle/cpu.json
docker compose run --rm cpu run --config configs/m1-smoke-v1.json --output /results/smoke
```

Results and checkpoints persist in the `neuroevo-results` Docker volume. The RTX 3070 (8 GiB) is the gating card; the RTX 3060 (12 GiB) is validated independently. [Commands, source pins, design decisions and acceptance procedure](docs/implementation.md). Smoke results never substitute for the five-seed acceptance protocol. M2–M5 remain gated on the preceding milestone.

For the H0/H1 wiring demo, build the current source as `neuroevo:h1` and run `configs/h1-fixed-cartpole-smoke-v1.json` or `configs/h1-fixed-memory-smoke-v1.json` through the same `run` entry point. `scripts/h1_verify.ps1` runs the frozen arithmetic and large-fixture checks plus both wiring demos/replays on all profiles. Those demos are explicitly smoke evidence, not M2 learning acceptance.

For the measured B0 baseline, use `scripts/b0_runs.ps1 -Image neuroevo:b0 -Profile rtx3070 -RunRoot /results/b0-reproduction` after building that image from this source. The script runs a matching G9 pilot, five acceptance seeds and portable replay for both tasks. `-Task memory` selects one task. `scripts/h2_fixed_runs.ps1` and `scripts/h2_direct_memory_runs.ps1` reproduce the H2 learning cells. The [H2 preparation record](benchmarks/H2-PREP.md) retains their frozen setup and pilot history.

For the ungated 5×5 visual diagnostic, build the current source as `neuroevo:visual` and run `scripts/h2_visual_runs.ps1 -Image neuroevo:visual -Profile rtx3070`. It runs three disjoint seeds for B0, direct genomes and fixed CPPN; `scripts/h2_visual_status.py` audits archived champions and held-out decisions.

For the full procedure, `scripts/qualify.ps1`, `scripts/acceptance.ps1`, `scripts/resume.ps1` and `scripts/performance.ps1` launch bounded containers and record the selected image digest. Pass `-Image neuroevo:release` after building with Compose. Curated measurements, failed cells, checkpoints and portable champions are in [benchmarks/results/evidence](benchmarks/results/evidence); their byte hashes are indexed there. The original Q2 recurrent screen overflowed its frozen connection cap on both cards; its proposed capacity promotion has not run. Its [capacity promotion preparation](benchmarks/Q2-CAPACITY-PREP.md) is ready for measurement after the M2 elapsed comparison completes.

## Revision history

Every number introduced by a revision is a proposed gate, not a measurement.

- **Revision 4, November 10, 2026.** Q2 is an upstream screening test on a frozen manifest and no longer stands in for the research budget, which G9 sets per seed from a pilot on the integrated workload. Added the shared memory-task manifest, the comparison protocol, a fully specified W8 reducer, named G5 gate cells and a calibration policy that needs only the gating card. Removed superseded wording.
- **Revision 3, November 10, 2026.** Dense connectivity separates immediate and delayed matrices and masks, including export and mixed-delay fixtures. Float32 matrix-product precision is pinned, and calibrated tolerances are empirical thresholds.
- **Revision 2, November 10, 2026.** Layered profile for substrates and synchronous profile for direct recurrent genomes; two graph encodings; a declared fitness-comparability policy; the fixed-topology baseline and added diagnostics; the gating card; in-container GPU selection under WSL; Gymnax locked in Q0.
