# Standalone neuroevolution platform

This repository currently contains the development blueprint only. No engine implementation or performance qualification has been completed.

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

Qualify each GPU independently using the same pinned image and workload definitions. Record actual GPU variants, VRAM, host RAM, driver and WSL versions during setup; these machines have not been inspected or benchmarked. Start with one GPU per run. Combined multi-GPU execution is a later extension, not a requirement for either card to work. One card is named the gating card at Q0 and decides when the next milestone may start; the second card still passes the full gate set before the platform is accepted.

## First action when implementation is authorized

Execute tasks Q0–Q2 in the [task breakdown](docs/tasks.md): define the pinned Docker image and Compose run profiles, verify GPU passthrough and JAX execution, and benchmark bounded upstream TensorNEAT/EMR workloads on both cards. Then implement the stateful vertical slice described in [Milestone 1](docs/milestone-1.md). This remains a documentation-only delivery; no container build, dependency installation or training has been performed.

## Revision history

Every number introduced by a revision is a proposed gate, not a measurement.

- **Revision 4, November 10, 2026.** Q2 is an upstream screening test on a frozen manifest and no longer stands in for the research budget, which G9 sets per seed from a pilot on the integrated workload. Added the shared memory-task manifest, the comparison protocol, a fully specified W8 reducer, named G5 gate cells and a calibration policy that needs only the gating card. Removed superseded wording.
- **Revision 3, November 10, 2026.** Dense connectivity separates immediate and delayed matrices and masks, including export and mixed-delay fixtures. Float32 matrix-product precision is pinned, and calibrated tolerances are empirical thresholds.
- **Revision 2, November 10, 2026.** Layered profile for substrates and synchronous profile for direct recurrent genomes; two graph encodings; a declared fitness-comparability policy; the fixed-topology baseline and added diagnostics; the gating card; in-container GPU selection under WSL; Gymnax locked in Q0.
