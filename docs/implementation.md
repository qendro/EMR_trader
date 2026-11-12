# Implementation record — November 10, 2026

This file tracks implementation evidence; the blueprint documents define gates.
M1's accepted source is retained separately. H0/H1 and B0 are implemented and verified; see [h1.md](h1.md), the [H1 report](../benchmarks/H1-REPORT.md), and the [B0 report](../benchmarks/B0-REPORT.md). H2's small-substrate learning and direct-memory system cell are measured in the [H2 report](../benchmarks/H2-REPORT.md). Q2's declared capacity/resource and synchronized stage screens now pass; see the [Q2 decision](decisions.md). M2 acceptance and M3–M5 remain open. No adaptive-platform acceptance is implied by a dependency or wiring smoke test.

The sections below record the M1 increment and its original decisions. H0/H1's qualified engine is `e4fbdd7588fd6008455fc109d12996fa21097bd40d07849f83149d2493b113b6`, built and tested as `neuroevo:h1` (image `5c567eb307576b52543497c16f7a6fdaedb333272067b4d774b508dbcc29fd46`). B0/H2's measured learning engine is `90ec21d50ecf68fef4e0c13e526bde40b32fd6c2d4e8b7cb934be`, retained under `benchmarks/reproduction/h2-90ec`. The current W8 engine is `54b32a9f470a556e28c6e537a4b1a5d342ddd15a0a58b993a864392057f1d0e7`, tested with 69 tests on all three profiles. Historical M1 results retain engine `8d3aba65…`, tag `neuroevo:m1-qualified`, and exact source under `benchmarks/reproduction/m1-8d3`. Native checkpoints require their recorded source hash.

The initial [A0 configuration boundary](a0.md) now validates a narrow adaptive
recipe, translates it through the pinned EMR public config API, and rejects
unsupported options. Phenotype extraction and adaptive rollout are A1/A2 work.

## Q0 decisions

Both target cards share this Windows host: RTX 3060 12,288 MiB and RTX 3070 8,192 MiB, driver 591.74. The **RTX 3070 gates progression**. Host RAM is 17,100,693,504 bytes, with six logical CPUs. Docker Desktop exposes approximately 7.7 GiB to Linux; containers use four CPUs and a 6 GiB memory limit. These limits leave Windows headroom and prohibit assuming pooled VRAM. See `benchmarks/results/host-inventory.json`.

Docker Desktop was initially stopped. Starting it restored the Linux daemon; `ubuntu:24.04 nvidia-smi` with `--gpus all` listed both cards. This establishes passthrough only. Windows Python 3.14 is not used for project execution.

The digest-pinned Python 3.11.14 base installs the exact hash-checked Linux resolution in `locks/requirements.txt`. The input constraints are separate from the resolved lock. Gymnax 0.0.9, Flax 0.10.6 and EMR's JAX 0.6.1/NumPy 1.26.4 resolve in one lane. TensorNEAT's eager imports require its declared simulator dependencies; these are installed but physics workloads are not implemented. The older Gymnax lane was selected deliberately instead of allowing an environment upgrade to change JAX.

Complete source archives retain upstream source and GPL notices. `locks/sources.json` pins both commits and archive checksums. The image verifies checksums before extracting/installing. No upstream evolution code is replaced. The native DefaultConn uses endpoint pairs as connection identity, and nodes retain their semantic keys. We preserve both native arrays without manufacturing a second innovation-number scheme. DefaultNode computes bias + response × sum and bypasses activation on output nodes. Hidden activation is restricted to ordinary tanh; outputs remain identity. Input/output transforms are absent and unsupported functions are rejected. The recurrent genome selects the upstream cycle-capable mutation path, while project inference supplies explicit persistent K=1 state (upstream recurrent inference resets per input).

The runtime is FP32 with the global float32 matrix-product policy set before tracing, and project dot products explicitly request HIGHEST. The scalar reference independently accumulates FP64 from FP32-representable values. Masked padding is selected out before arithmetic. Both encodings retain separate immediate and delayed matrices/masks. JAX layered execution remains an M2 task; its reference is available now.

## Frozen experiment definitions

`configs/m1-smoke-v1.json` is development evidence only. `configs/m1-acceptance-v1.json` preserves all five original acceptance seeds and requires a separate integrated G9 pilot. The pilot includes cold compilation, production logging/checkpoint frequency and final validation/held-out evaluation. The two-hour budget is **per seed**, from process start to final report; five seeds receive ten hours. Warm-generation budgets are never inferred from Q2 independent samples. Physical microbatch changes cannot change semantic RNG keys.

`configs/q2-v1.json` freezes the actual upstream revisions, dataset generation, dimensions, CPPN capacities/operators, substrate coordinates, discovery thresholds, independent-sample settling, sample counts and timing boundaries. A dataset artifact is written before timing. Feedforward and recurrent runs are separate fresh processes; the latter disables slot caching and uses per-genome discovery. `configs/w5-v1.json` references those exact decoder settings but declares sequential CartPole rollouts and persistent layered timing. It is an M3 manifest, not an implemented adaptive runner. Both fixed-evaluation-budget and fixed-wall-time comparisons are required, and matched layered controls precede encoding-specific conclusions.

`configs/memory-v1.json` is the common M1 diagnostic/M2/M3 protocol: 32 balanced training episodes, 64 validation, 256 interpolation, and separate 256 generalization episodes; seeded distractors include the query step. RNG namespaces and final-generation validation selection are frozen. The deterministic 10-episode oracle is separate and has no distractors.

`configs/w8-v1.json` specifies Welford statistics and Chan chunk merging, empty-chunk identity, population variance, a 32-sample minimum, additive 0.01 floor, constant and nearly constant fixtures, numerical tolerances and fixed chunk partitions. Reducer unit checks are independent work in V4; they do not establish that the M2 W8 environment is complete.

Larger-fan-in numerical calibration uses only the CPU profile and RTX 3070 gating card, before acceptance at those sizes. The resulting coefficient and fixture hashes are frozen. RTX 3060 results are a separate validation against that coefficient, never implied by calibration. A failure requires a dated decision identifying every earlier G1 result to rerun. Until calibration exists, only the specified fan-in ≤32 and stable 128-step thresholds apply.

## Run boundary and artifacts

All commands below run inside the image. Normal runs use immutable image source; tests during development may explicitly mount `/app/src` and set `PYTHONPATH`, which is recorded as development evidence. Named volumes `neuroevo-results` and `neuroevo-cache` survive container replacement. Copy curated evidence out to `benchmarks/results` after execution. Never place acceptance evidence only in a disposable container layer.

```powershell
docker compose build cpu
$env:NEUROEVO_IMAGE_DIGEST=(docker image inspect neuroevo:release --format '{{.Id}}').Trim()
docker compose run --rm cpu qualify --output /results/q1/cpu.json
docker compose run --rm rtx3070 qualify --output /results/q1/rtx3070.json
docker compose run --rm rtx3060 qualify --output /results/q1/rtx3060.json
docker compose run --rm cpu test -q
docker compose run --rm rtx3070 test -q
docker compose run --rm rtx3060 test -q
docker compose run --rm cpu oracle --output /results/oracle/cpu.json
docker compose run --rm cpu run --config configs/m1-smoke-v1.json --output /results/smoke
docker compose run --rm rtx3070 benchmark --config configs/m1-acceptance-v1.json --output /results/m1-3070
docker compose run --rm rtx3070 run --config configs/m1-acceptance-v1.json --output /results/m1-3070
docker compose run --rm cpu replay --phenotype /results/smoke/seed-900/champion.npz --output /results/smoke/replay.json
docker compose run --rm rtx3070 upstream --mode feedforward --smoke --output /results/q2-smoke/rtx3070-ff
docker compose run --rm rtx3070 upstream --mode recurrent --seed 700 --output /results/q2/rtx3070-rnn-700
```

Repeat full Q2 for seeds 700–702 and both modes/cards in separate fresh processes with cold per-run cache directories. A `--smoke` run is labelled and never closes Q2. A failed screening leaves A0 gated until the documented fallback decision. Q2's compile ceiling also needs an external process timeout for a blocked compiler; post-return duration checks cannot interrupt compilation.

Checkpoint files contain all native species/history/population/RNG state, configuration, source/package identity and the evaluated archive. Atomic replacement follows file and directory fsync. Only trusted local checkpoints may be loaded (pickle); portable inference uses validated `allow_pickle=False` NPZ. Checkpoints reject source/configuration/schema changes. Resume is at a generation boundary. `--stop-after 40` and a second process with `--resume` exercise the required 40/100 split.

## Evidence status

The [measured report](../benchmarks/REPORT.md) and [JSON status](../benchmarks/status.json) summarize actual runs. [Raw evidence](../benchmarks/results/evidence) includes per-generation rows, manifests, tests, failures, selected/evaluated phenotypes and complete native checkpoints. `index.json` records their byte hashes. Status values in these artifacts take precedence over source existence or the proposed gate values in the blueprint.

**M1's required gates pass on the final source.** CPU and both GPUs each passed 44 tests and all five CartPole acceptance seeds scored 500/500. All three profiles completed 100 post-warm-up generations without backend compilation. Gating-card G4 is 1.033× matched minimal-driver latency, below 1.20×. The native 40/100 CPU resume matches every backend-state leaf, evaluated archive and history quality field; portable closed-loop replay matches all 500 actions and rewards, with maximum output error below 9.38e-8. All five diagnostic memory champions score 100% on interpolation and separately reported generalization. Q2's recorded overflow and incomplete resource/stage audit remain open. H0 is the next implementation package; later milestones are not included in this first increment.

| Work | Implemented/tested | Measurement boundary |
|---|---|---|
| Q0/Q1 | Root Docker build, source/license verification, hash-locked dependencies, strict card selection, compiled primitive matrix and native full generation | Independently executed on CPU, RTX 3070 and RTX 3060; no CPU fallback |
| V0–V2 | One 32-node/128-edge capacity, DAG and K=1 synchronous FP32 runtime, independent scalar FP64 semantics for all three profiles, separate immediate/delayed matrices | 44 contract tests per profile; 16 independent 128-step differential fixtures per profile |
| V3 | Unmodified TensorNEAT operators through a narrow adapter, native state/IDs, stale-score rejection, stagnation observer | Same-backend operator/elite parity; state and fitness association checks |
| V4 | Compiled time scans, candidate/episode vectorization, semantic RNG keys, microbatch tails, termination/truncation, memory continuation and Welford/Chan reducer | Oracle/reset 100%/50%; batch/permutation/chunk/invalid-candidate tests; 10,000-sample reducer fixtures |
| V5 | Run/benchmark/replay commands, archive before tell, validation-only selection, fsync/atomic checkpoints, strict portable NPZ | Frozen five-seed CartPole protocol; separate-container 40/100 resume; closed-loop portable replay; G2/G4 results reported separately |
| Q2 | Original EMR screening wrapper and separately versioned raw upstream fixed/direct baseline drivers, frozen workloads and bounded watchdog | Both cards' recurrent seed 702 failed at 13,424 connections against cap 10,000; native v1 direct cells failed on all profiles; native v2 adds an explicit invalid-candidate boundary; full resource/stage audit remains open |
| H0/H1 | Implemented in the next increment; see the H1 report | Fixed coordinate decoder, layered graph/role-block runtime, mixed-delay portability, 256/1,024-node execution and versioned larger-fan-in calibration pass on each profile; H2 is not accepted |
| B0 | Implemented and measured; see the [B0 report](../benchmarks/B0-REPORT.md) | Fixed-topology layered baseline, 67 tests on each profile, matching G9 pilots and five-seed CartPole/memory results on CPU and both GPUs; all held-out targets met |
| H2 | Small-substrate CartPole/memory acceptance and direct-memory system comparison measured on RTX 3070; see [H2 report](../benchmarks/H2-REPORT.md) | Fixed CPPN passed 4/5 CartPole and 5/5 memory seeds with matching G9 pilots. Direct recurrent memory passed 4/5. [W8](../benchmarks/W8-REPORT.md) passed its three-profile contract check. Geometric diagnostic, fixed-time comparison and integrated performance gates remain open |
| A0/A1 | Configuration adapter and evaluated adaptive phenotype conversion implemented; see [A0](a0.md) and [A1](a1.md) | Pinned-cache graph/output parity on small synthetic and real EMR generations; A2 persistent adaptive evolution and later acceptance remain open |

The supported function vocabulary is sum aggregation, tanh hidden nodes and identity outputs, with upstream bias/response semantics and no input/output transform. Graph planning preserves both delay-labelled encodings; JAX layered inference explicitly rejects use until H1. Synchronous paths through one/two hidden nodes have one/two observations of latency. Native recurrent upstream inference, which resets and settles independently per sample, is never described as the platform's persistent K=1 rollout.

The isolated-node boundary is explicit: the platform's empty sum is zero, so an isolated node evaluates its bias. Pinned TensorNEAT `apply_aggregation` instead returns NaN when all incoming terms are NaN. Evolution operators remain unmodified, but upstream inference parity applies to compatible connected fixtures. Raw native Q2 direct inference exposed this edge case; v1 failure artifacts remain intact and the separately frozen v2 protocol rejects nonfinite candidates, records their counts and saves a bounded first-invalid-genome reproducer.

The raw upstream baseline matrix is complete on CPU and each card for seeds 700–702: nine direct v1 cells failed and nine fixed v1 cells completed; all 18 v2 cells completed under the explicit invalid-output boundary. The saved invalid genome reproduces the upstream NaN outputs. All 49 persisted Q2 datasets have identical FP32 input/target arrays. These baseline measurements retain independent-sample semantics and are not platform fixed/adaptive decoder acceptance. The Q2 capacity/resource screen now passes 12/12 cells; isolated stage timing remains open.

`m1-current-*` uses the final engine hash `8d3aba6552b6872ab47ddca1c1992ee839ce5339c9f48ba221ba780525fddebb`, root-built image `7172c6d3…`. Earlier `m1-final-*` used `e03c0613…`, image `1a6b04f3…`, and all five seeds scored 500 on CPU and each card. Those earlier measurements are retained with their exact source under `benchmarks/reproduction/m1-e03`; use retained local image `neuroevo:final` for its native checkpoints. Final strict-validation/overflow changes received fresh pilots and separate acceptance runs. No acceptance seed or threshold was changed after inspecting scores. For M1 CartPole and memory, the tuning count is one parameter configuration each; development smoke runs are not a search over acceptance parameters. Q2 native v1/v2 are separately recorded evaluation protocols, with unchanged operator parameters and an explicit invalid-fitness boundary in v2.

G9 pilots are fresh processes with persistent compilation caches, whose initial entry counts are recorded in `runtime.json`. Their first-call timings include whatever compilation/cache loading actually occurs; they are not empty-cache measurements. Some learning and Q2 runs overlapped on the shared host. Their measured times establish budget feasibility, and the report identifies them accordingly. G4/G5 comparisons use separate sequential measurement cells after other project experiments stop. No W7 sustained stability or large-graph acceleration claim is made.

M1 closes only after A/B/C, the reported D diagnostics and gating-card G1–G4 pass. M2–M5 are not pre-created or declared complete. The original Q2 capacity failure remains retained; the revised capacity/resource screen passes on both GPUs, with isolated stage timing still open. The next Q2 manifest is `q2-capacity-v2.json` (32,768 slots, refuse overflow), measured and audited; see [decisions.md](decisions.md). Upstream native stage timers are not synchronized isolated stage measurements, and their mean/std graph summaries do not constitute complete active-size distributions.

### Reproduce and collect

These PowerShell helpers only orchestrate Linux containers. Build with Compose first and pass the image explicitly:

```powershell
./scripts/qualify.ps1 -Image neuroevo:release -RunRoot /results/qualification-reproduction
./scripts/acceptance.ps1 -Image neuroevo:release -Profile cpu -RunPrefix m1-reproduction
./scripts/acceptance.ps1 -Image neuroevo:release -Profile rtx3070 -RunPrefix m1-reproduction
./scripts/acceptance.ps1 -Image neuroevo:release -Profile rtx3060 -RunPrefix m1-reproduction
./scripts/performance.ps1 -Image neuroevo:release
```

`scripts/resume.ps1` demonstrates the separate-container split against `m1-current-cpu`; `scripts/diagnostics.ps1` runs all four diagnostic manifests on CPU. Choose new output directories before independent repetitions; do not overwrite recorded experiments. The named result/cache volumes retain artifacts when the container is removed. To replay a checked-in portable champion without an evolution backend import:

```powershell
docker run --rm -e NEUROEVO_BACKEND=cpu --mount "type=bind,source=${PWD}\benchmarks\results\evidence,target=/evidence,readonly" --mount type=volume,source=neuroevo-results,target=/results neuroevo:release replay --phenotype /evidence/m1-current-cpu/seed-0/champion.npz --output /results/portable-replay.json
```

`scripts/q2_native.ps1` reproduces the original native v1 matrix, including its direct failures. `scripts/q2_native_v2.ps1` runs the separately frozen finite-output protocol. Its cache starts empty per profile/mode for seed 700 and is retained for 701/702; recorded initial counts distinguish those cases. `scripts/replay_native_invalid.py` replays the saved raw-upstream invalid fixture; it deliberately imports TensorNEAT, unlike portable platform inference. The NaN replay and the capacity failures remain negative evidence, not platform inference results.

The [Q2 capacity preparation](../benchmarks/Q2-CAPACITY-PREP.md) adds an external cold-compile watchdog, per-cell physical GPU monitoring and complete per-genome substrate size records. It ran after the user stopped M2 at 17/20 measured seeds. The [Q2 capacity report](../benchmarks/Q2-CAPACITY-REPORT.md) and [synchronized stage report](../benchmarks/Q2-STAGE-REPORT.md) each pass 12/12 cells for the declared upstream envelope. A0 adapter work can begin; M2 remains incomplete.

`collect_results.py` copies bounded JSON/XML/NPZ/checkpoint files out of the named volume and omits compilation caches. `report.py` generates the report from persisted measurements, retaining missing/failed cells. Run both inside the image with `/results` read-only and a host output bind mount. Source snapshots under `benchmarks/reproduction` are scientific provenance, not alternate packages used by the current implementation. Full raw traces and compilation caches stay in the named volumes.
