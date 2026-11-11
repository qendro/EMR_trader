# Engineering task breakdown

Use these work packages as implementation units. Each should produce a reviewable change with relevant tests and benchmark evidence. Dependency labels identify technical ordering; they do not request concurrent coding agents. Do not create all proposed modules before the first vertical slice needs them.

Implementation and evidence are tracked in [implementation.md](implementation.md). Task definitions below remain the acceptance requirements; source existence alone does not close a work package.

The first qualified increment implemented Q0/Q1 and V0–V5. M1's required gates passed on its retained source; the [M1 report](../benchmarks/REPORT.md) and [JSON status](../benchmarks/status.json) contain historical per-profile results. H0/H1 and B0 have measured reports: [H1](../benchmarks/H1-REPORT.md), [B0](../benchmarks/B0-REPORT.md). H2's small-substrate learning and direct-memory comparison are measured on the gating card in the [H2 report](../benchmarks/H2-REPORT.md); [W8](../benchmarks/W8-REPORT.md) has passed its three-profile contract check. Geometric diagnosis, fixed-time comparison and performance gates remain open, so M2 is not yet accepted. Q2 records independent connection-cap failures on both cards and remains open for its resource/stage audit and declared capacity-promotion experiment. A0–P1 are not implemented.

## Qualification

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| Q0 | Record each GPU/VRAM and Windows/WSL/driver/host-memory configuration; define pinned Linux/amd64 Dockerfile and Compose CPU/GPU profiles; lock EMR/submodule/JAX/CUDA dependencies together with Gymnax and its transitive packages; name the gating card; define in-container GPU selection, persistent mounts and caches | None | Rebuildable image digest and manifests; no host Python dependency; container replacement retains results/checkpoints |
| Q1 | Verify passthrough, in-container card selection and selected GPU identity; probe compiled scan/conditionals, RNG, indexing, reductions, scatter/gather, sort/top-k and TensorNEAT ask/transform/tell; compare container CPU with each CUDA profile | Q0 | Primitive matrix and full generation on both cards; GPU runs reject fallback; failure reproducers where needed |
| Q2 | Freeze the Q2 manifest, then run bounded upstream direct/fixed/EMR baselines inside Docker on each card; measure cold compile, discovery, execution, evolution, VRAM and host RAM; recurrent cache off | Q1 | Per-card feasibility report and resource envelope before A0; pass or fail against the Q2 screening criterion on the frozen manifest, with a dated fallback decision on a fail; identical semantic workloads for cross-card comparison |

## Stateful vertical slice

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| V0 | Define state containers, candidate/episode IDs, the three runtime profiles with delay labels, both graph encodings with separate immediate/delayed dense matrices, shared scenario data, score convention and cross-generation comparability policy, capacity/overflow policy, and explicit baseline matrix-product precision | Q0 | Contract examples, invalid-config tests, no framework registry |
| V1 | Implement scalar float64 NumPy graph semantics for all three profiles, delayed-cue environment in oracle and memory-manifest modes, fixed memory oracle, reset control, reaction-latency fixture, mixed-delay two-hidden-layer fixture, and precise terminated/truncated behavior | V0 | Balanced temporal-memory tests; manually traceable tiny graphs using the same FP32-representable inputs as JAX |
| V2 | Implement one-capacity planner and dense masked JAX policy step; explicit state carry, DAG ordering and synchronous recurrence; set float32 matrix-product precision before tracing and audit operation overrides | V1, Q1 for GPU | Differential reference tests and compiled step benchmark with recorded arithmetic policy |
| V3 | Wrap TensorNEAT initialize/ask/tell and map genes into direct graphs; audit activations, aggregation, input/output transforms, historical IDs, caps, and species stagnation and spawn allocation against stored bests | V0, Q1 | Operator parity on same backend/keys, no stale-score association, valid direct decoding |
| V4 | Implement scan rollout, candidate/episode vectorization, stable-key derivation, chunk/tail handling, reducers, the shared-scenario-data argument with its fixture, and Gymnax CartPole adapter | V2, V1; Q0 environment lock | Scalar/batched parity, no per-step host dispatch, state-isolation tests, no recompilation when shared data values change |
| V5 | Connect decode/evaluate/tell; archive evaluated champion; atomic checkpoint; export/reload; minimal run and benchmark entry points | V3, V4 | Complete M1 demo, interruption/resume, five-seed report, ungated D diagnostics, stagnation record and G9 pilot for the M1 workloads |

## Indirect and adaptive development

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| B0 | Add a fixed-topology weight-only baseline: a substrate-shaped graph under the layered profile with the hidden size of the fixed substrate for each task, truncation selection with Gaussian weight mutation, same evaluator | V5, H1 | Baseline scores on CartPole and the memory-task manifest; reused unchanged at M2, M3 and M5 |
| H0 | Add CPPN query adapter and fixed-coordinate decoder; separate CPPN and phenotype capacities; specify weight transforms/bias conventions | V3 | Scalar/vectorized query and weight parity |
| H1 | Emit fixed substrates with separate immediate/delayed dense matrices and masks; implement the persistent layered profile in JAX; test lossless edge-list conversion and mixed-delay export/reload; export compact genotype and decoded phenotype with interpretation and precision metadata | H0, V2 | Same evaluator as direct backend; layered and mixed-delay trajectory parity; 256/1,024-node fixtures |
| H2 | Run fixed-HyperNEAT memory/control experiments, the ungated geometric diagnostic and W8; test query chunks; run the G9 pilot before acceptance; produce the system and encoding-isolating comparison reports | H1, V4, V5, B0 | M2 quality and performance report, failed experiments retained |
| A0 | Map EMR public APIs and supported features to decoder contract; define configuration translation, capability rejection and minimal patch if necessary | Q2 screening criterion passed or decided, H1 | Adapter contract test; every enabled option has observable effect or explicit rejection |
| A1 | Convert adaptive masks, coordinates, input/output weight blocks and recurrent edges into dense-block graphs; preserve activation and settling semantics for reference; show layered outputs equal upstream feed-forward outputs when recurrent edges are removed | A0 | Upstream graph/output parity, layered feed-forward parity, overflow/threshold-boundary tests |
| A2 | Evaluate adaptive graphs with persistent carry under the layered profile, evolve via the same TensorNEAT loop, benchmark discovery and actual decoded graph distributions, run its own G9 pilot, repeat the geometric diagnostic and both comparison reports | A1, H2 | Genuine adaptive evolution plus temporal/control acceptance |

## Scale and acceptance

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| S0 | Add lossless compaction, index maps, fixed bucket signatures, stable routing and explicit capacity promotion | A1 | Padding/permutation/capacity properties; bounded compilation count |
| S1 | Prototype and measure an edge-list/segmented or other alternate runtime and density-based dispatch; adopt it only if G8 passes; keep dense fallback | S0, V2 | Graph parity across layouts; crossover benchmark; recorded adopt-or-retain decision |
| S2 | Budget query temporaries, phenotype storage, hidden state and environment state; tune population/scenario microbatches; add long-run memory tests | A2, S1 | M4 scaling envelope, no silent host offload or swap-based performance claim |
| P0 | Add one harder environment; record random/fixed-topology baselines; pilot and freeze score normalization and evaluation budget | A2, V4 | Registered acceptance protocol and independent held-out scenarios |
| P1 | Assemble cross-backend regression suite, comparison-protocol reports, reproducibility/export docs, final benchmarks, supported-features table and platform acceptance report | S2, P0 | Every mandatory M5 criterion has an artifact or is explicitly open |

## Optional research packages

| ID | Work package | Depends on | Completion evidence |
|---|---|---|---|
| X0 | Independent device processes/islands, bounded migration, global lineage mapping, receiver reevaluation, checkpoint coordination | P1 and suitable hardware | Two-device correctness/throughput comparison, migration semantics documented |
| X1 | Two-actor environment, per-actor state, team/opponent sampling and credit-assignment experiment | P1 | A named agent-organization mechanism with baseline; not merely a larger network |
| X2 | Continuous-parameter refinement with discrete topology frozen; define Baldwinian/Lamarckian inheritance semantics | P1 | Ablation at fixed evaluation and fixed elapsed-time budgets; correct gradient checks on supported backend |

## Independent work and critical dependencies

- Once V0 is agreed, V1/reference environments and V3/evolution integration can be developed independently. V2 needs V1’s semantics; V4 needs V2’s execution contract.
- Q2 can run while V1–V4 develop in the already-qualified CPU environment. Major adaptive integration A0 waits for Q2.
- Benchmark fixture generation and report formatting can develop independently after Q0/V0; performance conclusions wait for real runtime paths.
- B0 needs the runner from V5 and the layered runtime from H1, and must exist before H2 reports.
- H0 needs the genotype adapter, not a new environment. H1 must finish before A0 because both require a verified phenotype boundary.
- S0 and harder-environment preparation can progress independently after A1/A2 respectively. Alternate runtime optimization S1 waits for representative graph fixtures.
- Changes to recurrent semantics, gene identity, precision, or fitness ordering cross component boundaries and require coordinated contract updates rather than independent edits.

## Implementation discipline

Each package records: scope, selected runtime profile, changed interface, tests, benchmark workload IDs, backend/versions, observed limits, and next dependency. Keep reference calculations independent of the JAX implementation. Add tests for behavioral contracts and likely failures, not tests that merely duplicate code structure.

Do not add an abstract hierarchy to anticipate every future algorithm. Add one contract and a concrete implementation; generalize when the second implementation arrives. Do not combine a dependency upgrade, runtime semantic change, and performance rewrite in the same task. Preserve a known-good benchmark fixture for each accepted stage.
