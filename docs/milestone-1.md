# First development milestone: stateful neuroevolution vertical slice

This is the first implementation milestone after the minimal M0 dependency/backend gate. It delivers a working standalone engine and proves temporal state semantics before complex substrate discovery. Q2’s bounded upstream EMR assessment can proceed alongside it and must finish before major EMR integration.

## 1. Exact scope

Implement only:

- One explicit TensorNEAT evolution adapter, including complete backend state and stable candidate IDs.
- A direct-genome-to-graph decoder with a restricted, fully mapped activation/aggregation vocabulary.
- One fixed capacity: 32 total nodes and 128 connections per genome initially, including inputs/outputs in the node count.
- A masked dense FP32 JAX graph runtime with matrix-product precision explicitly set to `float32`, and an independent scalar NumPy reference computed in float64 from the same FP32-representable inputs and weights.
- DAG evaluation and a persistent synchronous profile with K=1. Cycle-capable evolution uses an appropriate TensorNEAT recurrent-genome configuration, verified in V3; do not enable cycles in a feedforward-only operator path. The persistent layered profile is frozen here as a contract with reference fixtures only; its JAX implementation belongs to M2.
- A delayed-cue environment with two modes: the deterministic oracle fixture with its known recurrent solution, and the memory-task manifest with seeded distractors that diagnostic D uses.
- A dependency-qualified Gymnax CartPole adapter, explicit action mapping and episode accounting.
- A population/scenario evaluator with time scanning, candidate chunking and carried state.
- A shared-scenario-data argument in the evaluator contract, proven by one small fixture; the long-horizon workload belongs to M2.
- A small generation controller, result summaries, checkpoint/resume and phenotype export/reload.
- A reproducible demo and component/end-to-end benchmark records.
- A pinned Linux Docker image and Compose CPU/RTX run profiles, with results and checkpoints persisted outside the container’s writable layer. All demos, tests and benchmarks run inside containers.

Do not implement capacity buckets, custom GPU kernels, automatic graph partitioning, a plugin framework, distributed execution, multiple agents, gradient training, market interfaces, or adaptive substrate discovery in this milestone. The next two milestones explicitly add fixed HyperNEAT and adaptive EMR.

## 2. Semantic contract to freeze first

For persistent synchronous execution, input values are clamped to the current observation. Each non-input node updates from the previous internal-state values plus current clamped inputs, using masked weighted summation, bias and tanh. All nodes update simultaneously. Read output-node values after the update and retain the resulting state for the next observation. No implicit reset or additional settling occurs.

For DAG execution, use a validated topological schedule within the observation. It is a distinct mode. Its outputs must match compatible upstream direct-network fixtures including configured input/output transforms. Reject unmapped gene functions. Do not claim equivalence between DAG propagation and K=1 synchronous propagation.

For persistent layered execution, which substrate decoders use from M2, zero-delay edges are evaluated in topological layers within the observation and unit-delay edges read the state retained from the previous observation. With one hidden population: h_t = f(x_t·W_in + h_{t-1}·W_hh + b_h), y_t = g(h_t·W_out + b_y), and h_t is carried. With no unit-delay edges its outputs equal DAG outputs. Freeze this now, with hand-traced reference fixtures, so that M2 does not reopen the contract.

Represent immediate and delayed connectivity as separate matrices and masks. Role-shaped substrates retain W_in and W_out, with separate W_hh_zero and W_hh_delay when both hidden connection types exist. The single-hidden-population profile needs only the delayed hidden block. Freeze a two-hidden-layer reference fixture with immediate forward edges and delayed feedback in M1; implement its JAX and export/reload checks in H1. No generic block-management framework is required.

The synchronous profile has a stated latency. A path through n hidden nodes reaches the output n observations after the input, and an add-node mutation inserts one observation of delay. A fixture measures this, and results from the profile report it.

An isolated non-input node evaluates its bias through its activation. Masked nodes are inert. Duplicate edges are rejected in this milestone, so dense matrix assembly cannot accidentally overwrite one. Invalid indices, nonfinite active weights or values, and capacity overflow are invalid-candidate results. Internal TensorNEAT padding is converted before runtime use. Input and output node sets must satisfy the declared interface.

Policy state is private to each candidate and episode. A reset clears it before the new first observation. A terminal transition contributes its reward once; subsequent padded steps contribute zero. A horizon cutoff records truncation. Independently sampled windows reset; continuation of a single trajectory across compute chunks preserves state.

## 3. Demonstrations

### A. Temporal-memory oracle

At time zero, observe a cue c in {-1,+1}. Subsequent observations contain a zero cue and a query flag that becomes one after a delay. The policy chooses a binary action at query time; reward is one for matching c. Generate equal numbers of both cues for every tested delay. No observation contains a future target, previous action, or hidden cue copy.

A one-state recurrent oracle uses a self-weight of 2, cue-input weight of 1 and tanh. The sign persists after the cue disappears. Decode the sign at query time. A reset-every-step version sees zero at query, picks the same deterministic tie action, and achieves exactly 50% on balanced pairs. Use delays 1, 2, 4, 8 and 16 for the deterministic fixture. The fixture has no distractors and stays separate from the memory-task manifest.

Acceptance: 100% oracle accuracy, 50% reset-control accuracy, NumPy/JAX agreement, and invariant results under candidate/episode permutation and chunking. This establishes runtime memory only. Evolving memory is a separate M2/M3 acceptance condition. Demonstration D attempts it early, without gating.

### B. Closed-loop evolution

Use Gymnax CartPole-v1 with a fixed, recorded version and environment parameters. Start from small direct feedforward genomes under the DAG profile and allow structural changes within the caps. Use a two-output action head with argmax (ties choose the first action), or freeze one equally explicit mapping before the run. Ensure the selected TensorNEAT node/output transforms are implemented exactly by the decoder/runtime.

Initial experiment manifest:

| Parameter | Value |
|---|---|
| Logical population P | 128 |
| Initial physical batch p | 32, adjustable after memory/throughput pilot without changing semantic random keys |
| Episodes per candidate B | 8 |
| Maximum episode length | 500 transitions |
| Generations | 100 |
| Evolution seeds | 0, 1, 2, 3, 4 |
| Precision | FP32 arrays; `jax_default_matmul_precision="float32"` before tracing; no lower-precision operation overrides |
| Graph limits | 32 total nodes, 128 edges |
| Fitness | Mean undiscounted episodic reward, maximized |
| Training scenarios | Generation-specific, shared across candidates |
| Held-out evaluation | 20 independent episodes per final policy; seed namespace disjoint from training |
| Target | Mean held-out return >=450 in at least three of five runs |
| Execution profile | DAG, feedforward genome operators |
| Validation scenarios | 32 episodes per candidate; seed namespace disjoint from training and held-out |
| Champion selection | Top 8 final-generation candidates by training fitness, re-scored on the validation scenarios; highest validation mean wins (proposed) |
| Cross-generation comparability | Generation-specific scenarios; species removed by stagnation recorded each generation |

Choose the candidate for held-out evaluation by the champion-selection rule in the manifest, not by the final held-out scores. Preserve failures and all five seeds. Do not repeatedly tune against this acceptance seed set; establish a fresh acceptance set if the protocol is revised.

Acceptance additionally requires at least one structural mutation to be observed and correctly decoded, correct best-candidate archiving, and no stale scores after reproduction. The library’s initial gene/species capacities and operator probabilities must be recorded; do not assume defaults appropriate to P=128 without inspecting them.

### C. Resume and export

Run to generation 40, checkpoint, restart the process and continue to 100. Compare with uninterrupted execution under the determinism policy. Export the evaluated champion’s phenotype and its interpretation metadata; reload without importing TensorNEAT or EMR in the inference path. Replay a fixed episode and compare actions, states and reward.

The artifact carries input/output and node ordering, activation vocabulary/version, graph encoding/profile version, update mode, K, weights/masks, state initialization, action mapping, dtype and matrix-product precision policy. Preserve edge delay labels or the separate immediate/delayed matrix-and-mask sets, including role mappings for dense substrates. Never reconstruct delays from coordinates or merge the two sets on export. Store genotype and decoder/source provenance beside it for later research reproduction. Do not require a serialized device executable for portability.

### D. Reported diagnostics, not gated

Run these on diagnostic seeds 100–104, disjoint from every acceptance set. They produce reports and decisions, not pass/fail results.

1. **Evolved memory with direct genomes.** Recurrent genomes under the persistent synchronous profile, K=1, on the memory-task manifest in [benchmarks.md](benchmarks.md), frozen before this run. Report interpolation and generalization accuracy per delay, solve rate and generations to threshold. If no seed reaches 90% interpolation accuracy, run known-solution initialization and weight-only controls before M2 relies on evolved memory.
2. **CartPole under the synchronous profile.** The manifest of B with recurrent genomes. Report the score difference against B and the measured reaction latency.
3. **Fixed-scenario control.** The manifest of B with one fixed training scenario set. Compare species removed by stagnation against B to size the comparability effect described in the architecture.

## 4. Build order and task handoffs

1. **Q0/Q1:** establish the image and CPU reference profile; verify Windows/WSL GPU passthrough and qualify CUDA on each target card. No GPU training claim precedes backend verification.
2. **V0:** commit the data containers and semantics above, including delay labels, both graph encodings, shared scenario data and the comparability policy; specify what is static, what is dynamic, and how invalid candidates are scored.
3. **V1:** build the float64 reference graph evaluator for all three profiles, the cue task in both modes, oracle, latency fixture and hand-traced fixtures. This creates the expected outputs for V2 and, for the layered profile, for H1.
4. **V2:** implement JAX state updates and the single-capacity planner; pass step and trajectory differential tests before evolution is connected.
5. **V3:** independently wrap TensorNEAT and decode supported direct graphs. Test operator behavior with fixed keys and gene identity. V2 and V3 meet at the phenotype contract.
6. **V4:** implement scan rollout and batch/chunk scheduling; add CartPole. Confirm environment auto-reset handling and terminal accounting before using its fitness.
7. **V5:** connect the generation stages, archive-before-tell, checkpoint/export, run summaries and benchmark driver. Execute demonstrations A–C, then the D diagnostics.

CPU reference development may continue while a driver/container issue is diagnosed, without per-timestep CPU/GPU bridges. GPU runs must stop on missing or wrong devices. Record M1 qualification separately for the RTX 3060 and RTX 3070; a pass on one card does not qualify the other. The gating card named at Q0 decides when M2 may start.

## 5. Required validation cases

| Area | Required cases |
|---|---|
| Graph evaluation | Tiny DAG, self-loop, two-node cycle, disconnected nodes, padded nodes, invalid indices, duplicate edge rejection, node/output mapping, layered-profile reference fixtures, synchronous reaction-latency fixture |
| Temporal state | No reset across observations; reset at episode boundary; independent candidates/episodes; continuation across time chunks |
| Numerical behavior | Independent scalar agreement, finite active values, deterministic tie action, invalid-candidate containment |
| Batching | p=1 versus 32, non-divisible tails such as P=35, episode permutations, stable scenario keys, shared scenario data as a dynamic argument with no recompilation when its values change |
| Evolution | Parameter and structural changes, cap enforcement, elite preservation, evaluation ID ordering, all-invalid handling, stagnation removals recorded per generation |
| Environment | Terminal reward once, post-terminal zero contribution, horizon truncation, no auto-reset state contamination, memory-manifest balance per cue-delay pair, distractor draws fixed by episode key |
| Reproducibility | Seed namespaces, 40/100 split resume, version/schema rejection, phenotype-only inference reload |
| Performance | Cold/warm timings, no new compilation for in-capacity topology changes, no per-step host transfer, bounded memory |

Use the tolerances in [benchmarks.md](benchmarks.md). Do not require identical long-run GPU evolutionary histories if the backend’s reductions are nondeterministic; establish exact CPU replay and documented GPU numerical/statistical evidence instead.

## 6. Exact demonstration outputs

The future implementation should offer one documented run entry point, one benchmark entry point and one replay entry point. These are interface requirements, not implementation code in this delivery.

The run directory must contain:

- Resolved experiment configuration and source/dependency/hardware manifest.
- Per-generation best/mean score, valid count, species count, active node/edge counts and elapsed time.
- Independent compile, decode/plan, rollout and evolve timing records, plus actual total wall time.
- GPU model/VRAM and memory measurements, image digest, host driver/WSL versions, and separate qualification results for each card.
- The memory oracle/reset-control results and five-seed CartPole results.
- The D diagnostic reports and the per-generation stagnation record, labeled as ungated.
- Checkpoint, evaluated-champion genome/phenotype, and replay comparison.
- A concise status: functional pass/fail, learning pass/fail, GPU correctness pass/fail/not-run per card, and the measured GPU-to-CPU throughput ratio on W3, which M1 reports without gating.

Record only bounded summaries in normal runs; full trajectories are opt-in for a handful of fixtures/finalists. Avoid an unbounded log of every activation or every population snapshot.

## 7. Completion checklist

M1 is functionally complete when A/B/C and all relevant correctness tests pass, including the declared learning target, from a clean pinned Docker image, and the D diagnostics have been run and reported. M1 closes, and M2 may start, when it is functionally complete and G1–G4 pass on the gating card. The second card runs G1–G3; its remaining gates stay listed as open and close no later than M5. G5 is measured on W3 and reported, because 32-node graphs are not expected to show useful acceleration. Replace the container and demonstrate checkpoint continuation from its persistent mount. Host prerequisites are documented separately from container dependencies.

The milestone ends with a measured feasible p/B setting, frozen baseline metrics, an explicit list of supported semantics and functions, and a clear handoff to H0 fixed-substrate decoding. No arbitrary throughput number is claimed before Q2/V5 measurements. No package installation or implementation work has been performed as part of this planning document.
