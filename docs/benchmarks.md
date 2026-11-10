# Testing, GPU execution, and benchmark specification

All numeric thresholds below are proposed engineering gates. They are not measured capabilities of the RTX 3060 or RTX 3070. Commit the workload, seed list, tolerances, and budget before acceptance runs. A pilot can calibrate a gate, but a change requires a dated decision before rerunning acceptance; do not change it silently after inspecting held-out outcomes.

## 1. Hardware qualification and local resource budget

Targets: NVIDIA RTX 3060 and RTX 3070 on Windows, running the same Linux/amd64 Docker image through Docker Desktop’s WSL 2 backend. Hardware variants, VRAM, CPU, host RAM, co-location and Windows versions have not been inspected. Record them during Q0; do not infer memory capacity from the GPU name. Benchmark each card independently before any combined-device experiment.

Start with a device allocation budget equal to the smaller of 60% of physical VRAM or currently free VRAM minus 1 GiB. If this leaves no usable budget, stop and release competing workloads. After measurement, a profile may increase to at most 75% of physical VRAM while retaining that free-memory reserve. These are project policies, not hard quotas enforced by Docker. Record allocator reservation separately from peak live/workspace memory. Set and record JAX’s allocator policy before initialization; do not let its default reservation consume the measurement headroom. Measure host RAM, WSL limits and swap independently. Freeze a host-memory budget after inventory, leaving headroom for Windows and Docker. Record background display/GPU use, thermals and sustained throughput.

The CPU reference and both CUDA profiles execute inside Docker. GPU startup must verify the requested card and fail on CPU fallback. On a shared host both cards are assumed visible in the container; the run selects one before JAX initializes and verifies it, as the execution document specifies. Container GPU support requires Docker Desktop’s WSL 2 backend and the Windows NVIDIA driver; setup and dependency ownership are specified in [Windows and Docker execution](execution-environment.md).

Q1 must run in separate container processes for each profile and capture actual device placement. Probe FP32/int32/bool arrays; matmul; vmap; scan/fori_loop/while_loop; conditional branches; indexed gather and scatter-add; masked reductions; top-k/sort; random streams; and a complete TensorNEAT generation. Run with JIT enabled. A workaround that disables JIT cannot satisfy the GPU performance gate. An unsupported operation must fail explicitly rather than route individual timesteps through Python.

Start from EMR’s JAX 0.6.1 reference stack with compatible CUDA 12 wheels and plugin/userspace dependencies locked inside the image. Run CPU comparison using the same image and explicit CPU backend. Record image digest, package/source versions, driver, WSL and Docker versions. Diagnose host passthrough separately from package compatibility. Do not install project Python packages or a second CUDA library stack on the Windows host. A clean image rebuild and a successful restart from persisted checkpoints are part of qualification.

## 2. Compilation and execution boundaries

Initially expose separately callable/benchmarkable compiled functions for:

1. Genome transformation and reproduction/speciation.
2. CPPN query and phenotype construction.
3. Graph validation/packing, where device-compatible.
4. A standalone policy step.
5. Rollout over a fixed horizon, including policy and environment steps.
6. Fitness reduction, if not already a natural part of rollout.

The rollout composes the same policy-step function inside a compiled scan; XLA may inline/fuse it. Independent testability does not require a Python call or hard kernel boundary per timestep. Do not assemble decode, evaluate, and tell into a monolithic generation JIT initially.

Vectorize candidate and episode axes; scan through time. Logical population P and physical batch p are distinct. Precompile only the signatures actually used; log each new signature. Use fixed-size tails, bounded activation vocabularies, explicit state, and arrays as dynamic arguments. Keep K, capacities and execution semantics static. Different topology values should reuse the executable.

Population microbatches may use a small host scheduling loop. The hot path must have no per-node, per-edge, per-agent, or per-timestep Python loop. Batching all operations with vmap does not guarantee good memory use; sweep p/B and inspect temporaries before scaling.

### Memory and work model

Use an explicit estimate before launching each larger cell. With physical population p, episode batch B, neuron capacity N and edge capacity E, one FP32 dense adjacency requires about 4pN² bytes; a layered graph storing full immediate and delayed matrices requires about 8pN² bytes before masks. Role-shaped substrates instead account for their actual blocks: 4p(IH+HO+H²) bytes for one recurrent hidden population, or 4p(IH+HO+2H²) when both immediate and delayed hidden blocks are stored. A canonical edge list with FP32 weights and two int32 endpoints requires about 12pE bytes before masks and delay labels. Compare actual allocated layouts rather than applying a universal density crossover. Hidden state alone requires about 4pBN bytes, usually with multiple live buffers. Add environment state, genome state, decoding queries, compiler workspaces and allocator slack. These terms are estimates, not peak-memory measurements.

Sequential rollout work scales approximately with pBTKE for an edge-based implementation, or pBTKN² for dense recurrence. Decoding has a separate candidate-grid/query cost, and recurrent pair discovery can be quadratic in active hidden nodes before pruning. Sparse final storage does not prove sparse discovery cost. The scheduler must budget both stages; a small output phenotype can still cause a decoding OOM.

Keep all p candidates' genomes/phenotypes on the same device during a microbatch; avoid recopying weights each observation. Data shared across candidates should remain shared. Reduce episode metrics online and keep only current recurrent state unless an explicit trace or gradient experiment needs history. Later buffer donation or rematerialization must be profiled and tested for ownership correctness; neither is required to build M1.

## 3. Workload matrix

Do not execute the full Cartesian product. Begin with smoke cells, then vary one axis at a time around the measured feasible point. Every cell records P, p, B, T, K, active/padded N and E, dtype, graph family, decoder, and device.

| Workload ID | Purpose | Initial cells and later expansion |
|---|---|---|
| W0-primitives | Backend compatibility | Tiny known arrays plus moderate batched shapes; container CPU versus each CUDA card |
| W1-upstream | Pre-integration feasibility | TensorNEAT P=32/128; EMR max_depth=1/2/3/4, P=32 then 128; feedforward and recurrent paths separately; sample count 4 then 64; cache off. The screening cell is the Q2 manifest below; max_depth=5 is reported |
| W2-graph | Policy throughput and correctness | N=16/64/256/1,024/2,048; E near 4N/16N plus dense small graphs; p=1/8/32; B=1/8; T=32/128; K=1/4; substrate-shaped dense-block graphs with input width 4 and 400; DAG, layered and synchronous profiles |
| W3-generation | End-to-end engine | M1 P=128, p initially 32, B=8, T=500 CartPole; 100 generations; N<=32/E<=128 |
| W4-memory | Persistent state and evolved memory | The memory-task manifest below; first run as the ungated M1 direct-genome diagnostic, then as the M2 and M3 acceptance task |
| W5-adaptive | EMR discovery versus rollout | Sweep depth and population independently on the integrated evaluator; record candidate grid, active nodes, edges, query slab size, and recurrent pair count. The sweep is diagnostic; the M3 gate cell is named in section 5 |
| W6-scale | Irregular topology target | P=256, N=2,048, E<=32,768, B=4, T=128, K=1; bounded p selected by pilot. The edge cap describes the sparse irregular family only; add a dense-block substrate fixture at the same N with a 400-wide input block |
| W7-stability | Memory, compilation and thermal drift | Fixed shapes and changing topology for at least 100 generations and 15 minutes total; extend beyond 100 if needed |
| W8-sequence | Shared-data long horizon | One shared read-only multichannel sequence indexed by all candidates; T>=10,000 in time chunks with carried state; small private state per episode; the reducer and its acceptance thresholds are specified below |

For W1, use the smallest complete upstream configuration first. Stop escalating a cell at the local memory ceiling or five minutes of compile time; label it over-budget. Expand beyond depth 4 only if earlier measured costs fit; an over-budget depth-4 cell fails the Q2 screening criterion in the roadmap and triggers its dated decision. Do not lower an edge cap and silently change the graph to make a cell pass. Any upstream metric extraction that synchronizes is included in end-to-end timings.

Graph fixtures include chains, random directed graphs, self-loops, hub nodes, disconnected components, modular graphs with irregular cross-links, and near-dense graphs. Benchmark both synthetic graphs and frozen real EMR discoveries. For candidate nodes near variance thresholds, report topology stability separately from ordinary floating-point error.

### Q2 manifest: the W1 screening cell

Q2 times one fully specified upstream workload. Freeze this manifest under `configs/` before any Q2 timing is taken, and record every field with the result. A field marked upstream default takes its value from the pinned EMR source and is written out explicitly in the frozen file.

| Field | Value |
|---|---|
| Source | Pinned EMR SHA and TensorNEAT fork SHA; upstream generation loop, unmodified |
| Dataset | Seeded synthetic set of 64 samples with 4 inputs and 2 outputs, generated once and stored with the manifest |
| Substrate | Input and output coordinates written out; max_depth=4; initial depth upstream default |
| CPPN genome | Node and connection capacities, initialization and mutation settings: upstream defaults |
| Discovery | Canonical variance rule; variance threshold, weight threshold and connection cap: upstream defaults; cache off |
| Activation and aggregation | Upstream base hidden and output activations, sum aggregation; no per-node palette, no neuromodulation |
| Recurrence | Two separate runs: feed-forward, and recurrent with upstream default settling steps and per-genome discovery |
| Execution | Standard mode, single device, FP32 under the arithmetic policy in section 6 |
| Population and seeds | P=128; three seeds |
| Timing | Cold compile time recorded; median warm generation time over at least 10 generations after compilation |
| Recorded outcomes | Active-node and edge counts per genome (minimum, median, maximum), candidate-grid size, peak VRAM and host RAM |

Active-node distribution is an outcome of the seeds and initialization, not a setting, so it is recorded and not prescribed. The input and output widths match CartPole, the first interface M3 uses. The result is upstream feasibility evidence; the roadmap states the screening ceiling and what it does not establish.

### Memory-task manifest (W4)

One manifest defines the evolved-memory task for the M1 diagnostic and for the M2 and M3 acceptance runs. Its values are proposed and are frozen before the M1 diagnostic runs. The deterministic oracle in milestone 1 remains a separate correctness fixture with no distractors.

| Field | Value |
|---|---|
| Observation | Two channels: cue and query flag |
| Episode | Step 0 shows cue c in {-1,+1} with query 0. Steps 1 to d-1 show a distractor on the cue channel with query 0. Step d shows a distractor with query 1 and ends the episode. d is the delay; episodes are padded to 17 steps with masks |
| Distractors | Independent uniform values in [-0.25,+0.25] on the cue channel at every step after step 0, drawn from the episode key |
| Action and reward | Read once, at step d, after the policy update of that step: +1 if the output is >=0, else -1. Reward 1 if it equals c, else 0. Outputs at other steps are ignored |
| Training delays | 1–8 |
| Training episodes per candidate | 32 per generation: each of the 16 cue-delay pairs twice. Distractor draws are generation-specific and shared across candidates |
| Fitness | Fraction of training episodes correct |
| Validation and champion | Top 8 final-generation candidates by training fitness, re-scored on 64 validation episodes (each pair four times); highest validation accuracy wins |
| Interpolation set | 256 held-out episodes at delays 1–8: each pair 16 times, with unseen distractor draws |
| Generalization set | 256 held-out episodes at delays 9–16 with the same balance; reported separately, never gated |
| Search budget | P=128 and at most 250 generations, subject to the G9 pilot |
| Seeds | Diagnostic 100–104 in M1; acceptance 0–4 in M2 and M3; training, validation and held-out namespaces disjoint |
| Acceptance | Interpolation accuracy >=90% in at least three of five seeds; applies to M2 and M3 only |
| Execution profile | Persistent synchronous for direct recurrent genomes; persistent layered for substrates |

Sampled distractors make every episode a distinct trajectory, so held-out episodes are not repeats of the 16 noise-free cue-delay trajectories. A known solution lies inside the search space: the oracle weights (self-weight 2, cue weight 1, tanh) keep the sign of the cue for any delay at this distractor amplitude, because the worst-case update tanh(2h − 0.25) rises from 0.76 toward a fixed point near 0.92. A memoryless policy scores 50% in expectation. Report interpolation and generalization accuracy per delay.

### W8 shared-sequence task and reducer

W8 is deliberately free of any application concept. The task is tracking: the observation is a causal slice of the shared sequence, the action is a bounded scalar, and the per-step reward depends on the action and the next value of one channel, less a penalty on action change. The task definition bounds every reward to [-1,+1].

**Reducer.** For an episode with n valid steps and rewards r_1..r_n, let m be the mean and s the population standard deviation, s = sqrt((1/n)·Σ(r_i − m)²). The score is m / (s + ε), with ε = 0.01 in reward units. The additive ε defines the zero-variance case and bounds the score to [-100,+100]: constant zero reward scores 0, and a constant reward c scores c/ε. An episode with fewer than 32 valid steps is an invalid-candidate result. ε and the minimum count are proposed values recorded in the manifest.

**Accumulation.** Keep n, the running mean and the running sum of squared deviations M, updated each step inside the scan: δ = r − mean; mean += δ/n; M += δ·(r − mean). Merge two chunks a and b with n = n_a + n_b, δ = mean_b − mean_a, mean = mean_a + δ·n_b/n, and M = M_a + M_b + δ²·n_a·n_b/n. Never accumulate Σr and Σr² and subtract.

**Reducer tests.** Feed fixed, recorded FP32 reward sequences to the FP32 reducer and to a float64 reference, so that accumulation error is isolated from trajectory divergence. Cover constant zero rewards, constant nonzero rewards, nearly constant rewards with a relative spread near FP32 resolution, a full-range sequence of 10,000 steps, and at least four chunk partitions of that sequence, including a single chunk and uneven chunks. Proposed thresholds against the float64 reference, calibrated once and frozen like any other gate: mean within 1e-5 absolute; s within 1e-5 absolute plus 1e-3 relative; score within 1e-3 absolute plus 1e-3 relative. Every chunk partition must agree with the single-chunk result within the same thresholds. The 128-step trajectory tolerances in section 6 do not apply to these statistics.

W8 checks four contract properties:

- Shared data is a dynamic argument: new values do not recompile, and executable size does not grow with sequence length.
- Chunked continuation over a long horizon preserves policy and episode state.
- The reducer passes the tests above at T>=10,000.
- Observation is causal: perturbing later values of the sequence leaves earlier actions unchanged.

## 4. Measurements and tools

| Metric | Definition / method |
|---|---|
| Cold startup and compilation | Fresh process and cold compilation cache; report imports/loading separately from trace/lower/compile where APIs permit |
| Warm-cache startup | Fresh process with populated persistent cache, separated from steady execution |
| Steady stage latency | Warmed function, explicit completion before stopping timer; median/p95 over 30 repetitions for short cells, at least 10 for expensive cells |
| Generation latency | Actual controller wall time including decoding, dispatch, synchronization, logging at declared frequency, and reproduction |
| Throughput | Valid agent-environment transitions/s, evaluated episodes/s, evaluated candidates/s and generations/s; also padded transitions/s |
| Discovery | Candidate positions/s, CPPN queries/s, active-node fraction, generated edges, query temporary peak |
| Memory | Peak live arrays/workspace where supported, allocator reservation separately, process RSS/footprint, system memory pressure and swap deltas |
| Compilation stability | Number and identity of signatures/executable builds across changing genomes and bucket occupancies |
| Host/device interaction | Host/device copies, synchronization, launch gaps, host callbacks and WSL/container I/O; separate PCIe transfer costs from kernel execution |
| Search quality | Held-out score, solve rate, uncertainty across independent evolution seeds, evaluations and wall time to threshold |
| Reproduction cost | Mutation/crossover/speciation/selection timing and population/species scaling |

Use JAX completion barriers for timing, not in every production step. JAX dispatch is asynchronous and first-use compilation is distinct from steady execution. [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)

Use JAX profiler traces/XProf where supported by the pinned stack, with traces written to persistent artifacts. JAX recommends XProf for memory analysis. Qualify Nsight Systems for execution/transfer timelines and Nsight Compute for selected kernels in a separate profiling image/profile if needed. WSL/driver/tool versions can restrict counters or tracing; report unavailable metrics rather than treating them as zero. NVIDIA lists GPU utilization and active-compute-process queries as unsupported in nvidia-smi under WSL, so no gate depends on them; qualify the working queries in Q0. [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) Record container host-memory usage and GPU memory through supported tools. [JAX profiling](https://docs.jax.dev/en/latest/profiling.html), [JAX memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)

Use three fresh-process repetitions for main performance comparisons and a sustained run. Record timing noise and thermal/load changes. Stage-isolated synchronization changes scheduling, so do not interpret the sum of independently synchronized medians as exact production latency. Compare controller overhead to a matched minimal driver with the same compiled stages, data, and synchronization policy.

## 5. Performance gates and redesign triggers

| Gate | Required evidence / initial threshold | Failure response |
|---|---|---|
| G1 backend correctness | All required primitives and a full generation run on claimed device, JIT enabled, differential suite passes | Mark lane unqualified; retain CPU development |
| G2 compilation | Zero unexpected compilations during 100 post-warm-up generations for fixed supported signatures | Inspect static arguments, varying shapes and function recreation |
| G3 memory | Each card stays within its frozen VRAM budget (60% initially, <=75% after pilot and free reserve retained), and its recorded host/WSL RAM budget; no sustained swap growth | Reduce p/query slabs; inspect reservation/retention; no pooled VRAM assumption or offloaded-throughput claim |
| G4 control overhead | Controller median latency <=1.20x matched minimal-driver median on W3, same diagnostics policy | Remove synchronization, copying and Python conversions |
| G5 GPU benefit | On the workload named for the milestone below, GPU median end-to-end throughput >=1.20x CPU, improvement larger than observed noise across three runs | Distinguish backend feasibility from useful acceleration; profile/revise or leave GPU-benefit gate open |
| G6 regression | No unexplained >10% median latency regression on accepted workload, confirmed in three runs | Diagnose or document a deliberate quality/cost tradeoff |
| G7 stability | W7 passes; after warm-up, live/working-set memory grows <10% for fixed shape/signature set and returns to a bounded plateau | Check references, cache growth, asynchronous queue buildup and leaks |
| G8 optimization adoption | A new layout/kernel improves representative stage latency >=15% or peak working set >=25%, while total generation latency regresses <=5%, and parity passes | Retain reference path; do not add complexity without useful benefit |
| G9 experiment feasibility | Before each acceptance experiment, a pilot on that integrated workload measures fixed overhead and warm seconds/generation, and the declared generation cap fits the research budget defined below | Reduce declared workload or explicitly request another compute lane later; never claim an unrun budget passed |

G5 is a useful-acceleration policy, not a promise that CUDA beats CPU on every small task. G1 may pass while G5 fails. The G5 workload is fixed per milestone so that the gate is neither unsatisfiable nor satisfied by a convenient cell. M1 measures W3 and reports the ratio without gating, because 32-node graphs are not expected to benefit. M2 gates on the W3 CartPole dimensions (P=128, B=8, T=500) with the fixed-substrate decoder and a 1,024-node recurrent phenotype under the layered profile. M3 gates on the same CartPole dimensions with the EMR decoder at max_depth=4 and P=128, or at the envelope recorded by the Q2 decision if that cell failed. M4 gates on the evolved-EMR generation at its measured envelope, with the W6 rollout ratio reported beside it. Each gate cell is one end-to-end generation: decode, plan, rollout, reduce and tell. A stage-only cell such as W2 never satisfies G5. GPU acceptance requires useful acceleration evidence for each target card against CPU execution on the same host in the same image. Also compare both GPUs on a shared workload; report individually tuned maximum throughput separately. If cards are in different hosts, record host differences and do not attribute the whole timing difference to the GPU. No minimum utilization percentage is required: high utilization can reflect wasted padded arithmetic.

The gating card named at Q0 runs every gate that applies to a milestone. The second card runs G1–G3 at each milestone and the full set at M5; its open gates are listed in each report. Numerical thresholds follow the same order: calibrated on the CPU profile and the gating card, validated on the second card. G8 governs adoption only: the definition of done does not require an accelerated alternate runtime if none passes G8.

**Research budget (G9).** The research budget is two hours of wall-clock time per evolution seed on one card, from process start to the final report. It includes startup, cold compilation, every generation, checkpoints at the declared frequency, champion selection and held-out evaluation. A five-seed experiment therefore has ten hours, run one seed at a time. The local demo budget is 15 minutes under the same accounting. Both budgets and the reserve below are proposed values.

The pilot runs the integrated workload itself, never an upstream proxy. It measures fixed overhead and median warm seconds/generation, then sets allowable generations = 0.9 × (budget − fixed overhead) / warm seconds per generation. A generation cap in a manifest must not exceed that number. If it does, resolve the workload/compute decision explicitly before the acceptance experiment, not midway through it. Q2 screens upstream discovery cost only and never substitutes for this pilot. Freeze absolute seconds/generation targets from these pilots and retain baseline values in the report; before a pilot, an absolute throughput prediction is unsupported.

Additional redesign triggers: discovery >50% of total generation time prompts query/layout profiling; padding overhead >2x active work prompts bucketing experiments; compilation >20% of total experiment wall time prompts signature reduction. These trigger experiments, not automatic changes to the algorithm. Avoid general-purpose JAX experimental sparse as the assumed fast path: its documentation describes it as a reference implementation unsuitable for performance-critical use. [JAX sparse documentation](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)

## 6. Correctness and reproducibility

**Reference tests:** manually trace tiny DAGs, a self-loop memory node, a two-node cycle, disconnected output, masked nodes, and biased input/output mappings. The NumPy reference uses straightforward loops and independent indexing, and computes in float64 from the same FP32-representable inputs and weights as JAX. It covers all three profiles, including a layered fixture whose output must change within the same observation as its input and a synchronous fixture that measures latency through one and two hidden nodes. Add a two-hidden-layer fixture with immediate forward edges and delayed feedback: check that scheduling uses only immediate edges, delayed reads use the previous state, and conversion plus export/reload preserves the trajectory. Test duplicate-edge handling explicitly: either reject duplicates at the contract boundary or deterministically combine them under a documented rule; do not let dense/sparse paths disagree.

**Arithmetic policy:** FP32 storage alone is insufficient. Set `jax_default_matmul_precision="float32"` before tracing correctness and baseline functions, including upstream comparison paths. Check that explicit operation-level settings do not override it with lower precision. Record the setting and any overrides with each compiled workload, checkpoint and exported policy. Use the same policy for CPU/GPU performance comparisons. JAX exposes matrix-product precision separately from input dtype. [JAX precision documentation](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)

**Differential tolerances:** FP32 one-step values start at atol=1e-6, rtol=1e-5 for active fan-in up to 32; bounded 128-step trajectories at atol=1e-4, rtol=1e-4. At larger fan-in, use c·u·S as an empirical pre-activation acceptance threshold, where u=2^-24, S is the float64 sum of absolute terms entering the node, and c is calibrated once on W2 fixtures in the CPU profile and on the gating card under the fixed arithmetic policy, then frozen before acceptance. The second card takes no part in calibration; it is validated against the frozen value when it runs G1. If it exceeds the threshold, diagnose the backend first. Revise c only by a dated decision that names the earlier evidence to rerun: every G1 differential result that used c, on every profile. This is not a proven worst-case error bound; multiplication, accumulation order and recurrent error propagation also matter. Compare one-step arithmetic using identical incoming state values to isolate kernel error, and assess full trajectories separately. Report the observed error-to-threshold ratio. These gates apply to stable fixtures, not arbitrary chaotic recurrent dynamics. Never calibrate c to absorb a precision-policy mismatch or widen tolerances merely to conceal a backend error. NaN/infinity is invalid unless an upstream gene-padding field explicitly uses it inside the adapter. Any later reduced-precision experiment gets its own label, numerical checks and performance comparison; it cannot replace baseline acceptance.

**Structural equality:** node/edge masks, input/output maps, padding validity, candidate ordering and integer gene identities must match exactly on fixtures away from threshold boundaries. Probe values immediately around thresholds to quantify rounding-induced topology differences. Run reduced-scale parity with upstream EMR before claiming larger comparisons.

**Batch properties:** scalar versus batch, candidate permutation/inverse permutation, full versus chunked population, episode permutation, padded tails, and hidden-state isolation. Partition random keys by stable semantic IDs, not current batch positions. Episode continuation across time chunks preserves state and key/time counters; a new independent episode resets them. Shared scenario randomness across candidates is intentional and distinct from accidental key reuse. Shared scenario data is passed once and indexed, never copied per candidate, and changing its values must not trigger compilation.

**Environment behavior:** verify reset timing, first observation, terminal reward inclusion, truncation, absorbing masks and auto-reset wrapper behavior. On a terminal transition, preserve its reward, mark subsequent steps inactive, and do not feed an auto-reset observation into the previous episode’s state.

**Evolution behavior:** same backend/keys reproduce upstream operator results; selection maximizes the declared score; mutation preserves caps/IDs; fitness is attached to the evaluated candidate; elites do not accidentally mutate; an all-invalid population aborts meaningfully. More than one accepted topology must appear during structural-evolution demonstrations. Record species removed by stagnation each generation and state the cross-generation comparability policy of the manifest.

**Reproducibility levels:** exact replay is required on the deterministic CPU reference path with pinned environment and ordered reductions. GPU replay requires identical RNG streams and structural decisions where deterministic operations permit them, plus documented numerical tolerance; do not promise bitwise results from nondeterministic reductions. Acceptance across devices relies on numerical and statistical agreement, not identical long evolutionary histories.

**Checkpoint tests:** split a run at generation 40, resume to 100, and compare to uninterrupted execution under the relevant determinism policy. Save all backend species/history state, RNG streams/counters, configuration, dataset/scenario generator version, source/dependency hashes, current population and evaluated archive. Checkpoints are not JIT-cache files. Reject incompatible schemas rather than guessing.

## 7. Environment progression

| Stage | Environment | Validates | Does not establish |
|---|---|---|---|
| Wiring | XOR and small regression/parity fixture | CPPN queries, weight transforms, scoring, upstream reproduction | Temporal memory or meaningful control learning |
| M1 oracle | Delayed binary cue, balanced +/- cues, later query | Carry/reset semantics, time ordering, no observation leakage | Evolvability of memory |
| M1 learning | Gymnax CartPole-v1 | Complete closed-loop evolution, batching, control score | Memory necessity in its fully observed form; any benefit of structural evolution, since a linear policy solves it |
| M1 diagnostic | Memory-task manifest evolved with direct recurrent genomes, ungated | Whether NEAT finds a memory loop before any decoder exists | An acceptance result |
| M2/M3 learning | Memory-task manifest: balanced delays with sampled distractors | Evolved memory, with interpolation and generalization accuracy reported separately | Broad agent intelligence; an advantage of indirect encoding, since inputs are few and non-geometric |
| M2/M3 geometric diagnostic | One-step visual discrimination on a small input grid, ungated | Whether coordinate-based encodings exploit input geometry, against direct genomes and the fixed-topology baseline | Sequential or memory ability |
| M2 contract check | W8 shared-sequence tracking over a long horizon | Shared read-only data, chunked continuation, non-additive reduction, causal observation | Any application-level result |
| Diagnostic extension | Partially observed CartPole/Pendulum with a declared observation mask | Memory in control, continuous/discrete action adaptation | Fair comparison unless mask and dynamics are fixed |
| M5 | Gymnax MinAtar Breakout or another dependency-compatible small game | Longer horizons, richer observations, exploration, simulation throughput | Hierarchy or multi-agent credit assignment |
| Optional | Pinned accelerator physics environment | Continuous high-dimensional control | Required milestone if compatibility is poor |
| Later research | JaxMARL cooperative or competitive task | Actor state ownership, partner sampling and coevolution | Provided merely by enabling EMR neuromodulation |

Gymnax provides classic control and MinAtar environments with functional accelerator-oriented interfaces. Reuse those environments, but verify wrappers and pin dependencies. [Gymnax](https://github.com/RobertTLange/gymnax)

JaxMARL provides a later multi-agent environment source; its integration requires a separately qualified dependency lane and agent-organization mechanisms. [JaxMARL](https://github.com/bold-lab-ai/JaxMARL)

No financial environment is required for standalone acceptance. Later trading needs causal data/fills, costs, accounting, and chronological validation as application tests.

## 8. Comparison protocol

Direct recurrent genomes run under the synchronous profile and substrates under the layered profile, so a comparison across them mixes encoding with execution timing. Two kinds of comparison are therefore reported, and a result is never moved from one kind to the other.

**System comparison.** Each pipeline runs as proposed: direct genomes under their own profile, fixed HyperNEAT, adaptive EMR and the fixed-topology baseline. It answers which complete system performs better on the task. Where profiles differ, it does not attribute the difference to the encoding. On feedforward tasks such as CartPole the profiles coincide and the confound is absent.

**Encoding-isolating comparison.** Execution timing and architecture family are matched. Every method runs a substrate-shaped graph under the layered profile: an input block, one hidden population with delayed hidden-to-hidden edges, and an output block.

| Method | What differs |
|---|---|
| Fixed-topology baseline B0 | Weights evolved directly; hidden size equal to the fixed substrate for that task |
| Fixed HyperNEAT | Weights generated by an evolved CPPN on the same substrate |
| Adaptive EMR | Hidden positions and weights generated by an evolved CPPN; discovered size differs, so active counts are reported beside the scores |

Direct recurrent genomes join this comparison only on feedforward tasks. For EMR the control matches timing and architecture family, not discovered size.

**Budgets.** Methods differ in cost per evaluation, so evaluation count and elapsed time cannot both be held equal. Report two results per task and never combine them: quality at a fixed evaluation budget, counted as candidate-episode evaluations; and quality at a fixed elapsed-time budget, which is the G9 research budget on the gating card with decoding included.

**Tuning.** Before acceptance runs, each method may try at most 8 configurations per task on non-acceptance seeds, and the count used is reported per method. The cap is a proposed value.

Report medians and ranges across evolution seeds. An encoding that loses is reported as losing.

## 9. Report artifact

Each acceptance report includes configuration/source/dependency hashes, hardware/backend evidence, workload dimensions, seeds, compile-cache state, per-stage and end-to-end timing, memory/pressure observations, correctness results, score distributions, capacity overflow, failed cells, and interpretation. Label every result as measured, estimated, unsupported, or not run. Preserve raw machine-readable rows alongside plots; publication plots should distinguish synthetic scale fixtures from evolved agents.
