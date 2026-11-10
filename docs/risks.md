# Risk register and empirical decisions

Severity describes impact on the objective, not a measured probability. Owners are components/work packages. Unresolved performance questions require the named experiment before a design commitment.

| ID / severity | Risk and trigger | Mitigation / experiment | Owner / decision point |
|---|---|---|---|
| R1 High | Windows driver/WSL GPU passthrough or pinned container CUDA dependencies prevent execution | Verify host-to-container GPU access, then JAX and full generations on each card; immutable image and compatible host-driver manifest; reject CPU fallback | Q0–Q2; infrastructure qualification |
| R2 High | Per-card VRAM or host/WSL memory is exhausted by queries, dense recurrence or compiler workspaces | Measure actual capacity; 60% initial VRAM budget with reserve, <=75% after pilot; independent host RAM budget; query slabs and microbatches; no VRAM pooling | Q2/S2 |
| R3 High | EMR candidate discovery grows much faster than useful network size | Record candidates versus active nodes; benchmark depth independently; per-genome pair queries and bounded slabs; frozen Q2 manifest and screening criterion with a dated fallback decision before A0; distinguish algorithmic restrictions from lossless optimization | Q2/A2/S2 |
| R4 Critical | Persistent runtime is confused with upstream per-input settling | Versioned runtime profiles; explicit K/reset semantics and delay labels; layered profile for substrates with feed-forward parity as the bridge; oracle, upstream settling parity, then persistent experiments | V1/V2/A1 |
| R5 High | Repeated recompilation makes topology evolution impractical | Dynamic topology arrays; fixed capacities/tails; signature logs; finite bucket family; no per-generation jitted function recreation | V2/S0 |
| R6 High | Sparse execution is slower than dense or poorly supported by the selected kernel path | Density/degree/B/K sweeps on both cards using real and synthetic graphs; retain dense fallback; custom kernels only after measured benefit | S1 |
| R7 High | Compaction or export changes graph semantics or silently drops edges | Separate immediate/delayed matrices and masks; mixed-delay conversion/export fixture; exact remaps, edge-set and activation preservation; overflow reject/promote; never silently cap degree | H1/A1/S0 |
| R8 High | Unspecified matrix-product precision, threshold rounding or unstable recurrence changes fitness rankings | Explicit float32 matrix-product policy before tracing; audit overrides and record policy; float64 reference from matched inputs; empirical fan-in-scaled thresholds calibrated on the CPU profile and gating card and validated on the second card; boundary probes and trajectory checks; reduced precision is a separate experiment | V2/A1/P1 |
| R9 Critical | Episode state or evaluation scores leak across candidates/chunks | Stable candidate IDs, state [p,B,...], permutation/chunk invariance, archive before tell, stale-score rejection | V4/V5 |
| R10 High | Research dependency APIs drift or feature flags are ignored | Exact SHAs, narrow adapters, capability rejection, per-feature tests; small patch branch only when necessary | Q0/A0 |
| R11 High | Slot-based cache returns another genome’s recurrent connections | Disable upstream slot cache; content/config cache only if measured worthwhile; test reproduction/reordering | A0/A2 |
| R12 High | NEAT search fails on memory despite a correct runtime | Fixed oracle first; one shared memory-task manifest; ungated direct-genome memory diagnostic in M1; known-solution initialization and weight-only controls; compare direct/fixed/adaptive under the comparison protocol | V5/H2/A2 |
| R13 High | Coordinate encoding or function palette biases search unhelpfully | Small controlled coordinate/palette ablation on training seeds; ungated geometric-input diagnostic against direct genomes and the fixed-topology baseline; keep held-out protocol fixed; report failure | H2/A2 |
| R14 Medium | Species computation or structural mutation dominates at large P | Time separately; compare bounded representative methods/islands only after baseline; preserve gene identity | S2/X0 |
| R15 High | Local burst performance hides sustained slowdown or memory growth | 15-minute minimum stability run; record power/load; repeated runs and warm-state memory plateau | Q2/S2 |
| R16 High | Huge networks look impressive but do not improve agents | Separate execution scale, learned complexity and held-out quality; standing fixed-topology weight-only baseline from M2; required system and encoding-isolating comparisons in the definition of done; no size-as-success metric | B0/H2–P1 |
| R17 High | Benchmark selection/tuning overfits held-out episodes | Frozen acceptance manifests, independent training seeds, run-level uncertainty, failed-run retention, a declared per-method tuning cap | P0/P1 |
| R18 Medium | Multi-GPU migration corrupts gene homology or carries incomparable scores | Global/namespaced lineage strategy, compatible decoders, receiver reevaluation, migration contract tests | X0, not first release |
| R19 Medium | Multiple roles/agents are assumed to emerge from graph size | Explicit team encoding, partner/opponent sampling and credit assignment experiments; reserved state/interface boundaries | X1 |
| R20 Medium | Scope expands into generic infrastructure before EMR works | Deliver vertical slice first; no plugin/service framework; every abstraction needs a current consumer | All tasks |
| R21 Medium | Distribution choices conflict with upstream licensing | Retain notices and dependency inventory; assess intended distribution before packaging/relicensing | Q0/P1 |
| R22 Medium | Docker/WSL filesystem overhead, temporary container storage or cache loss distorts runs and loses checkpoints | Linux-backed workspace/named volumes; explicit result/checkpoint mounts; separate cold/warm cache timings; container replacement test | Q0/V5/P1 |
| R23 Medium | WSL profiling, GPU selection or multi-GPU communication has platform-specific limits | In-container card selection with an identity check, because index filtering is unsupported under WSL; qualify available counters; keep missing metrics explicit; independent single-GPU runs first; test communication before multi-device claims | Q0/Q1/S2/X0 |
| R24 High | Generation-specific scenarios make fitness incomparable across generations, so stored species bests trigger stagnation by noise | Confirm upstream behavior at the pinned SHA; one declared comparability policy per manifest; record stagnation removals; fixed-scenario control in M1 | V3/V5; policy chosen before M2 acceptance |
| R25 Medium | The synchronous K=1 profile delays each hidden node by one observation and makes add-node mutations non-neutral | Layered profile for substrates; latency fixture; ungated synchronous CartPole comparison in M1; latency reported with results | V1/V5 |
| R26 Medium | Environment-suite dependencies conflict with the EMR JAX/NumPy pins or pull unplanned packages | Lock Gymnax and transitive packages in Q0; older release or separate lane decided there | Q0 |
| R27 Medium | The evaluator contract fits short simulator episodes but not long shared-data sequences | Shared scenario data in the V0 contract; V4 fixture; W8 at M2 | V0/V4/H2 |
| R28 High | Upstream feasibility timing is mistaken for the integrated research budget | Q2 is a screening ceiling on a frozen manifest; G9 pilot on each integrated workload, with compilation and evaluation counted; named G5 gate cells | Q2/V5/H2/A2 |
| R29 Medium | Cross-encoding results conflate encoding with execution timing or with unequal budgets | Separate system and encoding-isolating comparisons; B0 pinned to the layered profile and substrate shape; fixed-evaluation and fixed-time results reported separately | B0/H2/A2/P1 |
| R30 Medium | A ratio-style fitness is undefined or exploitable at low variance, or drifts over long horizons | Reducer defined with an additive floor and a minimum sample count; stable online accumulation with chunk merging; dedicated tests and thresholds on W8 | V4/H2 |

## Decision log to populate during implementation

| Decision | Current position | Required evidence |
|---|---|---|
| Container CUDA combination | EMR-compatible JAX 0.6.1/CUDA 12 starting point; exact image unqualified | Q0 lock and Q1/Q2 complete workload results on both cards |
| GPU variants and host placement | RTX 3060/3070 targets; VRAM and one-versus-two-host arrangement unknown | Q0 inventory; no assumption of combined execution |
| Research budget and throughput targets | Two hours per evolution seed per card, compilation and evaluation included (proposed); seconds per generation unmeasured | G9 pilot on each integrated workload |
| Dense versus sparse crossover | Dense reference first; an alternate runtime is adopted only if G8 passes | S1 latency/memory map including degree skew and actual EMR graphs |
| Bucket capacities | Single capacity first | Padding and compile-cost measurements; choose the smallest useful family |
| Runtime semantics | DAG; persistent layered for substrate decoders; persistent synchronous for direct recurrent genomes | Reference fixtures for all three; a change requires a new version/profile |
| Graph encoding | Edge lists for direct genomes; immediate/delayed matrices with separate masks for dense substrates | Mixed-delay conversion and export/reload tests; actual allocated-memory comparison on W2/W6 |
| Baseline arithmetic | FP32 arrays with `jax_default_matmul_precision="float32"`; float64 NumPy reference from matched inputs | V2/H1 differential tests, audited overrides and recorded precision metadata |
| Cross-generation fitness comparability | Generation-specific scenarios in M1, with stagnation removals measured | M1 fixed-scenario control; policy chosen before M2 acceptance runs |
| Minimum adaptive envelope | Proposed Q2 screening ceiling on the frozen manifest: max_depth=4, P=128, at most 28 s per warm generation, per card | Q2 measurements; dated fallback decision if missed |
| Gating card | Named at Q0; the smaller-VRAM card by default; numerical thresholds calibrated on it and the CPU profile, validated on the second card | Q0 inventory; second-card G1 result |
| Memory task | One shared manifest with sampled distractors and balanced delays (proposed) | M1 diagnostic results; any change is dated and applies to M1, M2 and M3 together |
| Comparison protocol | System and encoding-isolating comparisons; fixed-evaluation and fixed-time results reported separately; tuning cap of 8 configurations per method (proposed) | M2 comparison reports |
| Environment suite lock | Gymnax and transitive packages resolved against the EMR pins | Q0 resolution result |
| K and recurrent stability controls | K=1 first; fixed tanh; no hidden clipping | Temporal-learning and stability experiments; any clipping/normalization is explicit |
| EMR feature subset | Canonical variance, cache off, sum, small palette, supported standard mode | A0 capability report before more combinations |
| Adapter versus fork | Adapter preferred | Demonstrated missing public seam; patch scope and parity tests |
| Multi-device model | Independent runs/islands later | Qualified hardware, communication measurements and lineage design |
| Hierarchical/team genome | Deferred | Specific benchmark and credit-assignment hypothesis |

GPU qualification failure is a concrete backend constraint, not evidence that JAX or neuroevolution must be abandoned. Conversely, CPU success is not evidence of accelerator performance. Keep those conclusions separate in status reports.
