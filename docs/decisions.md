# Dated implementation decisions

## 2026-11-12: Q2 declared capacity passes; A0 may start

The separately frozen depth-4, P=128 Q2 capacity promotion used a 32,768-slot
recurrent cap with refusal on overflow. Its [capacity/resource report](../benchmarks/Q2-CAPACITY-REPORT.md)
and [synchronized stage report](../benchmarks/Q2-STAGE-REPORT.md) each pass all
12 cells: both GPUs, feedforward and recurrent modes, and seeds 700–702. The
largest discovered recurrent population member had 15,104 hidden connections.
Cold, warm, host-memory and GPU-memory ceilings passed. The stage probe used
the same frozen workload and matched every capacity cell's fitness and graph
sizes; its synchronized per-stage warm medians and p95 values are retained.

Q2's upstream feasibility screening criterion is therefore **passed for this
declared envelope**. This decision permits adaptive adapter work A0. It does
not claim that upstream independent-sample timing is the integrated research
budget, that adaptive rollout has been implemented, or that the interrupted
17/20-seed M2 fixed-time comparison is complete. The original 10,000-slot
failures remain in the evidence.

## 2026-11-10: H0/H1 layered and fixed-coordinate increment

H0/H1 are implemented with a concrete single-hidden role-block schema alongside the general mixed-delay graph plan. Fixed queries, virtual bias, coordinate order, strict threshold equality and independent genotype/phenotype capacities are frozen in the H1 manifests. Native TensorNEAT evolution is unchanged; the existing controller archives compact CPPN genes plus the decoded phenotype before tell. M1's exact source/image remain retained, and its checkpoint provenance is not relabelled as H1.

The first larger-fan-in calibration missed the CPU output reduction layout. Its c=30 failure and all earlier c-dependent evidence are retained and superseded by a dated, separately frozen v2 protocol. [The full decision and rerun scope](h1.md) document c=456, the actual absolute/relative errors and all per-profile checks. All 54 tests, 256/1,024-node execution fixtures and wiring replays pass on CPU and both cards. H0/H1 close as work packages; M2 remains open. B0 is next, followed by H2. No M2 learning/performance threshold was changed or claimed from smoke runs.

## 2026-11-10: Q2 connection-cap failure

The frozen `q2-v1.json` workload completed on the RTX 3060 for feedforward seeds 700–702 and recurrent seeds 700–701. Recurrent seed 702 stopped during generation 5: the upstream default `max_sparse_conns=10000` could not hold a discovered set of **13,424** hidden-to-hidden connections. The failure is preserved; it is not a timing pass, and no connection was silently removed.

Both cards completed feedforward seeds 700–702 and recurrent seeds 700–701. RTX 3070 recurrent seed 702 independently failed at the same 13,424-connection discovery. Q2 remains open until the per-card protocol is complete and resource evidence is audited. Original run directories remain unchanged.

Query slabbing cannot repair an insufficient output-connection capacity. The selected next experiment preserves depth 4, P=128, all discovery thresholds and both timing ceilings, and raises the declared fixed connection capacity to 32,768 with `sparse_cap_policy=refuse`. This is capacity promotion, not pruning. It adds approximately 38 MB of padded sparse records at P=128 (13 bytes per extra slot), before workspaces. This is an estimate, not measured peak memory. If another overflow occurs it is recorded and the envelope is reconsidered; no automatic truncation is permitted. A separately versioned manifest must be frozen before that experiment. A0 is not implemented in this increment.

## 2026-11-10: native backend observer compilation

The adapter's stagnation diagnostic originally called upstream JAX control-flow helpers eagerly each generation. It now compiles that observer once, with the full backend state and fitness as dynamic arguments. The diagnostic does not update native state. Operator parity, complete native 40/100 replay, controller resume and all numerical/batching checks were rerun. Final training runs use image `1a6b04f37a5a8e39806b1acbd1ece24a591656eade41656f2c6f28e925cd471b`; older pilot measurements are retained separately and do not supply final acceptance evidence.

## 2026-11-10: timing and archive completeness

Generation timing now includes history serialization and progress logging. Checkpoints include a complete timing row and accumulated elapsed budget across restarts. Final validation candidates and their training/validation scores are persisted before tell. The unchanged acceptance thresholds use fresh pilots for this source version. No held-out acceptance result was used to change parameters or thresholds.

## 2026-11-10: numerical overflow and strict import validation

Final audit found that an infinite FP32 pre-activation could become finite through saturating tanh. The runtime now marks that candidate invalid before activation; a regression fixture uses max-FP32 weight times observation 2. The independent float64 reference legitimately remains finite on that example and separately rejects nonfinite weights and its own accumulator overflow. Portable import also rejects unknown activation vocabularies, unsupported action mappings and noncanonical array dtypes; configuration validation checks task interfaces, seeds and finite targets.

The final engine hash is `8d3aba6552b6872ab47ddca1c1992ee839ce5339c9f48ba221ba780525fddebb`, built from the root Dockerfile as image `7172c6d3749ceef15f01283145f0049e622f26202a36d88957b7486ec31a2a80`. All 44 tests passed independently on CPU and both cards. Fresh pilots and learning runs use separate `m1-current-*` directories and the unchanged acceptance manifest. Earlier full learning/checkpoint results retain the exact `e03c0613…` source bundle and image; they do not retroactively qualify the final arithmetic guard.

## 2026-11-10: performance sampling and diagnostic controls

G4 compares three fresh-process controller/minimal-driver pairs, seed 900, production W3 dimensions, archive/logging diagnostics and checkpoint interval 10. The sample comprises generations 2–10 inclusive in both paths. This excludes first-process warm-up and terminal selection/forced terminal checkpoint symmetrically. Trial-0 minimal driver continues to generation 102 for a distinct 100-post-warm-generation G2 audit; only the same nine generations enter its G4 comparison. Quality fields must match exactly. These measurements are run sequentially after other project experiments stop. G5 uses those three controller samples per profile, with observed ranges reported; no M1 acceleration requirement is introduced.

The diagnostic DAG control uses seeds 100–104 with generation-specific scenarios. Fixed-scenario and synchronous controls use the same seeds and capacities. All are explicitly diagnostic, not replacements for acceptance seeds 0–4. Final held-out memory accuracy and first training-threshold generation are separate measurements: the latter is not a held-out time-to-solve claim. Persistent compilation caches are retained and their initial entry counts recorded; process-cold timings must not be described as empty-cache compilation when cache entries already exist.

## 2026-11-10: M1 gate closure and small-workload timing

The final engine passed 44 tests and all five frozen CartPole acceptance seeds on CPU and each card. All three profiles completed 100 post-warm-up generations with zero calls to the actual JAX backend compiler and unchanged top-level signatures. Complete native CPU resume and backend-independent replay passed. G3 stayed within the recorded host/VRAM envelope without observed swap-out growth. G4 ratios of the three-run controller and minimal-driver medians are CPU 0.989, RTX 3070 1.033 and RTX 3060 0.950, below the unchanged 1.20 gate. The raw CPU pair with ratio 1.486 remains in the report and its aggregate; no timing outlier was dropped.

M1's mandatory gates therefore close on this first increment. W3 GPU/CPU throughput ratios are 0.142 (RTX 3070) and 0.132 (RTX 3060): these small dynamic DAGs run faster on CPU. M1 reports G5 without gating, as specified before measurement. This result does not justify introducing alternate kernels or reduced precision. H0 is the next implementation package. Q2's independent capacity failure and resource/stage audit stay open before A0; raw upstream fixed/direct timing does not implement H0/H1.

## 2026-11-10: raw upstream baseline invalid candidates

The original `q2-native-v1` direct wrapper passed raw nonfinite fitness to the native algorithm. Mutated genomes produced NaN outputs/fitness, and strict JSON rejected the NaN best-fitness summary (CPU seed 700 at generation 5, seeds 701/702 at generation 7). These failed cells and partial histories remain unchanged. They are not valid baseline timing passes. The fixed baseline's independent-sample settling semantics remain separately labelled.

Before fresh runs, `q2-native-v2.json` freezes explicit invalid-output handling: any nonfinite output/score makes that candidate invalid, its fitness is -1e30, valid/invalid counts are recorded per generation, and an all-invalid population aborts. Best fitness uses finite candidates only. The native operators, capacities, functions, seeds, dataset and discovery thresholds are unchanged. The separate v2 runner/manifest and output directories prevent its times from replacing v1 failures. This is an experimental evaluation-boundary change, not an upstream evolution patch or a change to the qualified M1 engine.

V2 also freezes one initially empty persistent compilation cache per profile/mode. Seeds 700, 701 and 702 run in that order in fresh processes. Seed 700 measures the empty-cache lane; 701/702 retain the cache, with initial entry counts recorded, and do not claim empty-cache compile time. The original v1 matrix keeps its fresh disposable cache per cell. This distinction is part of the manifest before v2 starts and changes no Q2 screening ceiling.

Source inspection identifies an inference edge case: pinned TensorNEAT `common/functions/__init__.py::apply_aggregation` explicitly returns NaN when all incoming terms are NaN, including empty incoming connectivity. The platform contract deliberately defines an empty sum as zero and evaluates an isolated node's bias. M1 reuses native evolution, not this padding convention; upstream inference parity is claimed for compatible connected fixtures, not isolated nodes. Raw Q2 baselines retain upstream inference and therefore need the separately declared invalid-candidate boundary. This difference is not patched in the upstream source.

The v1 matrix finished with nine failed direct cells and nine completed fixed cells across CPU and both cards. All 18 v2 cells then completed. Direct seeds 700/701/702 recorded 26/21/38 invalid candidate evaluations on each profile; fixed cells recorded zero invalid outputs. The first-invalid CPU genome replay has finite active attributes, isolated output ID 4, and 64 nonfinite outputs in column 0. Its replay reproduces those outputs. The dataset audit verifies all 49 persisted datasets, v2 protocol/script identity, complete histories, counts and empty/retained cache policy. Q2 remains open for the original EMR capacity failure, capacity-promotion run and full resource/stage audit; successful native baselines do not close that gate.

## 2026-11-10: H2 learning and W8 task protocol

The frozen fixed-HyperNEAT acceptance manifests used the same 0–4 seed,
population and episode budgets as B0. Fixed CartPole passed 4/5 seeds and
fixed memory passed 5/5 on RTX 3070. The separate direct recurrent memory
system cell passed 4/5. Its generic 128-step synthetic replay overflowed FP32
on seeds 0 and 2; the exported champions nevertheless reproduced every actual
held-out memory decision and matched the independent scalar reference. Both
outcomes are retained in the H2 report and raw evidence. No parameter or
threshold was changed after the held-out scores were inspected.

`w8-v1.json` had fixed the Welford/Chan reducer but did not define an executable
tracking environment. `w8-tracking-v2.json` freezes the first such environment:
two current channels, next-step channel-zero target, bounded action/reward,
10,000 steps and 250-step continuation chunks. It was written before its
three-profile measurements. The v1 reducer tolerances and minimum count were
reused unchanged. This is a contract check on a fixed exported policy, not an
evolutionary learning acceptance result.
