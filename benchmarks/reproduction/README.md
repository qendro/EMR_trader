# Recorded source snapshots

These are provenance for experiments performed before the final source audit.
They are not imported by the current package or its tests. Each `source.json`
contains the byte hash of all sorted package Python files, as used by checkpoint
validation. Source archives and dependency locks remain at the repository root.

| Directory | Recorded use | Retained local image |
|---|---|---|
| `m1-e03` | Earlier full learning runs and CPU 40/100 resume, engine `e03c0613…` | `neuroevo:final`, digest `1a6b04f37a5a8e39806b1acbd1ece24a591656eade41656f2c6f28e925cd471b` |
| `m1-8d3` | Qualified M1 after strict import and overflow fixes, engine `8d3aba65…` | `neuroevo:m1-qualified`, digest `7172c6d3749ceef15f01283145f0049e622f26202a36d88957b7486ec31a2a80` |
| `h1-e4fb` | Qualified H0/H1 core, engine `e4fbdd75…`; calibration protocols and coefficients retained | `neuroevo:h1`, digest `5c567eb307576b52543497c16f7a6fdaedb333272067b4d774b508dbcc29fd46` |
| `b0-90ec` | Measured B0 baseline, engine `90ec21d5…`; source, tests and image-matched configs retained | `neuroevo:b0`, digest `adb5659fed40c81f17c6ae5392a1fc9572ea944b63717dbb08c3786171f80321` |
| `h2-90ec` | Measured small-substrate H2 learning and direct-memory system cell; image-matched source and configs | `neuroevo:h2`, digest `1570d9ac91843f152342fa9a8fe09a62e4e05de47b67cb52e2aeecbd30133f0a` |
| `q2-c53` | Original upstream Q2 screens, including both connection-cap failures | `neuroevo:q2-lane`, digest `c294142b115e520a548f15452a6620efadf3675385861d3dd54b2a8c6146207c` |
| `diagnostics-66` | Earlier diagnostic containers built from `neuroevo:qualified`; development source mounts may differ and are identified in run manifests | `neuroevo:qualified`, digest `1ec09aa0fdef06b7132e4f56efd1093957405cbbce7a40613d16b8679a4a4a26` |

The final M1 learning results are `m1-current-*`, engine `8d3aba65…`; their exact
source is retained in `m1-8d3`. Root `src/` has subsequently advanced to H0/H1,
engine `e4fbdd75…` for the H1 report; B0 advances root `src/` to `90ec21d5…`.
The earlier `m1-final-*` directories do not qualify the
final pre-activation overflow guard. Native checkpoints intentionally reject a
different source hash. Portable phenotype inference has no such evolution-state
dependency and uses versioned, validated NPZ arrays without importing TensorNEAT
or EMR.

Docker image names are local convenience tags. Recorded digests, engine hashes,
dependency locks and source commits identify experiments. Rebuilding after docs
or evidence change can produce a different image digest without changing the
engine; a new run records that digest and must have its own matching G9 pilot.
