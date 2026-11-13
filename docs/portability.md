# Checkpoint and phenotype portability

Run experiments inside the pinned Docker image with a persistent result
mount. Each seed directory records its manifest, backend settings, selected
finalists, history, native checkpoint, and both evaluated-winner and champion
phenotypes. The evaluated winner is written before reproduction changes the
native population. The final champion is chosen on validation episodes and
evaluated once on the held-out namespace.

To continue a native run, use the same image, config, seed and result directory
with `run --resume`. The runner checks the config hash and source provenance
before loading `checkpoint.bin`; it discards JSONL rows newer than the last
checkpoint. A different source version must be treated as a new experiment.

For inference, use `replay --phenotype <champion.npz> --output <report.json>`.
The exported phenotype contains weights, masks, node IDs, profile and action
metadata. Replay compares 128 carried steps against the float64 NumPy
reference and checks that TensorNEAT/EMR are not imported. This tests the
portable policy artifact independently of the native evolution checkpoint.

The current [P1 v2 audit](../benchmarks/p1-crossbackend-v2-audit.json) records
identical replay actions on CPU and both GPUs, 117 passing tests per profile,
and three generations of checkpoint continuation after container replacement.
The [acceptance ledger](../benchmarks/PLATFORM-STATUS.md) separates this
portability result from earlier milestone comparisons that remain open.
