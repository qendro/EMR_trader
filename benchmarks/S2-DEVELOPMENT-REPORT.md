# S2 development check

A one-generation CPU adaptive memory smoke used the registered depth-1,
population-8 v2 smoke configuration with a 600,000-byte rollout estimate
budget. The unmodified task and candidate seed produced best training score
1.0 and mean 0.6328125. Training used microbatch 8 across its four hidden
buckets. Validation used batch 7 for the 16-wide bucket, and the 256-episode
held-out and generalization rollouts used batch 2 for the 8-wide finalist.
The [history](results/s2-memory-smoke-v2/seed-900/history.json) records each
decision. One generation is a scheduling smoke, not a long-run memory gate.

The process reported peak RSS 1,223,278,592 bytes and zero cgroup swap.
This includes JAX and native discovery; the 600,000-byte number describes
the scheduler's estimated rollout arrays only. A decoded-phenotype host
storage ceiling now refuses oversized populations before conversion and
records actual compact-plus-padded bytes. Its cap-refusal test passes, but
The separately [frozen S2 smoke](../configs/s2-adaptive-memory-smoke-v1.json)
used discovery batches of two, a 100,000-byte converted-phenotype limit and
the same 600,000-byte rollout estimate limit for three full generations.
Its best and mean training histories exactly matched the pre-S0 and S0
memory smokes. Converted compact-plus-padded phenotype storage was 14,981
bytes per generation; process peak RSS was 1,373,237,248 bytes, cgroup swap
zero. The higher process peak reflects native/JAX allocations outside the
converted-plan budget. The opt-in public EMR slice path has array-by-array
phenotype parity with the unsliced path in the CPU test.

Full feedforward query temporaries, 100-generation compilation stability,
W6, W7 and single-card peak-memory evidence remain open.
