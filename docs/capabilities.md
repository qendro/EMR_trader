# Supported platform capabilities

| Area | Supported | Boundary |
| --- | --- | --- |
| Direct genomes | TensorNEAT node/connection evolution, DAG and synchronous profiles | Declared node/edge capacity; invalid candidates receive explicit failed fitness |
| Fixed HyperNEAT | CPPN coordinates, fixed role blocks, persistent layered carry | Configured coordinates and query slab; no implicit geometry expansion |
| Adaptive EMR | Public substrate discovery, per-genome active positions and edges, native tell | Pinned EMR version; sum aggregation, tanh hidden units, identity platform output |
| Weight-only baseline | Dense fixed topology, mutation, truncation selection and elitism | Fixed topology and declared parameter bounds |
| Runtime | FP32 JAX dense graph/role blocks, opt-in recurrent edge blocks | Edge blocks only for layered adaptive plans; dense remains default |
| Environments | Delayed-cue memory, full/partial CartPole, 5x5 orientation diagnostic | One policy action head; no market simulator |
| Persistence | Native checkpoints, evaluated/champion phenotype export, portable replay | Checkpoint resume requires matching engine provenance; exported phenotype replays without evolution imports |
| Hardware | Container CPU, RTX 3060 and RTX 3070 profiles | A measured gate on one GPU does not close the other GPU |

The A0 adaptive recipe rejects unsupported dynamic functions, custom
aggregation, neuromodulation and silent connection truncation. A discovered
graph keeps its full-grid indices in the archive and is compacted only for
execution. `BucketFamily` refuses overflow until the node capacity is
explicitly promoted. `discovery_microbatch`, phenotype storage and rollout
budgets are optional manifest controls; measured device and process peaks
remain the authority for capacity claims.

See the [acceptance ledger](../benchmarks/PLATFORM-STATUS.md) for which
capabilities have passing research and performance evidence.
