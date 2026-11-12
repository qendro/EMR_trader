# S0 adaptive packing development check

The first S0 CPU smoke uses the existing depth-1, population-8 adaptive
memory and CartPole manifests with source-mounted development code. This is
correctness evidence, not a clean performance measurement: the gating GPU's
older immutable-image acceptance run was active on the same host.

| Task | Training best by generation | Routed hidden widths | Run result |
| --- | --- | --- | --- |
| Memory | 1.0 / 1.0 / 1.0 | 4, 8, 16, 21 | Same quality history as pre-S0 smoke |
| CartPole | 68.0 / 50.5 / 60.75 | 4, 8, 16, 20 | Same quality history as pre-S0 smoke |

The memory run's declared node cap is 24: two inputs, one output and at most
21 hidden execution slots. Its per-generation routing selected four of the
five possible bucket widths, with stable memberships in all three generations.
The memory rollout had four compiled signatures after the first generation;
final validation and held-out episode shapes added four more, as expected.
There was no new training signature in generation 1. The full bucket and
scenario-shape ceiling remains to be audited over a longer changing-topology
run before S0 is considered accepted. The bounded raw artifacts and hashes
are in the [S0 evidence index](results/evidence/s0-development-v1/index.json).

The exact index map, padding, permutation and promotion contracts are in
[S0](../docs/s0.md). S1 will prototype a recurrent edge list against these
actual graph sizes and retain the dense path unless the G8 adoption threshold
is measured to pass.
