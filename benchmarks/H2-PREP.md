# H2 comparison preparation

The frozen small-substrate and direct-memory acceptance cells have now run on
RTX 3070. See the [measured H2 report](H2-REPORT.md); this document preserves
the pre-run pilot and comparison setup.

The small fixed substrates and B0 baseline use the same layered runtime,
population 128, hidden sizes (CartPole 8, memory 4), episode counts and
generation caps. Their frozen acceptance manifests use seeds 0–4. B0's measured
scores and complete artifacts are in [B0-REPORT.md](B0-REPORT.md).

On November 10, 2026, the RTX 3070 integrated G9 pilots for the fixed-HyperNEAT
manifests passed on image `neuroevo:b0` (engine
`90ec21d50ecf68fef4e0c13e526bde8a469262fde40b32fd6c2d4e8b7cb934be`).

| Fixed task | Config SHA-256 | Warm seconds/generation | Fixed overhead seconds | Declared generations | Pilot result |
|---|---|---:|---:|---:|---|
| CartPole | `b7d1302cced3a628b9a5de6a9b12b00098c4f31be52c708539b1c197f939ae4a` | 1.124 | 45.1 | 100 | pass |
| Memory | `da323d66efa011292604a27933648b2389b2c2cb18dc317f6b15692f96a6ca05` | 1.166 | 19.0 | 250 | pass |

These pilots are feasibility measurements on 12-generation integrated runs.
They do not establish five-seed learning quality. The fixed acceptance runs can
be launched with `scripts/h2_fixed_runs.ps1 -Image neuroevo:b0 -Profile rtx3070`.
That script uses a separate acceptance result root, obtains a new matching
pilot before each experiment, then runs the five seeds and portable replay.

The direct recurrent memory diagnostic used seeds 100–104 under synchronous
timing. `configs/h2-direct-memory-acceptance-v1.json` freezes a distinct
seeds-0–4 system-comparison cell with the same shared memory workload. It has
not been piloted or run. Direct recurrent memory results belong only in the
system comparison because execution timing differs from layered substrates.
The already measured M1 direct CartPole result can be identified by its older
source/image provenance; feedforward timing is compatible.

The [W8 contract check](W8-REPORT.md) has since passed on CPU and both GPUs.
The geometric diagnostic, fixed-elapsed-time comparison and integrated
performance gates remain open. M2 remains open.
