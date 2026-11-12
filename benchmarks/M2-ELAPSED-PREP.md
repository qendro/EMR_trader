# M2 fixed elapsed-time comparison preparation

The remaining M2 quality comparison gives B0 and the fixed-coordinate CPPN
two hours per seed on the RTX 3070, with decoding, checkpoints, champion
selection and held-out evaluation inside the limit. Each method runs five
seeds on CartPole and five on the delayed-cue memory task. These runs are
separate from the completed fixed-evaluation-budget comparison.

The four calibration manifests used seed 900 and 12 integrated generations.
The resulting frozen acceptance caps were 47,900, 35,800, 5,700 and 5,200
generations for B0 CartPole, B0 memory, fixed CartPole and fixed memory.
The first matching B0 CartPole pilot found an allowable cap of 47,622, so
its 47,900 cap failed G9. The failure is retained. Before any acceptance
seed, its manifest was versioned to a 45,000-generation cap. The other
three caps did not change. See [the freeze record](m2-elapsed-freeze.json)
and [revision record](m2-elapsed-revision.json).

The first four matching G9 pilots passed in image
`sha256:f69c6ea5dbb7d24b0b1f13e71edcddc0e92db82d2a856b618840aa402b238da8`:

| Cell | Frozen cap | Pilot allowable generations |
| --- | ---: | ---: |
| B0 CartPole v2 | 45,000 | 47,806 |
| B0 memory v1 | 35,800 | 37,867 |
| Fixed CartPole v1 | 5,700 | 6,149 |
| Fixed memory v1 | 5,200 | 5,781 |

The first fixed CartPole acceptance seed was interrupted after generation
456 when its Docker container exited 255. It was not OOM killed. The run had
not reached its first checkpoint at generation 500; no acceptance result was
produced. Its partial history and console log are retained under
[`m2-elapsed-v2-driver591`](results/evidence/m2-elapsed-v2-driver591/).
The container's replacement driver was 610.88, versus 591.74 in the first
pilots. The acceptance provenance check correctly rejected those older pilots.
All four pilots were rerun on the same image and new driver before restarting
the seed:

| Cell | Frozen cap | Driver 610.88 allowable generations |
| --- | ---: | ---: |
| B0 CartPole v2 | 45,000 | 53,035 |
| B0 memory v1 | 35,800 | 40,495 |
| Fixed CartPole v1 | 5,700 | 6,120 |
| Fixed memory v1 | 5,200 | 5,672 |

The first v2 fixed CartPole seed reached generation 4,905, then failed the
7,200-second budget before its 5,700-generation cap. Its population grew
from roughly 7 to 30 CPPN nodes, and late generations took roughly 1.7–2.0
seconds instead of the pilot's 1.07 seconds. Its budget failure, history,
checkpoint and console log are retained under
[`m2-elapsed-v2`](results/evidence/m2-elapsed-v2/). This is a failed
acceptance attempt; it has no held-out selection result.

The revised runner stops after the first completed generation at or past
6,480 seconds from process start, then performs validation, held-out
evaluation, checkpointing and reporting within the 7,200-second budget.
The 100,000-generation field is a safety ceiling; it does not determine
the comparison endpoint. All four methods use this same rule. The G9 pilot
for a wall-stopped run checks the 720-second finalization reserve. The
intermediate v3 B0 caps missed their matching pilots, and the v4 caps passed
but could finish before the shared elapsed cutoff. Both revisions are
retained in [the v3 freeze](m2-elapsed-v3-freeze.json),
[v4 cap revision](m2-elapsed-v4-revision.json) and
[v5 freeze](m2-elapsed-v5-freeze.json).

The long-run controller writes generation history as JSONL and restores it
from the last checkpoint on resume. The full v5 acceptance seeds and their
quality comparison remain pending. Their output root is
`/results/m2-elapsed-v5` in the `neuroevo-results` Docker volume;
`scripts/m2_elapsed_acceptance.py` runs one seed at a time and records
progress there. The four matching v5 pilots pass on image
`sha256:d4f395ef90f2cbdb5124f4677b40416ceb93f92b157b1b9143154c8cb4a95e64`
with driver 610.88.

Calibration runs, failed cells, intermediate revisions and passing pilots
are retained under [the hashed evidence index](results/evidence/index.json).

## User-directed early stop, November 12, 2026

The v5 comparison was stopped after 17 of 20 seeds so the independent Q2
capacity screen could use the GPUs. The container was stopped during B0 memory
seed 2; its completed checkpoints remain in the `neuroevo-results` volume.
Restarting the controller can resume that seed from its last checkpoint.
The stopped container's exit code 137 reflects the requested stop and was not
reported as an out-of-memory event.

| Method and task | Complete seeds | Held-out scores |
| --- | ---: | --- |
| Fixed CPPN CartPole | 5/5 | 500, 490.90, 487.95, 498.25, 500 |
| B0 CartPole | 5/5 | 500, 500, 500, 500, 500 |
| Fixed CPPN memory | 5/5 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| B0 memory | 2/5 | 1.00, 1.00; seeds 2–4 unknown |

These are the measured values printed by the v5 controller. The two completed
B0 memory seeds suggest a strong result but do not establish the required
five-seed comparison. No score is assigned to an uncompleted seed, and M2
remains open until the missing runs and final audit are complete.
