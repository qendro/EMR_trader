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

The long-run controller writes generation history as JSONL and restores it
from the last checkpoint on resume. This avoids rewriting the entire history
on every generation. The full acceptance seeds and their quality comparison
are still pending. Their output root is `/results/m2-elapsed-v2` in the
`neuroevo-results` Docker volume; `scripts/m2_elapsed_acceptance.py` runs
one seed at a time and records progress there.

The calibration runs, first failed cap pilot and both sets of passing final
pilots are retained under [the hashed evidence index](results/evidence/index.json).
