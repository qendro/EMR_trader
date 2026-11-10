# Immutable upstream source

These gzip archives contain complete `git archive` snapshots, including LICENSE files, at the exact commits in `locks/sources.json`. TensorNEAT is the submodule commit recorded by EMR. Their checksums are verified by the Docker build. No project patches are applied.

Regenerate from a fresh checkout using `git clone --recursive`, `git checkout` to the recorded commit and `git submodule update --init`, then `git archive --format=tar` on each repository. Run `scripts/compress_sources.py` inside the pinned Python base container; deterministic gzip timestamps make checksums reproducible. Source archives are distributed under their bundled GPL notices. No license for newly written project code is selected here.
