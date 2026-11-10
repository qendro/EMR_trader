"""Run in the base container after git archive; retain complete upstream notices/source."""
import gzip
import hashlib
import json
from pathlib import Path

lock = json.loads(Path("locks/sources.json").read_text())
for name, key in (("emr", "emr"), ("tensorneat", "tensorneat")):
    raw = Path(f"vendor/{name}.tar")
    destination = raw.with_suffix(".tar.gz")
    destination.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
    lock[key]["archive_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    lock[key]["license"] = "GPL-3.0-or-later (bundled LICENSE)"
    raw.unlink()
Path("locks/sources.json").write_text(json.dumps(lock, indent=2)+"\n")
