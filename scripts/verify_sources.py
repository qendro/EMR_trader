import hashlib
import json
from pathlib import Path
import tarfile

lock = json.loads(Path("/locks/sources.json").read_text())
for name in ("tensorneat", "emr"):
    archive = Path(f"/vendor/{name}.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock[name]["archive_sha256"]:
        raise ValueError(f"Source archive checksum mismatch: {name}")
    with tarfile.open(archive) as source:
        source.extractall(f"/opt/{name}", filter="data")
