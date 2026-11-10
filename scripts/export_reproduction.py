"""Retain the exact source that produced a checkpoint, independent of later edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import neuroevo

p=argparse.ArgumentParser()
p.add_argument("--output",required=True)
args=p.parse_args()
root=Path(args.output)
package=Path(neuroevo.__file__).parent
files=sorted(package.rglob("*.py"))
digest=hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
for original in files:
    destination=root/"src/neuroevo"/original.relative_to(package)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,destination)
for folder in ("configs","tests","locks"):
    for original in Path("/app",folder).glob("*"):
        if original.is_file() and original.suffix in (".json",".txt",".py"):
            destination=root/folder/original.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,destination)
shutil.copyfile("/app/pyproject.toml",root/"pyproject.toml")
(root/"source.json").write_text(json.dumps({"engine_sha256":digest,
    "purpose":"exact source for recorded training/checkpoint artifacts; current importer adds stricter validation"},indent=2)+"\n")
