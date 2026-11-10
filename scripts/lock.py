"""Convert pip's Linux resolution report to exact, hash-checked requirements."""
import json
from pathlib import Path

report = json.loads(Path("locks/resolution.json").read_text())
lines = []
for item in report["install"]:
    m = item["metadata"]
    digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
    lines.append(f'{m["name"]}=={m["version"]} --hash=sha256:{digest}')
Path("locks/requirements.txt").write_text("\n".join(sorted(lines, key=str.lower)) + "\n")
