import json
from pathlib import Path
report=json.loads(Path("locks/compatibility-resolution.json").read_text())
lines=[]
for item in report["install"]:
    m=item["metadata"]
    digest=item["download_info"]["archive_info"]["hashes"]["sha256"]
    lines.append(f'{m["name"]}=={m["version"]} --hash=sha256:{digest}')
Path("locks/compatibility.txt").write_text("\n".join(lines)+"\n")
