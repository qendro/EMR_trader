"""Read bounded JSON summaries without importing any numerical dependency."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
for p in sorted(root.rglob("selection.json")):
    value=json.loads(p.read_text())
    print(json.dumps({"path":str(p),"heldout":value["heldout_score"],
                      "generalization":value.get("generalization_score"),"kind":value["kind"]}))
