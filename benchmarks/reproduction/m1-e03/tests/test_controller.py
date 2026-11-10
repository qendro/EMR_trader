import json
from neuroevo.runner import load_config,run
from neuroevo.artifacts import load_checkpoint
from neuroevo.runner import provenance


def test_controller_resume_history_and_finalists(tmp_path):
    config=load_config("configs/m1-smoke-v1.json")
    run(config,900,tmp_path,{},stop_after=2)
    saved=load_checkpoint(tmp_path/"checkpoint.bin",{**config,"run_seed":900},provenance())
    assert all("total_seconds" in r for r in saved["history"])
    run(config,900,tmp_path,{},resume=True)
    rows=json.loads((tmp_path/"history.json").read_text())
    assert [r["generation"] for r in rows]==list(range(4))
    assert all(r["total_seconds"]>0 for r in rows)
    selection=json.loads((tmp_path/"selection.json").read_text())
    assert len(selection["finalist_ids"])==4
    assert (tmp_path/"validation-finalists-genes.npz").exists()
