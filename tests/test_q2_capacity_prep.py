"""Host-only checks for the Q2 capacity evidence boundary."""
import hashlib
import json
from pathlib import Path

from scripts.q2_capacity import cold_completed
from scripts.q2_capacity_audit import audit_cell


def test_cold_watchdog_requires_second_completed_generation():
    assert not cold_completed('{"Q2":"recurrent","generation":0}\n', 1)
    assert cold_completed('noise\n{"Q2":"recurrent","generation":1}\n', 1)


def test_capacity_audit_checks_full_edge_distribution(tmp_path):
    manifest_bytes=Path('configs/q2-capacity-v2.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    digest=hashlib.sha256(manifest_bytes).hexdigest()
    cell=tmp_path/'rtx3070-feedforward-700'
    cell.mkdir()
    hardware={'backend':'gpu','gpu':['uuid','NVIDIA GeForce RTX 3070','8192','610.88'],
              'image_digest':'sha256:fixture'}
    (cell/'manifest.json').write_text(json.dumps({'manifest':manifest,'hardware':hardware,
                                                  'seed':700,'mode':'feedforward'}))
    (cell/'launcher.json').write_text(json.dumps({'manifest_sha256':digest,'profile':'rtx3070',
        'mode':'feedforward','seed':700,'image_digest':'sha256:fixture','exit_code':0,
        'oom_killed':False,'external_timeout':None,'cold_generations_observed':True,
        'physical_gpu_memory_used_mib_peak':1000}))
    (cell/'result.json').write_text(json.dumps({'timing_pass':True,
        'cold_initialize_generation_seconds':30}))
    (cell/'dataset.npz').write_bytes(b'fixture')
    sizes={'candidate_positions':1364,'active_nodes_per_genome':[5]*128,
        'input_edges_per_genome':[4]*128,'output_edges_per_genome':[2]*128,
        'recurrent_edges_per_genome':[0]*128,'total_edges_per_genome':[6]*128}
    rows=[{'generation':i,'phase':'cold' if i<2 else 'warm','seconds':1.0,
        'population_sizes':sizes,'resources':{'process_peak_rss_bytes':1000000,
        'cgroup_memory_peak_bytes':2000000},'device_memory':{'peak_pool_bytes':1000000},
        'metrics':{'custom_metrics':{'step_timings':{'step0':0.1}}}} for i in range(12)]
    (cell/'generations.json').write_text(json.dumps(rows))
    good=audit_cell(tmp_path,manifest,digest,'rtx3070','feedforward',700)
    assert good['status']=='pass'
    rows[0]['population_sizes']={**sizes,'total_edges_per_genome':[7]*128}
    (cell/'generations.json').write_text(json.dumps(rows))
    bad=audit_cell(tmp_path,manifest,digest,'rtx3070','feedforward',700)
    assert bad['status']=='failed'
    assert 'edge-count-sum' in bad['errors']
