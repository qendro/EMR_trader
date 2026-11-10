"""Validate the completed M1 report against raw artifacts; no evolution imports."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
p.add_argument('--source',default='/results')
p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(args.source)
def read(path): return json.loads((root/path).read_text())
config=json.loads(Path('/app/configs/m1-acceptance-v1.json').read_text())
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
engine=None
rows=[]
for profile in ('cpu','rtx3070','rtx3060'):
    q=read(f'qualification-current/q1-{profile}.json')
    assert q['status']=='pass'
    xml=ET.parse(root/f'qualification-current/tests-{profile}.xml').getroot().find('testsuite')
    assert int(xml.get('tests'))==44 and all(int(xml.get(k))==0 for k in ('failures','errors','skipped'))
    e=read(f'm1-current-{profile}/experiment.json')
    assert e['complete_protocol'] and len(e['runs'])==5
    assert [r['seed'] for r in e['runs']]==[0,1,2,3,4]
    assert e['config']==config and e['passed_seeds']>=3
    pilot=read(f'm1-current-{profile}/pilot.json')
    assert pilot['passes'] and pilot['label']=='G9-pilot-not-learning-acceptance'
    for summary in e['runs']:
        seed=summary['seed']
        prefix=f'm1-current-{profile}/seed-{seed}'
        manifest=read(f'{prefix}/manifest.json')
        source=summary['provenance']['engine_sha256']
        engine=engine or source
        assert source==engine==q['provenance']['engine_sha256']
        assert summary['provenance']==pilot['provenance']==manifest['provenance']
        assert summary['hardware']==pilot['hardware']==manifest['hardware']
        assert summary['config_sha256']==pilot['config_sha256']==config_hash
        assert manifest['config']==config and manifest['seed']==seed
        assert summary['generations']==100 and summary['wall_seconds']<=7200
        assert summary['structural_mutation_observed']
        history=read(f'{prefix}/history.json')
        assert [r['generation'] for r in history]==list(range(100))
        assert all(r['valid']==128 and r['nodes_max']<=32 and r['edges_max']<=128 for r in history)
        assert max(r['nodes_max'] for r in history)>6
        selected=read(f'{prefix}/selection.json')
        assert selected==summary['selection']
        assert len(selected['episode_scores'])==20 and selected['heldout_valid']
        assert selected['heldout_score']==sum(selected['episode_scores'])/20
        assert selected['candidate_id'] in selected['finalist_ids']
        assert len(selected['finalist_ids'])==8
        assert all(i.startswith(f'seed-{seed}:99:') for i in selected['finalist_ids'])
        best=selected['finalist_validation_scores'].index(max(selected['finalist_validation_scores']))
        assert selected['candidate_id']==selected['finalist_ids'][best]
        for name in ('checkpoint.bin','champion.npz','champion-genes.npz','evaluated-winner.npz',
                     'evaluated-winner-genes.npz','validation-finalists-genes.npz'):
            assert (root/prefix/name).stat().st_size>0
        rows.append({'profile':profile,'seed':seed,'heldout_score':selected['heldout_score'],
                     'history_generations':len(history)})
for name in ('resume-40-100.json','closed-loop-replay-cpu.json','statistics-cpu.json'):
    assert read(f'qualification-current/{name}')['status']=='pass'
value={'schema':1,'status':'pass','engine_sha256':engine,'config_sha256':config_hash,
       'runs':rows,'checks':'raw manifests, pilots, tests, sequential histories, validation association, artifacts and resume/replay'}
Path(args.output).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':'pass','runs':len(rows),'engine_sha256':engine}))
