"""Freeze wiring demonstrations; these never substitute for H2 acceptance."""
import json
from pathlib import Path

root=Path('configs')
for task in ('cartpole','memory'):
    source='m1-smoke-v1.json' if task=='cartpole' else 'm1-memory-diagnostic-v1.json'
    c=json.loads((root/source).read_text())
    c.update(id=f'h1-fixed-{task}-smoke-v1',kind='smoke',seeds=[900],generations=4,
             finalists=4,checkpoint_every=2,budget_status='wiring-only-not-H2-acceptance')
    c['adapter'].update(inputs=4,outputs=1,population=8,nodes=32,edges=128,profile='dag')
    c['adapter']['species'].update(species_size=2,species_elitism=1,genome_elitism=1)
    c['decoder']=dict(schema=1,kind='fixed-coordinate-v1',inputs=4 if task=='cartpole' else 2,
        outputs=2 if task=='cartpole' else 1,hidden=8 if task=='cartpole' else 4,
        recurrent=True,threshold=.3,max_weight=5.,query_slab=64)
    (root/f'h1-fixed-{task}-smoke-v1.json').write_text(json.dumps(c,indent=2)+'\n')
