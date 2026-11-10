"""Freeze B0 and small-substrate H2 protocols before learning measurements."""
import copy
import json
from pathlib import Path

root=Path('configs')
for task in ('cartpole','memory'):
    source='m1-acceptance-v1.json' if task=='cartpole' else 'm1-memory-diagnostic-v1.json'
    base=json.loads((root/source).read_text())
    base.update(kind='acceptance',seeds=[0,1,2,3,4],budget_status='requires-matching-integrated-G9-pilot')
    hidden=8 if task=='cartpole' else 4
    inputs,outputs=(4,2) if task=='cartpole' else (2,1)
    b=copy.deepcopy(base)
    b.update(id=f'b0-{task}-acceptance-v1',method='fixed-topology-weight-only',
             comparison='layered-system-and-architecture-family-control; no encoding claim from this method alone')
    b['adapter']=dict(schema=1,kind='fixed-weight-v1',population=128,inputs=inputs,hidden=hidden,
        outputs=outputs,profile='layered',recurrent=True,init_std=.5,mutation_std=.15,
        mutation_rate=.2,survival_fraction=.1,elite_count=2,weight_bound=5.)
    (root/f'b0-{task}-acceptance-v1.json').write_text(json.dumps(b,indent=2)+'\n')
    smoke=copy.deepcopy(b)
    smoke.update(id=f'b0-{task}-smoke-v1',kind='smoke',seeds=[900],generations=4,finalists=4,
                 checkpoint_every=2,budget_status='wiring-only-not-baseline-score-protocol')
    smoke['adapter']['population']=8
    if task=='cartpole':
        smoke.update(episodes=2,horizon=64,validation_episodes=4,heldout_episodes=4,microbatch=4)
    (root/f'b0-{task}-smoke-v1.json').write_text(json.dumps(smoke,indent=2)+'\n')
    fixed=copy.deepcopy(base)
    fixed.update(id=f'h2-fixed-{task}-acceptance-v1',method='fixed-coordinate-CPPN',
        protocol_status='frozen-not-run',comparison='layered-system-and-architecture-family-control')
    fixed['adapter'].update(inputs=4,outputs=1,nodes=32,edges=128,profile='dag')
    fixed['decoder']=dict(schema=1,kind='fixed-coordinate-v1',inputs=inputs,hidden=hidden,outputs=outputs,
                         recurrent=True,threshold=.3,max_weight=5.,query_slab=4096)
    (root/f'h2-fixed-{task}-acceptance-v1.json').write_text(json.dumps(fixed,indent=2)+'\n')

direct=json.loads((root/'m1-memory-diagnostic-v1.json').read_text())
direct.update(id='h2-direct-memory-acceptance-v1',kind='acceptance',seeds=[0,1,2,3,4],
              budget_status='requires-matching-integrated-G9-pilot',
              method='direct-recurrent-genome',
              comparison='system-only-for-memory-synchronous-profile')
(root/'h2-direct-memory-acceptance-v1.json').write_text(json.dumps(direct,indent=2)+'\n')
