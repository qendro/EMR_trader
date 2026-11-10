"""Summarize persisted evidence without importing a numerical backend."""
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

root=Path('/results')
for p in sorted(root.glob('qualification-*/tests-*.xml')):
    x=ET.parse(p).getroot().find('testsuite')
    print(json.dumps({'path':str(p),'kind':'tests',**x.attrib}))
for p in sorted(root.glob('m1-*/experiment.json')):
    v=json.loads(p.read_text())
    print(json.dumps({'experiment':str(p),'complete':v['complete_protocol'],
        'passed':v['passed_seeds'],'scores':[r['selection']['heldout_score'] for r in v['runs']],
        'wall_seconds':[r['wall_seconds'] for r in v['runs']]}))
for p in sorted(root.glob('m1-*/pilot.json')):
    v=json.loads(p.read_text())
    print(json.dumps({'pilot':str(p),'pass':v['passes'],
        'warm':v['warm_generation_seconds'],'fixed':v['fixed_overhead_seconds'],
        'allowable':v['allowable_generations']}))
for directory in ('q2-v2','q2-complete','q2-native'):
    for p in sorted((root/directory).glob('*/*.json')):
        if p.name not in ('result.json','failure.json'): continue
        v=json.loads(p.read_text())
        print(json.dumps({'q2':str(p),'result':v}))
for name in ('resume-40-100.json','closed-loop-replay-cpu.json'):
    p=root/'qualification-final'/name
    if p.exists(): print(json.dumps({name:json.loads(p.read_text())}))
