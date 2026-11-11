"""Freeze two-hour comparison generation caps from non-acceptance G9 calibration pilots."""
import argparse
import hashlib
import json
import math
from pathlib import Path


parser=argparse.ArgumentParser()
parser.add_argument('--pilots',default='/results/m2-elapsed-calibration-v1')
parser.add_argument('--configs',default='/configs-out')
parser.add_argument('--report',default='/reports/m2-elapsed-freeze.json')
args=parser.parse_args()
pilots=Path(args.pilots)
output=Path(args.configs)
output.mkdir(parents=True,exist_ok=True)
rows=[]
for method in ('b0','fixed'):
    for task in ('cartpole','memory'):
        calibration=json.loads(Path(f'configs/m2-elapsed-{method}-{task}-calibration-v1.json').read_text())
        pilot_path=pilots/f'{method}-{task}'/'pilot.json'
        pilot_bytes=pilot_path.read_bytes()
        pilot=json.loads(pilot_bytes)
        assert pilot['passes'] and pilot['pilot_seed']==900
        assert pilot['config_sha256']==hashlib.sha256(json.dumps(
            calibration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        allowable=pilot['allowable_generations']
        # Retain the G9 ten-percent reserve, then another five percent for
        # long-run checkpoint/history costs absent from a 12-generation pilot.
        generations=math.floor(.95*allowable/100)*100
        assert 0<generations<allowable
        config={**calibration,
            'id':f'm2-elapsed-{method}-{task}-acceptance-v1',
            'kind':'acceptance','seeds':[0,1,2,3,4],
            'generations':generations,'required_seeds':3,
            'budget_status':'requires-matching-integrated-G9-pilot',
            'comparison':'two-hour G9 elapsed-budget quality; separate from fixed-evaluation comparison',
            'calibration':{'pilot_sha256':hashlib.sha256(pilot_bytes).hexdigest(),
                'pilot_image_digest':pilot['hardware']['image_digest'],
                'warm_generation_seconds':pilot['warm_generation_seconds'],
                'fixed_overhead_seconds':pilot['fixed_overhead_seconds'],
                'allowable_generations':allowable,'cap_fraction_of_allowable':.95,
                'tuning_trials':0}}
        path=output/f'm2-elapsed-{method}-{task}-acceptance-v1.json'
        path.write_text(json.dumps(config,indent=2)+'\n')
        rows.append({'method':method,'task':task,'generations':generations,
                     'allowable_generations':allowable,
                     'expected_training_candidate_episodes':generations*config['adapter']['population']*config['episodes'],
                     'config_sha256':hashlib.sha256(json.dumps(
                         config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()})
target=Path(args.report)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps({'schema':1,'status':'frozen-before-acceptance','rows':rows},indent=2)+'\n')
print(json.dumps({'status':'frozen','caps':{f"{r['method']}-{r['task']}":r['generations'] for r in rows}}))
