"""Run the frozen two-hour M2 comparison, one isolated seed at a time.

Restarting this script skips complete seeds and resumes an incomplete seed from
its last checkpoint. It never inspects held-out results to stop evolution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary=stream.name
    os.replace(temporary,path)


parser=argparse.ArgumentParser()
parser.add_argument('--output',default='/results/m2-elapsed-v5')
args=parser.parse_args()
root=Path(args.output)
image_digest=os.environ['NEUROEVO_IMAGE_DIGEST']
cells=(('fixed','cartpole','v3'),('b0','cartpole','v5'),
       ('fixed','memory','v3'),('b0','memory','v4'))
started=time.time()
completed_cells=[]
for method,task,version in cells:
    config_path=Path(f'configs/m2-elapsed-{method}-{task}-acceptance-{version}.json')
    config=json.loads(config_path.read_text())
    digest=hashlib.sha256(canonical(config)).hexdigest()
    cell=root/f'{method}-{task}'
    pilot=json.loads((cell/'pilot.json').read_text())
    assert pilot['passes'] and pilot['config_sha256']==digest
    assert pilot['pilot_budget_rule']=='elapsed-stop-with-finalization-reserve'
    assert pilot['hardware']['image_digest']==image_digest
    assert config['kind']=='acceptance' and config['history_policy']=='jsonl'
    runs=[]
    for seed in config['seeds']:
        directory=cell/f'seed-{seed}'
        summary_path=directory/'summary.json'
        selection_path=directory/'selection.json'
        complete=False
        if summary_path.exists() and selection_path.exists():
            summary=json.loads(summary_path.read_text())
            complete=(0 < summary['generations'] < config['generations'] and
                      summary['stop_reason']=='elapsed' and
                      summary['wall_seconds'] <= config['budget_seconds_per_seed'] and
                      summary['config_sha256']==digest and summary['seed']==seed and
                      summary['hardware']==pilot['hardware'])
        if not complete:
            atomic_json(root/'progress.json',{'status':'running','method':method,
                'task':task,'seed':seed,'started_unix_seconds':started,
                'completed_cells':completed_cells})
            directory.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','neuroevo','run','--config',str(config_path),
                     '--output',str(cell),'--seed',str(seed)]
            if (directory/'checkpoint.bin').exists():
                command.append('--resume')
            print(json.dumps({'event':'start','method':method,'task':task,
                              'seed':seed,'resume':'--resume' in command,
                              'generation_cap':config['generations']}),flush=True)
            with (directory/'console.log').open('a',encoding='utf-8') as log:
                with subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT) as process:
                    code=process.wait()
            if code:
                atomic_json(root/'progress.json',{'status':'failed','method':method,
                    'task':task,'seed':seed,'exit_code':code,
                    'log':str(directory/'console.log')})
                raise RuntimeError(f'Elapsed comparison failed: {method} {task} seed {seed}')
        summary=json.loads(summary_path.read_text())
        selection=json.loads(selection_path.read_text())
        if not (0 < summary['generations'] < config['generations'] and
                summary['stop_reason']=='elapsed' and
                summary['wall_seconds'] <= config['budget_seconds_per_seed'] and
                summary['config_sha256']==digest and summary['seed']==seed and
                summary['hardware']==pilot['hardware']):
            raise RuntimeError(f'Invalid elapsed result: {method} {task} seed {seed}')
        runs.append({**summary,'selection':selection})
        print(json.dumps({'event':'complete','method':method,'task':task,
                          'seed':seed,'generations':summary['generations'],
                          'wall_seconds':summary['wall_seconds'],
                          'heldout_score':selection['heldout_score']}),flush=True)
    atomic_json(cell/'experiment.json',{'config':config,'runs':runs,
        'passed_seeds':sum(r['selection']['passes_learning'] for r in runs),
        'complete_protocol':True})
    completed_cells.append(f'{method}-{task}')
    print(json.dumps({'event':'cell-complete','method':method,'task':task}),flush=True)
atomic_json(root/'progress.json',{'status':'complete','completed_unix_seconds':time.time(),
                                  'started_unix_seconds':started})
print(json.dumps({'event':'all-complete'}),flush=True)
