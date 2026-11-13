"""Audit portable replay and container-replacement continuation evidence."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def audit(root, version='v1'):
    root=Path(root)
    if version not in ('v1','v2'):
        raise ValueError('Unsupported P1 audit version')
    tests={}
    for profile in ('cpu','rtx3070','rtx3060'):
        test_file=(f'results/p1-tests-{profile}.xml' if version=='v1'
                   else f'results/p1-v2-tests-{profile}.xml')
        document=ET.parse(root/test_file)
        suite=document.getroot().find('testsuite')
        if suite is None:
            suite=document.getroot()
        tests[profile]={name:int(suite.attrib.get(name,0))
                        for name in ('tests','failures','errors','skipped')}
    tests_pass=all(row['tests']>=116 and row['failures']==0 and row['errors']==0
                   for row in tests.values())
    replay={profile:json.loads((root/f'p1-replay-{profile}-{version}.json').read_text())
            for profile in ('cpu','rtx3070','rtx3060')}
    actions=[row['actions'] for row in replay.values()]
    same_actions=all(sequence==actions[0] for sequence in actions)
    replay_pass=all(row['status']=='pass' and row['steps']==128 and
                    row['max_abs_error']<=1e-4 for row in replay.values())
    directory=root/f'results/p1-resume-{version}/seed-900'
    summary=json.loads((directory/'summary.json').read_text())
    g5_version='v1' if version=='v1' else 'v3'
    g5=json.loads((root/f'm4-g5-evolved-{g5_version}-audit.json').read_text())
    engine_sha256=summary['provenance']['engine_sha256']
    history=json.loads((directory/'history.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    resume_pass=bool(summary['generations']==3 and len(history)==3 and
                     [row['generation'] for row in history]==[0,1,2] and
                     selection['heldout_valid'] and
                     engine_sha256==g5['engine_sha256'])
    return {'replay_max_abs_error':{name:row['max_abs_error']
                                   for name,row in replay.items()},
            'regression_tests':tests,
            'replay_actions_identical':same_actions,
            'engine_sha256':engine_sha256,
            'resume_generations':summary['generations'],
            'resume_history_rows':len(history),
            'passes':bool(tests_pass and replay_pass and same_actions and resume_pass)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmarks-root',required=True)
    parser.add_argument('--version',default='v1')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(args.benchmarks_root,args.version),indent=2)+'\n')
