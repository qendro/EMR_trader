"""Assemble the M4/M5 acceptance ledger from machine-readable audits."""
import argparse
import json
from pathlib import Path


def load(root,name):
    return json.loads((root/name).read_text())


def assemble(root):
    root=Path(root)
    p0=load(root,'p0-comparison-v1.json')
    w6=load(root,'w6-v1-audit.json')
    w7=load(root,'w7-stability-v1-audit.json')
    g5=load(root,'m4-g5-evolved-v1-audit.json')
    g5_followup=load(root,'m4-g5-evolved-v2-audit.json')
    p1=load(root,'p1-crossbackend-v1-audit.json')
    memory=load(root,'a2-memory-acceptance-v1-audit.json')
    cartpole=load(root,'a2-cartpole-acceptance-v1-audit.json')
    gates={
        'a2_memory':memory['passed_seeds']>=memory['required_seeds'],
        'a2_cartpole':cartpole['passed_seeds']>=cartpole['required_seeds'],
        'p0_harder_task':p0['any_method_passes'],
        'w6_both_cards':w6['passes'],
        'w7_stability':w7['passes'],
        'm4_g5_both_cards':all(g5['passes'].values()),
        'p1_crossbackend_replay_resume':p1['passes'],
    }
    return {'gates':gates,'all_listed_gates_pass':all(gates.values()),
            'p0_methods':[{k:row[k] for k in ('method','passes','raw_seed_passes')}
                          for row in p0['methods']],
            'm4_gpu_speedups':g5['gpu_speedup_over_cpu'],
            'm4_followup_gpu_speedups':g5_followup['gpu_speedup_over_cpu'],
            'm4_followup_both_cards_pass':all(g5_followup['passes'].values()),
            'w7_generations':w7['generations'],
            'w7_wall_seconds':w7['wall_seconds']}


def markdown(result):
    names={
        'a2_memory':'A2 adaptive memory learning',
        'a2_cartpole':'A2 adaptive CartPole learning',
        'p0_harder_task':'P0 partially observed control learning',
        'w6_both_cards':'W6 2,048-node fixture on both cards',
        'w7_stability':'W7 100-generation, 15-minute stability',
        'm4_g5_both_cards':'M4 G5 end-to-end speedup on both cards',
        'p1_crossbackend_replay_resume':'P1 cross-backend replay and resume',
    }
    lines=['# M4/M5 acceptance dossier','',
        'This report reads the retained machine-readable audits. A failed '
        'research or performance gate stays open; implementation and passing '
        'tests do not convert it into a pass.','',
        '| Gate | Result |','| --- | --- |']
    for key,passed in result['gates'].items():
        lines.append(f"| {names[key]} | {'pass' if passed else 'open'} |")
    lines.extend(['','## Harder-task methods','',
        '| Method | Research target | Raw 120-step seeds (diagnostic) |',
        '| --- | --- | ---: |'])
    for row in result['p0_methods']:
        lines.append(f"| {row['method']} | {'pass' if row['passes'] else 'open'} | "
                     f"{row['raw_seed_passes']}/5 |")
    lines.extend(['','The [P0 comparison](P0-COMPARISON-REPORT.md) contains '
        'paired scores and bootstrap intervals. The [W6](w6-v1-audit.json), '
        '[W7](w7-stability-v1-audit.json), [G5 v1](m4-g5-evolved-v1-audit.json), '
        '[G5 v2](m4-g5-evolved-v2-audit.json) and '
        '[P1](p1-crossbackend-v1-audit.json) audits retain measured limits. '
        'The [G5 report](M4-G5-REPORT.md) explains the versioned follow-up.','',
        f"All listed gates: {'pass' if result['all_listed_gates_pass'] else 'open'}.",
        'M5 also depends on historical M2/M3 comparison and per-card gates '
        'tracked in the implementation record; this dossier does not close '
        'those by inheritance.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--json-output',required=True)
    parser.add_argument('--markdown-output',required=True)
    args=parser.parse_args()
    result=assemble(args.root)
    Path(args.json_output).write_text(json.dumps(result,indent=2)+'\n')
    Path(args.markdown_output).write_text(markdown(result))
