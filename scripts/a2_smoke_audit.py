"""Audit four bounded A2 development cells without treating them as acceptance."""
import argparse
import json
from pathlib import Path
import statistics

from neuroevo.artifacts import write_json


def cell(path):
    p=Path(path)
    manifest=json.loads((p/'manifest.json').read_text())
    history=json.loads((p/'history.json').read_text())
    summary=json.loads((p/'summary.json').read_text())
    selection=json.loads((p/'selection.json').read_text())
    if len(history)!=3 or summary['generations']!=3 or any(
            len(r['adaptive_graphs'])!=manifest['config']['adapter']['population'] or
            r['valid']!=manifest['config']['adapter']['population'] for r in history):
        raise AssertionError(f'Incomplete A2 smoke cell: {p}')
    graphs=[g for row in history for g in row['adaptive_graphs']]
    return {'manifest':manifest,'history':history,'summary':summary,'selection':selection,
            'active_counts':sorted(set(g['active_hidden_nodes'] for g in graphs)),
            'edge_counts':sorted(set(g['expressed_edges'] for g in graphs)),
            'distinct_graphs':len(set(g['graph_sha256'] for g in graphs)),
            'query_count':sum(r['cppn_queries_total'] for r in history),
            'median_warm_decode_seconds':statistics.median(r['decode_plan_seconds'] for r in history[1:]),
            'median_warm_rollout_seconds':statistics.median(r['rollout_seconds'] for r in history[1:])}


def audit(paths):
    cells={name:cell(path) for name,path in paths.items()}
    rows={}
    for task in ('memory','cartpole'):
        cpu,gpu=(cells[f'{task}-{profile}'] for profile in ('cpu','rtx3070'))
        if cpu['manifest']['config']!=gpu['manifest']['config'] or \
                cpu['manifest']['provenance']!=gpu['manifest']['provenance']:
            raise AssertionError(f'{task}: different config or source')
        if cpu['summary']['hardware']['backend']!='cpu' or \
                gpu['summary']['hardware']['gpu'][1]!='NVIDIA GeForce RTX 3070':
            raise AssertionError(f'{task}: backend identity mismatch')
        for field in ('active_counts','edge_counts','distinct_graphs','query_count'):
            if cpu[field]!=gpu[field]:
                raise AssertionError(f'{task}: graph distribution differs on GPU: {field}')
        if [[r[k] for k in ('best','mean','valid')] for r in cpu['history']] != \
                [[r[k] for k in ('best','mean','valid')] for r in gpu['history']]:
            raise AssertionError(f'{task}: quality history differs on GPU')
        def topology(history):
            return [[{k:v for k,v in graph.items() if k!='graph_sha256'}
                     for graph in row['adaptive_graphs']] for row in history]
        if topology(cpu['history'])!=topology(gpu['history']):
            raise AssertionError(f'{task}: per-genome topology counts differ on GPU')
        if cpu['selection']!=gpu['selection']:
            raise AssertionError(f'{task}: final selection differs on GPU')
        if len(cpu['active_counts'])<2 or cpu['distinct_graphs']<2:
            raise AssertionError(f'{task}: adaptive graph variation was not observed')
        rows[task]={
            'active_hidden_counts':cpu['active_counts'],
            'expressed_edge_counts':cpu['edge_counts'],
            'distinct_graphs':cpu['distinct_graphs'],
            'exact_weighted_graph_digests_match':all(
                c['graph_sha256']==g['graph_sha256']
                for cr,gr in zip(cpu['history'],gpu['history'])
                for c,g in zip(cr['adaptive_graphs'],gr['adaptive_graphs'])),
            'total_cppn_queries':cpu['query_count'],
            'training_best_by_generation':[r['best'] for r in cpu['history']],
            'heldout_score':cpu['selection']['heldout_score'],
            'heldout_valid':cpu['selection']['heldout_valid'],
            'cpu_warm_decode_median_seconds':cpu['median_warm_decode_seconds'],
            'gpu_warm_decode_median_seconds':gpu['median_warm_decode_seconds'],
            'cpu_warm_rollout_median_seconds':cpu['median_warm_rollout_seconds'],
            'gpu_warm_rollout_median_seconds':gpu['median_warm_rollout_seconds'],
        }
    return {'schema':1,'status':'development-smoke-pass','cells':4,
            'backend_parity':'exact-quality-selection-and-per-genome-topology-counts',
            'acceptance':False,'tasks':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('/results'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    names={'memory-cpu':'a2-memory-cpu-final-v2',
           'memory-rtx3070':'a2-memory-gpu-smoke-v2',
           'cartpole-cpu':'a2-cartpole-cpu-final-v1',
           'cartpole-rtx3070':'a2-cartpole-gpu-smoke-v1'}
    report=audit({name:args.results/directory/'seed-900' for name,directory in names.items()})
    write_json(args.output,report)
    print(json.dumps(report,indent=2))
