"""Audit a completed adaptive five-seed run against its frozen G9 pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from neuroevo.artifacts import canonical, write_json


def read(path):
    return json.loads(Path(path).read_text())


def audit(root):
    root=Path(root)
    pilot=read(root/'pilot.json')
    if not pilot['passes'] or pilot['hardware']['development_mount']:
        raise ValueError('A2 acceptance requires a successful immutable-image G9 pilot')
    experiment=read(root/'experiment.json')
    config=experiment['config']
    if config['kind']!='acceptance' or len(config['seeds'])!=5 or \
            hashlib.sha256(canonical(config)).hexdigest()!=pilot['config_sha256']:
        raise ValueError('A2 acceptance manifest differs from pilot or five-seed protocol')
    rows=[]
    for seed in config['seeds']:
        path=root/f'seed-{seed}'
        manifest,summary,selection,history=(read(path/name) for name in
            ('manifest.json','summary.json','selection.json','history.json'))
        if (manifest['seed']!=seed or manifest['config']!=config or
                manifest['provenance']!=pilot['provenance'] or
                manifest['hardware']!=pilot['hardware'] or
                summary['hardware']!=pilot['hardware'] or
                summary['config_sha256']!=pilot['config_sha256'] or
                len(history)!=config['generations'] or
                summary['generations']!=config['generations'] or
                summary['wall_seconds']>config['budget_seconds_per_seed']):
            raise AssertionError(f'Incomplete or mismatched A2 seed {seed}')
        if any(not 0<r['valid']<=config['adapter']['population'] or
               r['phenotype_nodes_max']>config['adapter']['nodes'] or
               r['phenotype_edges_max']>config['adapter']['edges'] or
               len(r['adaptive_graphs'])!=config['adapter']['population']
               for r in history):
            raise AssertionError(f'Invalid A2 phenotype or fitness record in seed {seed}')
        active={g['active_hidden_nodes'] for r in history for g in r['adaptive_graphs']}
        graphs={g['graph_sha256'] for r in history for g in r['adaptive_graphs']}
        rows.append({
            'seed':seed,'heldout_score':selection['heldout_score'],
            'heldout_valid':selection['heldout_valid'],
            'passes_learning':selection['passes_learning'],
            'validation_score':selection['validation_score'],
            'wall_seconds':summary['wall_seconds'],
            'min_valid_candidates':min(r['valid'] for r in history),
            'active_hidden_count_range':[min(active),max(active)],
            'distinct_weighted_graphs':len(graphs),
            'max_phenotype_nodes':max(r['phenotype_nodes_max'] for r in history),
            'max_phenotype_edges':max(r['phenotype_edges_max'] for r in history),
            'warm_decode_median_seconds':statistics.median(
                r['decode_plan_seconds'] for r in history[2:]),
            'warm_rollout_median_seconds':statistics.median(
                r['rollout_seconds'] for r in history[2:]),
            'warm_evolve_median_seconds':statistics.median(
                r['evolve_seconds'] for r in history[2:]),
        })
    passed=sum(row['passes_learning'] for row in rows)
    if (experiment['passed_seeds']!=passed or not experiment['complete_protocol'] or
            len(experiment['runs'])!=5):
        raise AssertionError('A2 experiment summary differs from seed artifacts')
    return {'schema':1,'status':'research-gate-pass' if passed>=config['required_seeds']
            else 'research-gate-failed','task':config['task'],'seeds':rows,
            'passed_seeds':passed,'required_seeds':config['required_seeds'],
            'pilot_warm_generation_seconds':pilot['warm_generation_seconds'],
            'image_digest':pilot['hardware']['image_digest']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    write_json(args.output,result)
    print(json.dumps({'task':result['task'],'status':result['status'],
                      'passed_seeds':result['passed_seeds']}))
