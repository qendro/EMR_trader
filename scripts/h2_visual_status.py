"""Audit every visual diagnostic run and its archived champion."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from neuroevo.backend import initialize
initialize()
import jax.numpy as jnp
import numpy as np
from neuroevo.adapters.cppn import CPPNQuery
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.adapters.weights import WeightAdapter
from neuroevo.artifacts import canonical, reload
from neuroevo.contracts import stack_plans
from neuroevo.evaluation import visual_chunk
from neuroevo.fixed import FixedDecoder
from neuroevo.planning import plan
from neuroevo.visual import visual_scenarios


def read(path):
    return json.loads(path.read_text())


def spread(values):
    return {'median':median(values),'range':[min(values),max(values)]}


def audit(root, method):
    folder=root/f'rtx3070-{method}'
    experiment=read(folder/'experiment.json')
    config=experiment['config']
    digest=hashlib.sha256(canonical(config)).hexdigest()
    assert config['id']==f'h2-visual-{method}-diagnostic-v1'
    assert experiment['complete_protocol'] and len(experiment['runs'])==3
    assert [r['seed'] for r in experiment['runs']]==config['seeds']==[700,701,702]
    adapter=(WeightAdapter(config['adapter']) if method=='b0' else TensorNEATAdapter(config['adapter']))
    decoder=FixedDecoder(CPPNQuery(adapter),config['decoder']) if method=='fixed' else None
    scores,walls,valids,positions=[],[],[],[]
    provenance=None
    for run in experiment['runs']:
        seed=run['seed']
        directory=folder/f'seed-{seed}'
        manifest=read(directory/'manifest.json')
        selection=read(directory/'selection.json')
        history=read(directory/'history.json')
        assert manifest['config_sha256']==run['config_sha256']==digest
        assert manifest['config']==config and manifest['seed']==seed
        assert run['hardware']==manifest['hardware']
        assert run['provenance']==manifest['provenance']
        provenance=provenance or run['provenance']
        assert provenance==run['provenance']
        assert run['generations']==config['generations']==60
        assert [r['generation'] for r in history]==list(range(60))
        assert selection==run['selection'] and selection['heldout_valid']
        assert selection['candidate_id'] in selection['finalist_ids']
        assert len(selection['episode_scores'])==config['heldout_episodes']==160
        assert (directory/'checkpoint.bin').stat().st_size>0
        p,meta=reload(directory/'champion.npz')
        assert meta['candidate_id']==selection['candidate_id']
        with np.load(directory/'champion-genes.npz',allow_pickle=False) as genes:
            if method=='b0': rebuilt=adapter.decode(genes['weights'])
            elif method=='fixed': rebuilt=decoder.decode(genes['nodes'],genes['conns'])
            else: rebuilt=plan(adapter.decode(genes['nodes'],genes['conns']),
                               config['adapter']['nodes'],config['adapter']['edges'])
        for actual,expected in zip(p,rebuilt):
            np.testing.assert_array_equal(actual,expected)
        obs,labels,sites=visual_scenarios(seed,'interpolation',0,160)
        correct,valid=visual_chunk(stack_plans([p]),obs,labels,profile=meta['profile'])
        assert bool(valid[0])
        np.testing.assert_array_equal(np.asarray(correct[0]),selection['episode_scores'])
        assert np.isclose(selection['heldout_score'],np.mean(selection['episode_scores']))
        assert all(row['valid']>0 for row in history)
        scores.append(selection['heldout_score'])
        walls.append(run['wall_seconds'])
        valids.extend(row['valid'] for row in history)
        positions.append([float(np.asarray(correct[0])[np.asarray(sites)==i].mean()) for i in range(4)])
    assert experiment['passed_seeds']==sum(s>=config['target'] for s in scores)
    return {'config_sha256':digest,'engine_sha256':provenance['engine_sha256'],
            'image_digest':experiment['runs'][0]['hardware']['image_digest'],
            'seed_scores':scores,'heldout_score':spread(scores),'wall_seconds':spread(walls),
            'heldout_by_position':positions,'minimum_valid_candidates':min(valids),
            'champion_reconstruction_and_replay':'pass'}


parser=argparse.ArgumentParser()
parser.add_argument('--source',default='/results/h2-visual-diagnostic-v1')
parser.add_argument('--output',default='/reports/h2-visual-status.json')
args=parser.parse_args()
cells={method:audit(Path(args.source),method) for method in ('b0','direct','fixed')}
assert len({v['engine_sha256'] for v in cells.values()})==1
assert len({v['image_digest'] for v in cells.values()})==1
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps({'schema':1,'status':'measured-ungated','methods':cells},indent=2)+'\n')
print(json.dumps({'status':'pass','scores':{k:v['seed_scores'] for k,v in cells.items()}}))
