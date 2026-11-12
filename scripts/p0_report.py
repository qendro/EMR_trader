"""Audit a complete P0 acceptance experiment against frozen paired baselines."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from neuroevo.p0 import paired_quality
from neuroevo.artifacts import canonical


def audit(experiment_path, baseline_path):
    experiment=json.loads(Path(experiment_path).read_text())
    baseline=json.loads(Path(baseline_path).read_text())
    config=experiment['config']
    if (config['task']!='cartpole-partial' or config['kind']!='acceptance' or
            not experiment['complete_protocol'] or baseline['task']!=config['task']):
        raise ValueError('P0 experiment is incomplete or mismatched')
    runs=sorted(experiment['runs'],key=lambda r:r['seed'])
    seeds=[r['seed'] for r in runs]
    if seeds!=sorted(config['seeds']) or len(seeds)!=len(set(seeds)):
        raise ValueError('P0 evolution seed set changed')
    config_hash=hashlib.sha256(canonical(config)).hexdigest()
    pilot=json.loads((Path(experiment_path).parent/'pilot.json').read_text())
    if not pilot['passes'] or pilot['config_sha256']!=config_hash:
        raise ValueError('P0 matching G9 pilot is missing or failed')
    for run in runs:
        directory=Path(experiment_path).parent/f"seed-{run['seed']}"
        manifest=json.loads((directory/'manifest.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        selection=json.loads((directory/'selection.json').read_text())
        history_path=directory/'history.json'
        history=(json.loads(history_path.read_text()) if history_path.exists()
                 else json.loads(gzip.decompress((directory/'history.json.gz').read_bytes())))
        if (manifest['config_sha256']!=config_hash or manifest['config']!=config or
                manifest['provenance']!=pilot['provenance'] or
                manifest['hardware']!=pilot['hardware'] or
                summary['generations']!=config['generations'] or
                len(history)!=config['generations'] or
                summary['stop_reason']!='generation-cap' or
                selection!=run['selection']):
            raise ValueError(f"P0 seed {run['seed']} failed its artifact audit")
    policy=[r['selection']['episode_scores'] for r in runs]
    random=[baseline['seeds'][str(seed)] for seed in seeds]
    result=paired_quality(policy,random,
        scale=config['quality_protocol']['normalization_scale'])
    if result['episodes_per_seed']!=config['heldout_episodes']:
        raise ValueError('P0 held-out episode count changed')
    return {'experiment':config['id'],'config_sha256':config_hash,
        'engine_sha256':pilot['provenance']['engine_sha256'],'seeds':seeds,
        'raw_seed_passes':experiment['passed_seeds'],
        'paired_quality':result,
        'passes':bool(result['passes'] and
                      experiment['passed_seeds']>=config['required_seeds'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(args.experiment,args.baseline),indent=2)+'\n')
