"""Verify fresh-process M2 G5 trials and compare integrated throughput."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


parser=argparse.ArgumentParser()
parser.add_argument('--source',default='/results/m2-g5-v1')
parser.add_argument('--output',default='/reports/m2-g5-status.json')
parser.add_argument('--config',default='configs/m2-g5-fixed-1024-v1.json')
parser.add_argument('--profiles',default='cpu,rtx3070,rtx3060')
args=parser.parse_args()
root=Path(args.source)
config=read(Path(args.config))
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
profiles={}
engine=None
image=None
for profile in args.profiles.split(','):
    assert profile in config['profiles']
    rows=[]
    for seed in config['seeds']:
        folder=root/f'{profile}-{seed}'
        manifest=read(folder/'manifest.json')
        result=read(folder/'result.json')
        history=read(folder/'history.json')
        assert manifest['config']==config and manifest['seed']==seed
        assert manifest['config_sha256']==result['config_sha256']==config_hash
        assert result['status']=='measured' and result['seed']==seed
        assert result['hardware']==manifest['hardware'] and result['provenance']==manifest['provenance']
        assert [r['generation'] for r in history]==[0,1]
        assert result['warm_generation']==history[0] and result['measured_generation']==history[1]
        assert result['candidate_episode_evaluations']==1024
        for row in history:
            assert row['valid_candidates']==128 and row['cppn_queries_per_candidate']==1043450
            assert row['recurrent_edges_max']>0
            assert row['total_seconds']>row['decode_plan_seconds']+row['rollout_reduce_seconds']
        assert history[0]['rollout_signatures']==history[1]['rollout_signatures']
        assert history[0]['query_signatures']==history[1]['query_signatures']
        assert result['memory']['cgroup_peak_bytes']<=6*1024**3
        assert result['memory']['cgroup_swap_bytes']==0
        if profile!='cpu':
            total_mib=int(result['hardware']['gpu'][2])
            assert result['hardware']['allocator_fraction']<=.6
            assert result['memory']['device_allocator']['peak_pool_bytes']<=.6*total_mib*1024**2
        engine=engine or result['provenance']['engine_sha256']
        image=image or result['hardware']['image_digest']
        assert engine==result['provenance']['engine_sha256']
        assert image==result['hardware']['image_digest']
        rows.append({'seed':seed,'seconds':history[1]['total_seconds'],
                     'decode_plan_seconds':history[1]['decode_plan_seconds'],
                     'rollout_reduce_seconds':history[1]['rollout_reduce_seconds'],
                     'tell_seconds':history[1]['tell_seconds'],
                     'memory':result['memory']})
    times=[r['seconds'] for r in rows]
    profiles[profile]={'trials':rows,'seconds_median':median(times),
                       'seconds_range':[min(times),max(times)],
                       'throughput_candidate_episodes_per_second':1024/median(times),
                       'g3_recorded_allocator_and_cgroup_passes':True}
cpu=profiles['cpu']['seconds_median']
for profile in ('rtx3070','rtx3060'):
    if profile not in profiles:
        continue
    gpu=profiles[profile]
    gpu['gpu_over_cpu_throughput']=cpu/gpu['seconds_median']
    gpu['g5_passes']=gpu['gpu_over_cpu_throughput']>=config['gate']['gpu_over_cpu_throughput_min']
    gpu['faster_than_cpu_noise_range']=gpu['seconds_range'][1]<profiles['cpu']['seconds_range'][0]
status={'schema':1,'status':'measured','config_sha256':config_hash,
        'engine_sha256':engine,'image_digest':image,'profiles':profiles,
        'gating_card_passes':profiles['rtx3070']['g5_passes'] and profiles['rtx3070']['faster_than_cpu_noise_range']}
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':'pass','gating_card_passes':status['gating_card_passes'],
                  'seconds_median':{p:r['seconds_median'] for p,r in profiles.items()}}))
