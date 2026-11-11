"""Audit monitored M2 G5 memory pressure on both GPU profiles."""
import argparse
import csv
import json
from pathlib import Path


parser=argparse.ArgumentParser()
parser.add_argument('--source',default='/results/m2-g3-telemetry')
parser.add_argument('--telemetry',default='/telemetry')
parser.add_argument('--output',default='/reports/m2-g3-status.json')
args=parser.parse_args()
root=Path(args.source)
telemetry=Path(args.telemetry)
with (telemetry/'gpu.csv').open(encoding='utf-8-sig',newline='') as stream:
    gpu_rows=[{k:v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
vm=[json.loads(line) for line in (telemetry/'current-vm.jsonl').read_text().splitlines() if line.strip()]
assert len(vm)>=30 and len(gpu_rows)>=60
swap_out_delta=vm[-1]['pswpout_pages']-vm[0]['pswpout_pages']
swap_used_start=vm[0]['SwapTotal_bytes']-vm[0]['SwapFree_bytes']
swap_used_end=vm[-1]['SwapTotal_bytes']-vm[-1]['SwapFree_bytes']
vm_status={'samples':len(vm),'available_min_bytes':min(row['MemAvailable_bytes'] for row in vm),
           'swap_out_pages_delta':swap_out_delta,'swap_used_start_bytes':swap_used_start,
           'swap_used_end_bytes':swap_used_end,
           'no_sustained_swap_growth':swap_out_delta==0 and swap_used_end<=swap_used_start}
profiles={}
for profile in ('rtx3070','rtx3060'):
    folder=root/profile
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='measured' and result['hardware']['backend']=='gpu'
    hardware=result['hardware']
    memory=result['memory']
    name=hardware['gpu'][1]
    total=int(hardware['gpu'][2])
    samples=[row for row in gpu_rows if row['name']==name]
    assert len(samples)>=20
    peak_used=max(int(row['used_mib']) for row in samples)
    min_free=min(int(row['free_mib']) for row in samples)
    pool=memory['device_allocator']['peak_pool_bytes']
    passes=(peak_used<=.6*total and min_free>=1024 and
            pool<=.6*total*1024**2 and
            memory['cgroup_peak_bytes']<=6*1024**3 and
            memory['cgroup_swap_bytes']==0 and
            hardware['allocator_fraction']<=.6 and vm_status['no_sustained_swap_growth'])
    profiles[profile]={'samples':len(samples),'physical_used_mib_max':peak_used,
                       'physical_free_mib_min':min_free,'total_mib':total,
                       'allocator_peak_pool_bytes':pool,
                       'cgroup_peak_bytes':memory['cgroup_peak_bytes'],
                       'cgroup_swap_bytes':memory['cgroup_swap_bytes'],
                       'g3_passes':passes,'hardware':hardware}
status={'schema':1,'status':'pass' if all(row['g3_passes'] for row in profiles.values()) else 'fail',
        'scope':'one full monitored 1024-node G5 trial on each GPU; 3-trial allocator/cgroup checks in G5 status',
        'vm':vm_status,'profiles':profiles}
target=Path(args.output)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':status['status'],'physical_used_mib_max':
                  {k:v['physical_used_mib_max'] for k,v in profiles.items()}}))
if status['status']!='pass':
    raise AssertionError('M2 G3 monitored memory check failed')
