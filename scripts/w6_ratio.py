"""Compare same-shape W6 warm chunk latency on CPU and each GPU."""
import argparse
import json
from pathlib import Path


def compare(cpu_path,gpu_paths):
    cpu=json.loads(Path(cpu_path).read_text())
    gpu=[json.loads(Path(path).read_text()) for path in gpu_paths]
    if cpu['device_kind'].lower().find('cpu')<0:
        raise ValueError('W6 CPU probe did not select CPU')
    rows=[]
    for result in gpu:
        if result['visible_devices']!=1 or 'GeForce RTX' not in result['device_kind']:
            raise ValueError('W6 GPU probe did not select one card')
        for left,right in zip(cpu['cells'],result['cells']):
            keys=('layout','active_nodes_per_graph','episodes','horizon',
                  'carry_steps','microbatch','min_active_edges','max_active_edges')
            if any(left[key]!=right[key] for key in keys):
                raise ValueError('W6 CPU/GPU workload mismatch')
            rows.append({'device_kind':result['device_kind'],'layout':right['layout'],
                'cpu_warm_chunk_seconds':left['warm_chunk_median_seconds'],
                'gpu_warm_chunk_seconds':right['warm_chunk_median_seconds'],
                'gpu_speedup_over_cpu':left['warm_chunk_median_seconds']/
                                       right['warm_chunk_median_seconds']})
    return {'label':'W6-stage-only-not-M4-G5','rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',required=True)
    parser.add_argument('--gpu',nargs=2,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(compare(args.cpu,args.gpu),indent=2)+'\n')
