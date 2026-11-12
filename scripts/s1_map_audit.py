"""Audit the S1 size/degree/episode crossover map on CPU and both GPUs."""
import argparse
import json
from pathlib import Path
import statistics


def audit(paths):
    profiles={}
    for label,path in paths.items():
        result=json.loads(Path(path).read_text())
        cells=result['cells']
        signatures={(c['hidden'],c['degree_distribution'],c['episodes']) for c in cells}
        expected={(h,d,b) for h in (20,128,512)
                  for d in ('uniform','hub','local') for b in (1,4,16)}
        if signatures!=expected or len(cells)!=27 or any(c['max_abs_error']>2e-4
                                                 for c in cells):
            raise ValueError(f'S1 crossover map incomplete or unequal on {label}')
        if label=='cpu':
            if 'cpu' not in result['device_kind'].lower():
                raise ValueError('S1 CPU map selected the wrong backend')
        elif result['visible_devices']!=1 or label[-4:] not in result['device_kind']:
            raise ValueError('S1 GPU map selected the wrong card')
        profiles[label]=result
    medians={}
    for label,result in profiles.items():
        medians[label]={str(h):float(statistics.median(
            c['edge_speedup'] for c in result['cells'] if c['hidden']==h))
            for h in (20,128,512)}
    return {'profiles':{name:{'device_kind':row['device_kind'],
                         'cells':len(row['cells'])} for name,row in profiles.items()},
            'median_edge_speedup_by_hidden':medians,
            'max_abs_error':max(c['max_abs_error'] for result in profiles.values()
                                for c in result['cells']),
            'map_complete':True,'production_default':'dense',
            'decision':'large synthetic stage benefits do not establish large evolved-generation G8 adoption'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',required=True)
    parser.add_argument('--rtx3070',required=True)
    parser.add_argument('--rtx3060',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit({'cpu':args.cpu,'rtx3070':args.rtx3070,'rtx3060':args.rtx3060})
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
