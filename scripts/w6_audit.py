"""Verify both-card W6 workload dimensions and bounded execution evidence."""
import argparse
import json
from pathlib import Path


def audit(paths, pilots=None):
    cards=[]
    for path in paths:
        result=json.loads(Path(path).read_text())
        cells=result['cells']
        if len(cells)!=2 or [cell['layout'] for cell in cells]!=['edge','dense']:
            raise ValueError('W6 layout pair is incomplete')
        checks=[]
        for cell in cells:
            checks.append(bool(cell['population']==256 and
                cell['active_nodes_per_graph']==2048 and
                cell['episodes']==4 and cell['horizon']==128 and
                cell['carry_steps']==1 and 1<=cell['microbatch']<256 and
                cell['compiled_signatures']==1 and
                cell['new_signatures_after_first_chunk']==0 and
                cell['memory']['cgroup_swap_bytes']==0 and
                cell['warm_chunk_median_seconds']>0))
        checks.append(cells[0]['max_active_edges']<=32768 and
                      cells[0]['min_active_edges']>0 and cells[0]['inputs']==4)
        checks.append(cells[1]['inputs']==400 and cells[1]['min_active_edges']>0)
        checks.append(result['visible_devices']==1 and
                      'GeForce RTX' in result['device_kind'] and
                      bool(result['cuda_visible_devices']))
        cards.append({'path':str(path),'device_kind':result['device_kind'],
            'cuda_visible_devices':result['cuda_visible_devices'],
            'edge_active_edges':cells[0]['max_active_edges'],
            'dense_active_edges':cells[1]['max_active_edges'],
            'edge_warm_chunk_seconds':cells[0]['warm_chunk_median_seconds'],
            'dense_warm_chunk_seconds':cells[1]['warm_chunk_median_seconds'],
            'passes':all(checks)})
    if len({card['cuda_visible_devices'] for card in cards})!=len(cards):
        raise ValueError('W6 cards were not distinct')
    if pilots is not None:
        if len(pilots)!=len(paths):
            raise ValueError('W6 pilot count mismatch')
        for card,pilot_path,full_path in zip(cards,pilots,paths):
            pilot=json.loads(Path(pilot_path).read_text())
            full=json.loads(Path(full_path).read_text())
            valid=bool(pilot['visible_devices']==1 and
                pilot['cuda_visible_devices']==full['cuda_visible_devices'] and
                all(cell['population']==8 and cell['active_nodes_per_graph']==2048 and
                    cell['microbatch']==full_cell['microbatch'] and
                    cell['memory']['cgroup_swap_bytes']==0
                    for cell,full_cell in zip(pilot['cells'],full['cells'])))
            card['pilot_path']=str(pilot_path)
            card['pilot_passes']=valid
            card['passes']=card['passes'] and valid
    return {'cards':cards,'passes':all(card['passes'] for card in cards)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',nargs=2,required=True)
    parser.add_argument('--pilots',nargs=2)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(audit(args.results,args.pilots),indent=2)+'\n')
