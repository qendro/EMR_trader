"""Verify Q2 system baselines use identical persisted FP32 sample arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--source',default='/results')
p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(args.source)
protocol=script_hash=None
for profile in ('cpu','rtx3070','rtx3060'):
    for mode in ('direct','fixed'):
        for seed in (700,701,702):
            cell=root/'q2-native-v2'/f'{profile}-{mode}-{seed}'
            result=json.loads((cell/'result.json').read_text())
            manifest=json.loads((cell/'manifest.json').read_text())
            runtime=json.loads((cell/'runtime.json').read_text())
            protocol=protocol or manifest['workload']
            script_hash=script_hash or manifest['script_sha256']
            assert manifest['workload']==protocol and manifest['script_sha256']==script_hash
            assert manifest['seed']==seed and manifest['mode']==mode
            assert runtime['cache_entries_at_start']==0 if seed==700 else runtime['cache_entries_at_start']>0
            assert result['status']=='measured' and result['manifest_id']=='q2-native-direct-fixed-v2'
            history=json.loads((cell/'generations.json').read_text())
            assert [r['generation'] for r in history]==list(range(12))
            assert sum(r['phase']=='warm' for r in history)==10
            assert all(r['valid_candidates']+r['invalid_candidates']==128 and r['valid_candidates']>0 for r in history)
            assert result['invalid_candidate_evaluations']==sum(r['invalid_candidates'] for r in history)
datasets=[]
for directory in ('q2-v2','q2-complete','q2-native-v1','q2-native-v2'):
    datasets.extend((root/directory).glob('*/dataset.npz'))
reference=None
rows=[]
for path in sorted(datasets):
    with np.load(path,allow_pickle=False) as data:
        values={name:data[name] for name in ('inputs','targets')}
    assert values['inputs'].shape==(64,4) and values['targets'].shape==(64,2)
    assert all(a.dtype==np.float32 for a in values.values())
    reference=reference or values
    for name in values:
        np.testing.assert_array_equal(values[name],reference[name])
    rows.append({'path':str(path.relative_to(root)),
                 'arrays_sha256':hashlib.sha256(values['inputs'].tobytes()+values['targets'].tobytes()).hexdigest()})
assert len(rows)>=30,'Complete both-card native and EMR cells before the final audit'
Path(args.output).write_text(json.dumps({'status':'pass','datasets':rows,
    'native_v2_cells':18,'native_v2_script_sha256':script_hash,
    'native_v2_protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'native_v2_histories_and_cache_policy':'pass'},indent=2)+'\n')
print(json.dumps({'status':'pass','datasets':len(rows),'arrays_sha256':rows[0]['arrays_sha256']}))
