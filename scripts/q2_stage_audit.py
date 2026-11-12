"""Audit separately synchronized Q2 stage probes against the capacity runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


STAGES = ('step0_cppn_ask_transform', 'step1_cppn_queries',
          'step2_variance_masks', 'step3_phase2_discovery',
          'step4_build_matrices', 'step5_evaluation', 'step6_neat_evolution')
# Timer call sites in pinned emr_hyperneat/engine/generation_mixin.py.
BOUNDARIES = {
    'feedforward': (1065, 710, 713, 718, 724, 734, 755, 760, 762),
    'recurrent': (1065, 1076, 1081, 1117, 1195, 1198, 1210, 1213,
                  1301, 1352, 1392, 1548, 2002, 2014, 2016),
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def p95(values):
    values = sorted(values)
    point = .95*(len(values)-1)
    lo = int(point)
    return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(point-lo)


def audit_cell(root, capacity_root, manifest, manifest_hash, profile, mode, seed):
    name = f'{profile}-{mode}-{seed}'
    path = root/name
    run, launcher, result, rows = (read(path/file) for file in
                                   ('manifest.json','launcher.json','result.json','generations.json'))
    original = read(capacity_root/name/'generations.json')
    reference_dataset = capacity_root/name/'dataset.npz'
    if any(value is None for value in (run, launcher, result, rows, original)):
        return {'cell': name, 'status': 'incomplete'}
    errors = []
    if (run.get('manifest') != manifest or run.get('seed') != seed or
            run.get('mode') != mode or run.get('label') != 'Q2-stage-probe' or
            launcher.get('stage_probe') is not True or
            launcher.get('profile') != profile or launcher.get('mode') != mode or
            launcher.get('seed') != seed or
            launcher.get('manifest_sha256') != manifest_hash or
            run.get('hardware',{}).get('image_digest') != launcher.get('image_digest') or
            launcher.get('exit_code') != 0 or launcher.get('oom_killed') or
            launcher.get('external_timeout') or
            not launcher.get('cold_generations_observed') or
            result.get('stage_gate') != 'synchronized-probe-measured-pending-audit' or
            result.get('cold_initialize_generation_seconds',float('inf')) >
            manifest['maximum_cold_seconds'] or not result.get('timing_pass') or
            (path/'failure.json').exists() or (path/'external-failure.json').exists()):
        errors.append('identity-or-process-failure')
    expected = manifest['cold_generations']+manifest['warm_generations']
    if (len(rows) != expected or len(original) != expected or
            [r.get('generation') for r in rows] != list(range(expected)) or
            [r.get('phase') for r in rows] !=
            ['cold']*manifest['cold_generations']+['warm']*manifest['warm_generations']):
        errors.append('generation-protocol')
    samples = {stage: [] for stage in STAGES}
    residuals = []
    for row, reference in zip(rows, original):
        if row.get('population_sizes') != reference.get('population_sizes'):
            errors.append('population-differs-from-capacity-run')
        for key in ('best_fitness','max_fitness','mean_fitness','min_fitness',
                    'num_species','hidden_nodes_mean','connections_mean'):
            if row.get('metrics',{}).get(key) != reference.get('metrics',{}).get(key):
                errors.append('fitness-or-topology-differs-from-capacity-run')
                break
        records = row.get('stage_sync_records') or []
        expected_functions = (('run_generation_verbose',)+
                              ('_feedforward_generation',)*(len(BOUNDARIES[mode])-1)
                              if mode == 'feedforward' else
                              ('run_generation_verbose',)*len(BOUNDARIES[mode]))
        if (row.get('stage_timing_status') != 'synchronized-live-array-boundaries' or
                tuple(record.get('line') for record in records) != BOUNDARIES[mode] or
                tuple(record.get('function') for record in records) != expected_functions or
                any(type(record.get('arrays_waited')) is not int or
                    record['arrays_waited'] <= 0 or
                    not math.isfinite(record.get('barrier_seconds',float('nan'))) or
                    record['barrier_seconds'] < 0 for record in records)):
            errors.append('stage-boundary-or-barrier-record')
        times = (row.get('metrics') or {}).get('custom_metrics',{}).get('step_timings') or {}
        if any(type(times.get(stage)) not in (int,float) or
               not math.isfinite(times[stage]) or times[stage] < 0 for stage in STAGES):
            errors.append('stage-time-missing-or-invalid')
            continue
        if mode == 'feedforward' and times['step3_phase2_discovery'] != 0:
            errors.append('feedforward-discovery-time')
        if row.get('phase') == 'warm':
            for stage in STAGES:
                samples[stage].append(times[stage])
            residuals.append(row['seconds']-sum(times[stage] for stage in STAGES))
    if any(len(values) != manifest['warm_generations'] for values in samples.values()):
        errors.append('warm-stage-sample-count')
    if any(value < -0.01 for value in residuals):
        errors.append('stage-sum-exceeds-generation')
    dataset = path/'dataset.npz'
    if not dataset.exists() or not reference_dataset.exists():
        errors.append('missing-dataset')
    elif hashlib.sha256(dataset.read_bytes()).digest() != \
            hashlib.sha256(reference_dataset.read_bytes()).digest():
        errors.append('dataset-differs-from-capacity-run')
    return {'cell': name, 'status': 'pass' if not errors else 'failed',
            'errors': sorted(set(errors)),
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.exists() else None,
            'stage_seconds': {stage: {'median': statistics.median(values), 'p95': p95(values)}
                              for stage, values in samples.items() if values},
            'unattributed_generation_seconds_median':
                statistics.median(residuals) if residuals else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='benchmarks/results/evidence/q2-stage-v1')
    parser.add_argument('--capacity-source', default='benchmarks/results/evidence/q2-capacity-v2')
    parser.add_argument('--manifest', default='configs/q2-capacity-v2.json')
    parser.add_argument('--output', default='benchmarks/q2-stage-v1-audit.json')
    args = parser.parse_args()
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    cells = [audit_cell(Path(args.source),Path(args.capacity_source),manifest,manifest_hash,profile,mode,seed)
             for profile in ('rtx3070','rtx3060') for mode in manifest['modes']
             for seed in manifest['seeds']]
    hashes = {cell.get('dataset_sha256') for cell in cells if cell['status']=='pass'}
    if len(hashes) != 1:
        for cell in cells:
            if cell['status'] == 'pass':
                cell['status'] = 'failed'
                cell['errors'].append('dataset-hash-mismatch')
    status = ('failed' if any(cell['status']=='failed' for cell in cells) else
              'incomplete' if any(cell['status']=='incomplete' for cell in cells) else
              'synchronized-stage-probe-pass')
    report = {'schema':1,'status':status,
              'manifest_sha256':manifest_hash,
              'cells':cells}
    Path(args.output).write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({'status':status,'passed':sum(c['status']=='pass' for c in cells),
                      'total':len(cells)}))


if __name__ == '__main__':
    main()
