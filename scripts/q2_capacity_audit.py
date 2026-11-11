"""Audit the frozen Q2 capacity matrix from copied evidence, without JAX."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


GPU_MIB = {'rtx3070': 8192, 'rtx3060': 12288}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def audit_cell(root, manifest, manifest_hash, profile, mode, seed):
    directory = root/f'{profile}-{mode}-{seed}'
    errors = []
    run = read(directory/'manifest.json')
    launcher = read(directory/'launcher.json')
    result = read(directory/'result.json')
    failure = read(directory/'failure.json') or read(directory/'external-failure.json')
    rows = read(directory/'generations.json')
    dataset = directory/'dataset.npz'
    if not all((run, launcher, result, rows)):
        missing = [name for name, value in (('manifest', run), ('launcher', launcher),
                   ('result', result), ('generations', rows)) if value is None]
        return {'profile': profile, 'mode': mode, 'seed': seed,
                'status': 'failed' if failure else 'incomplete',
                'missing': missing, 'failure': failure}
    if not dataset.exists():
        errors.append('missing-dataset')
    if run.get('manifest') != manifest or run.get('seed') != seed or run.get('mode') != mode:
        errors.append('manifest-or-cell-identity')
    hardware = run.get('hardware', {})
    gpu = hardware.get('gpu') or []
    if (hardware.get('backend') != 'gpu' or len(gpu) < 4 or
            gpu[1] != f'NVIDIA GeForce RTX {profile[-4:]}' or
            int(gpu[2]) != GPU_MIB[profile] or
            hardware.get('image_digest') != launcher.get('image_digest')):
        errors.append('GPU-or-image-identity')
    if (launcher.get('manifest_sha256') != manifest_hash or
            launcher.get('profile') != profile or launcher.get('mode') != mode or
            launcher.get('seed') != seed):
        errors.append('launcher-identity')
    if (launcher.get('exit_code') != 0 or launcher.get('oom_killed') or
            launcher.get('external_timeout') or not launcher.get('cold_generations_observed') or failure):
        errors.append('process-failure-or-timeout')
    expected = manifest['cold_generations']+manifest['warm_generations']
    if (len(rows) != expected or [r.get('generation') for r in rows] != list(range(expected)) or
            [r.get('phase') for r in rows] !=
            ['cold']*manifest['cold_generations']+['warm']*manifest['warm_generations']):
        errors.append('generation-protocol')
    distributions = {'active_nodes': [], 'input_edges': [], 'output_edges': [],
                     'recurrent_edges': [], 'total_edges': []}
    process_peaks, cgroup_peaks, pool_peaks = [], [], []
    for row in rows:
        sizes = row.get('population_sizes') or {}
        fields = {key: sizes.get(f'{key}_per_genome') for key in distributions}
        if (sizes.get('candidate_positions') != 1364 or
                any(not isinstance(values, list) or len(values) != manifest['population']
                    for values in fields.values())):
            errors.append('missing-or-wrong-population-distribution')
            continue
        if any(any(type(v) is not int or v < 0 for v in values) for values in fields.values()):
            errors.append('invalid-population-count')
            continue
        if any(v > 1364 for v in fields['active_nodes']):
            errors.append('active-node-count-exceeds-grid')
        if any(v > manifest['recurrence']['max_sparse_conns']
               for v in fields['recurrent_edges']):
            errors.append('recurrent-connection-cap-exceeded')
        if any(total != wi+wo+hh for total, wi, wo, hh in zip(
                fields['total_edges'], fields['input_edges'],
                fields['output_edges'], fields['recurrent_edges'])):
            errors.append('edge-count-sum')
        if mode == 'feedforward' and any(fields['recurrent_edges']):
            errors.append('feedforward-has-recurrent-edges')
        custom=(row.get('metrics') or {}).get('custom_metrics') or {}
        if (custom.get('hh_truncated_genomes',0) or
                ('total_hh_connections' in custom and
                 custom['total_hh_connections'] != sum(fields['recurrent_edges']))):
            errors.append('upstream-recurrent-count-or-truncation')
        if ('min_hidden_nodes' in custom and
                (custom['min_hidden_nodes'] != min(fields['active_nodes']) or
                 custom.get('max_hidden_nodes') != max(fields['active_nodes']))):
            errors.append('upstream-active-node-count')
        for key, values in fields.items():
            distributions[key].extend(values)
        resource = row.get('resources') or {}
        memory = row.get('device_memory') or {}
        for key, target in (('process_peak_rss_bytes', process_peaks),
                            ('cgroup_memory_peak_bytes', cgroup_peaks)):
            value = resource.get(key)
            if type(value) is not int:
                errors.append(f'missing-{key}')
            else:
                target.append(value)
        pool = memory.get('peak_pool_bytes')
        if type(pool) is not int:
            errors.append('missing-device-pool-peak')
        else:
            pool_peaks.append(pool)
        if not custom.get('step_timings'):
            errors.append('missing-upstream-stage-dispatch-times')
    warm = [r['seconds'] for r in rows if r.get('phase') == 'warm' and 'seconds' in r]
    median_warm = statistics.median(warm) if warm else None
    if (median_warm is None or median_warm > manifest['maximum_warm_median_seconds'] or
            not result.get('timing_pass')):
        errors.append('warm-timing-ceiling')
    if result.get('cold_initialize_generation_seconds', float('inf')) > manifest['maximum_cold_seconds']:
        errors.append('cold-timing-ceiling')
    host_budget = manifest['host_memory_budget_bytes']
    if (not process_peaks or not cgroup_peaks or
            max(process_peaks, default=0) > host_budget or
            max(cgroup_peaks, default=0) > host_budget):
        errors.append('host-memory-budget')
    physical_peak = launcher.get('physical_gpu_memory_used_mib_peak')
    if (not pool_peaks or max(pool_peaks, default=0) <= 0 or max(pool_peaks, default=0) >
            manifest['vram_fraction']*GPU_MIB[profile]*1024*1024 or
            type(physical_peak) is not int or
            GPU_MIB[profile]-physical_peak < manifest['free_vram_reserve_bytes']/1048576):
        errors.append('GPU-memory-budget-or-monitor')
    summary = {key: {'min': min(values), 'median': statistics.median(values),
                     'max': max(values)} for key, values in distributions.items() if values}
    return {'profile': profile, 'mode': mode, 'seed': seed,
            'status': 'pass' if not errors else 'failed',
            'errors': sorted(set(errors)), 'warm_median_seconds': median_warm,
            'cold_seconds': result.get('cold_initialize_generation_seconds'),
            'peak_process_rss_bytes': max(process_peaks, default=None),
            'peak_cgroup_memory_bytes': max(cgroup_peaks, default=None),
            'peak_device_pool_bytes': max(pool_peaks, default=None),
            'peak_physical_GPU_used_mib': physical_peak,
            'population_distribution': summary,
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.exists() else None,
            'failure': failure}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='benchmarks/results/evidence/q2-capacity-v2')
    p.add_argument('--manifest', default='configs/q2-capacity-v2.json')
    p.add_argument('--output', default='benchmarks/q2-capacity-v2-audit.json')
    args = p.parse_args()
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    cells = [audit_cell(Path(args.source), manifest, hashlib.sha256(manifest_bytes).hexdigest(),
                        profile, mode, seed)
             for profile in GPU_MIB for mode in manifest['modes'] for seed in manifest['seeds']]
    hashes={c.get('dataset_sha256') for c in cells if c['status']=='pass'}
    if len(hashes)>1:
        for cell in cells:
            if cell['status']=='pass':
                cell['status']='failed'
                cell['errors'].append('dataset-bytes-differ-across-cells')
    status = ('failed' if any(c['status'] == 'failed' for c in cells) else
              'incomplete' if any(c['status'] == 'incomplete' for c in cells) else
              'capacity-and-resource-screen-pass-stage-audit-open')
    report = {'schema': 1, 'status': status, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'stage_timing_status': 'unsynchronized-upstream-dispatch-times; isolated-stage-probe-open',
              'cells': cells}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': status, 'passed': sum(c['status']=='pass' for c in cells),
                      'total': len(cells)}))


if __name__ == '__main__':
    main()
