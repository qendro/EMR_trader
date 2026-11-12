"""Run the frozen Q2 capacity matrix after competing experiments finish.

The launcher enforces a cold-phase deadline outside the JAX process and keeps
one fresh process and cache per card, mode and seed. --dry-run is host-only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time


GPU = {
    'rtx3070': ('NVIDIA GeForce RTX 3070', '8192'),
    'rtx3060': ('NVIDIA GeForce RTX 3060', '12288'),
}


def command(argv, *, check=True, stdin=None):
    result = subprocess.run(argv, input=stdin, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f'{argv[:3]} failed ({result.returncode}): {result.stderr.strip()}')
    return result


def gpu_memory(name):
    result = command(['nvidia-smi', '--query-gpu=name,memory.used',
                      '--format=csv,noheader,nounits'], check=False)
    if result.returncode:
        return None
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) == 2 and row[0].strip() == name:
            try:
                return int(row[1].strip())
            except ValueError:
                return None
    return None


def record(image, output, value):
    code = ('import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); '
            'p.mkdir(parents=True,exist_ok=True); v=json.load(sys.stdin); '
            '(p/"launcher.json").write_text(json.dumps(v["launcher"],indent=2)); '
            '(p/"console.log").write_text(v["console"]); '
            '(p/"external-failure.json").write_text(json.dumps(v["failure"],indent=2)) '
            'if v.get("failure") else None')
    command(['docker', 'run', '--rm', '-i', '--mount',
             'type=volume,source=neuroevo-results,target=/results',
             '--entrypoint', 'python', image, '-c', code, output],
            stdin=json.dumps(value))


def cold_completed(logs, last_cold_generation):
    for line in logs.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and 'Q2' in event and event.get('generation', -1) >= last_cold_generation:
            return True
    return False


def run_cell(image, digest, manifest, manifest_hash, root, profile, mode, seed,
             stage_probe=False):
    run_name = Path(root).name
    name = f'{run_name}-{profile}-{mode}-{seed}'
    output = f'{root}/{profile}-{mode}-{seed}'
    cache = f'/cache/{run_name}/{profile}-{mode}-{seed}'
    exists = command(['docker', 'run', '--rm', '--mount',
        'type=volume,source=neuroevo-results,target=/results,readonly',
        '--mount', 'type=volume,source=neuroevo-cache,target=/cache,readonly',
        '--entrypoint', 'python', image, '-c',
        'import pathlib,sys; raise SystemExit(any(pathlib.Path(p).exists() for p in sys.argv[1:]))',
        output, cache], check=False)
    if exists.stderr.strip():
        raise RuntimeError(f'Could not inspect Q2 output/cache: {exists.stderr.strip()}')
    if exists.returncode:
        raise RuntimeError(f'Output or cold cache already exists for {name}; choose a versioned root')
    gpu_name, gpu_mib = GPU[profile]
    argv = ['docker', 'run', '-d', '--name', name, '--gpus', 'all',
        '--memory=6g', '--cpus=4', '--shm-size=1g',
        '--mount', 'type=volume,source=neuroevo-results,target=/results',
        '--mount', 'type=volume,source=neuroevo-cache,target=/cache',
        '-e', 'NEUROEVO_BACKEND=gpu', '-e', f'NEUROEVO_GPU_NAME={gpu_name}',
        '-e', f'NEUROEVO_GPU_MIB={gpu_mib}',
        '-e', f'NEUROEVO_IMAGE_DIGEST={digest}',
        '-e', f'JAX_COMPILATION_CACHE_DIR={cache}', image, 'upstream',
        '--manifest', 'configs/q2-capacity-v2.json', '--mode', mode,
        '--seed', str(seed), '--output', output]
    if stage_probe:
        argv.append('--stage-probe')
    physical_baseline = gpu_memory(gpu_name)
    started = time.monotonic()
    command(argv)
    samples = []
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.is_set():
            used = gpu_memory(gpu_name)
            samples.append({'unix_seconds': time.time(), 'physical_used_mib': used})
            stop_monitor.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    cold_done = False
    killed_for = None
    overall = (manifest['maximum_cold_seconds'] +
               4*manifest['warm_generations']*manifest['maximum_warm_median_seconds'] + 120)
    try:
        while True:
            state = json.loads(command(['docker', 'inspect', name,
                '--format', '{{json .State}}']).stdout)
            elapsed = time.monotonic()-started
            if not cold_done:
                logs = command(['docker', 'logs', '--tail', '500', name], check=False)
                cold_done = cold_completed(logs.stdout + logs.stderr,
                                           manifest['cold_generations']-1)
            if not state['Running']:
                break
            if not cold_done and elapsed > manifest['maximum_cold_seconds']:
                killed_for = 'external-cold-compile-timeout'
            elif elapsed > overall:
                killed_for = 'external-total-process-timeout'
            if killed_for:
                command(['docker', 'kill', name])
                state = json.loads(command(['docker', 'inspect', name,
                    '--format', '{{json .State}}']).stdout)
                break
            time.sleep(2)
    finally:
        stop_monitor.set()
        thread.join(timeout=3)
    console = command(['docker', 'logs', name], check=False)
    used = [s['physical_used_mib'] for s in samples if s['physical_used_mib'] is not None]
    result = {'profile': profile, 'mode': mode, 'seed': seed,
        'stage_probe': stage_probe,
        'manifest_sha256': manifest_hash, 'image_digest': digest,
        'container': name, 'output': output, 'cache': cache,
        'exit_code': state['ExitCode'], 'oom_killed': state['OOMKilled'],
        'cold_generations_observed': cold_done,
        'external_timeout': killed_for, 'process_wall_seconds': time.monotonic()-started,
        'physical_gpu_memory_used_mib_baseline': physical_baseline,
        'physical_gpu_memory_used_mib_peak': max(used+[physical_baseline] if physical_baseline is not None else used)
            if used else None,
        'physical_gpu_memory_samples': samples}
    failure = ({'status': 'failed', 'reason': killed_for or 'container-exit',
                'exit_code': state['ExitCode'], 'oom_killed': state['OOMKilled']}
               if killed_for or state['ExitCode'] else None)
    record(image, output, {'launcher': result,
                           'console': console.stdout+console.stderr, 'failure': failure})
    print(json.dumps({k: result[k] for k in ('profile','mode','seed','exit_code',
        'oom_killed','cold_generations_observed','external_timeout','process_wall_seconds')}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image')
    p.add_argument('--manifest', default='configs/q2-capacity-v2.json')
    p.add_argument('--run-root')
    p.add_argument('--profiles', nargs='+', choices=tuple(GPU), default=list(GPU))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--stage-probe', action='store_true')
    args = p.parse_args()
    if args.image is None:
        args.image = 'neuroevo:q2-stage-v1' if args.stage_probe else 'neuroevo:q2-capacity-v2'
    if args.run_root is None:
        args.run_root = '/results/q2-stage-v1' if args.stage_probe else '/results/q2-capacity-v2'
    if not args.run_root.startswith('/results/') or '..' in Path(args.run_root).parts:
        raise ValueError('Q2 run root must be a directory inside /results')
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest['id'] != 'q2-capacity-promotion-v2' or
            manifest['recurrence']['max_sparse_conns'] != 32768 or
            manifest['recurrence']['sparse_cap_policy'] != 'refuse'):
        raise ValueError('Expected the frozen Q2 capacity promotion manifest')
    cells = [(profile, mode, seed) for profile in args.profiles
             for mode in manifest['modes'] for seed in manifest['seeds']]
    if args.dry_run:
        print(json.dumps({'image': args.image, 'run_root': args.run_root,
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'cold_deadline_seconds': manifest['maximum_cold_seconds'],
            'stage_probe': args.stage_probe, 'cells': cells}, indent=2))
        return
    active = command(['docker', 'ps', '--filter', 'name=^/neuroevo-m2-elapsed$',
                      '--format', '{{.Names}}']).stdout.strip()
    if active:
        raise RuntimeError('M2 elapsed comparison is still running; defer Q2 measurements')
    digest = command(['docker', 'image', 'inspect', args.image,
                      '--format', '{{.Id}}']).stdout.strip()
    baked_hash = command(['docker', 'run', '--rm', '--entrypoint', 'python',
        args.image, '-c',
        'import hashlib,pathlib; print(hashlib.sha256(pathlib.Path("configs/q2-capacity-v2.json").read_bytes()).hexdigest())']).stdout.strip()
    if baked_hash != hashlib.sha256(manifest_bytes).hexdigest():
        raise RuntimeError('Q2 image contains a different capacity manifest')
    results = []
    for profile, mode, seed in cells:
        results.append(run_cell(args.image, digest, manifest,
            hashlib.sha256(manifest_bytes).hexdigest(), args.run_root, profile, mode, seed,
            args.stage_probe))
    print(json.dumps({'complete_cells':sum(r['exit_code']==0 for r in results),
                      'total_cells':len(results)}), flush=True)


if __name__ == '__main__':
    main()
