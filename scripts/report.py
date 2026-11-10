"""Generate a compact report from measured artifacts; missing gates stay open."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
p.add_argument('--source',default='/results')
p.add_argument('--output',default='/reports')
p.add_argument('--telemetry',default='/reports/results/current-gpu.csv')
args=p.parse_args()
root,out=Path(args.source),Path(args.output)
out.mkdir(parents=True,exist_ok=True)
def read(path):
    path=root/path
    return json.loads(path.read_text()) if path.exists() else None
def median(values): return statistics.median(values)
profiles=('cpu','rtx3070','rtx3060')
status={'schema':1,'date':'2026-11-10','profiles':{},'q2':[], 'diagnostics':{}}
lines=['# M1 implementation evidence','',
       'Measured on this Windows/WSL2 host in Linux Docker containers. RTX 3070 gates progression. '
       'Every lane uses the same hash-locked dependencies; smoke and pilot runs are excluded from learning acceptance.','',
       '| Profile | Tests | Held-out means, seeds 0–4 | Wall seconds per seed | G9 warm seconds |',
       '|---|---:|---|---|---:|']
telemetry={}
path=Path(args.telemetry)
if path.exists():
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,skipinitialspace=True):
            name=row['name'].strip()
            v=telemetry.setdefault(name,{'samples':0,'used_mib_max':0.,'free_mib_min':float('inf'),
                                         'temperature_c_max':0.})
            v['samples']+=1
            v['used_mib_max']=max(v['used_mib_max'],float(row['used_mib']))
            v['free_mib_min']=min(v['free_mib_min'],float(row['free_mib']))
            v['temperature_c_max']=max(v['temperature_c_max'],float(row['temperature_c']))
vm_path=out/'results/current-vm.jsonl'
if vm_path.exists():
    samples=[json.loads(line) for line in vm_path.read_text().splitlines() if line.strip()]
    status['vm_observed_window']={'samples':len(samples),
        'start_unix_seconds':samples[0]['unix_seconds'],'end_unix_seconds':samples[-1]['unix_seconds'],
        'mem_available_min_bytes':min(s['MemAvailable_bytes'] for s in samples),
        'swap_used_start_bytes':samples[0]['SwapTotal_bytes']-samples[0]['SwapFree_bytes'],
        'swap_used_end_bytes':samples[-1]['SwapTotal_bytes']-samples[-1]['SwapFree_bytes'],
        'swap_out_pages':samples[-1]['pswpout_pages']-samples[0]['pswpout_pages']}
for profile in profiles:
    lane={}
    q=read(f'qualification-current/q1-{profile}.json')
    lane['q1']=q['status'] if q else 'not-run'
    oracle=read(f'qualification-current/oracle-{profile}.json')
    lane['oracle']=oracle['status'] if oracle else 'not-run'
    x=root/f'qualification-current/tests-{profile}.xml'
    if x.exists():
        suite=ET.parse(x).getroot().find('testsuite')
        lane['tests']={k:int(suite.get(k)) for k in ('tests','errors','failures','skipped')}
    d=read(f'qualification-current/differential-{profile}.json')
    lane['differential']={'status':d['status'],
        'max_error_threshold_ratio':max(r['one_step_max_error_threshold_ratio'] for r in d['fixtures']),
        'max_trajectory_error':max(r['trajectory_max_abs_error'] for r in d['fixtures'])} if d else {'status':'not-run'}
    e=read(f'm1-current-{profile}/experiment.json')
    pilot=read(f'm1-current-{profile}/pilot.json')
    lane['g9']=({k:pilot[k] for k in ('passes','fixed_overhead_seconds','warm_generation_seconds','allowable_generations')}
                if pilot else {'status':'not-run'})
    if e:
        lane['learning']={'complete_protocol':e['complete_protocol'],'passed_seeds':e['passed_seeds'],
            'heldout_scores':[r['selection']['heldout_score'] for r in e['runs']],
            'wall_seconds':[r['wall_seconds'] for r in e['runs']],
            'structural_mutation_all_seeds':all(r['structural_mutation_observed'] for r in e['runs']),
            'engine_sha256':e['runs'][0]['provenance']['engine_sha256'],
            'image_digest':e['runs'][0]['hardware']['image_digest']}
        memory=[r['memory'] for r in e['runs']]
        lane['memory']={'cgroup_peak_bytes':max(m['cgroup_peak_bytes'] for m in memory),
            'process_peak_rss_bytes':max(m['process_peak_rss_bytes'] for m in memory),
            'swap_bytes_max':max(m['cgroup_swap_bytes'] for m in memory)}
        if profile != 'cpu':
            gpu=e['runs'][0]['hardware']['gpu']
            lane['memory']['gpu_allocator_peak_bytes']=max(m['device_allocator']['peak_bytes_in_use'] for m in memory)
            lane['memory']['gpu_pool_peak_bytes']=max(m['device_allocator']['peak_pool_bytes'] for m in memory)
            lane['memory']['physical_samples']=telemetry.get(gpu[1])
        scores=', '.join(f'{v:g}' for v in lane['learning']['heldout_scores'])
        wall=', '.join(f'{v:.1f}' for v in lane['learning']['wall_seconds'])
        lines.append(f"| {profile} | {lane.get('tests',{}).get('tests','pending')} | {scores} | {wall} | {pilot['warm_generation_seconds']:.4f} |")
    else: lines.append(f'| {profile} | {lane.get("tests",{}).get("tests","pending")} | pending | pending | pending |')
    history=read(f'g4-{profile}/driver-0/history.json')
    driver_result=read(f'g4-{profile}/driver-0/result.json')
    if driver_result and 'memory' in driver_result and 'memory' in lane:
        m=driver_result['memory']
        for field in ('cgroup_peak_bytes','process_peak_rss_bytes'):
            lane['memory'][field]=max(lane['memory'][field],m[field])
        lane['memory']['swap_bytes_max']=max(lane['memory']['swap_bytes_max'],m['cgroup_swap_bytes'])
        if profile!='cpu':
            lane['memory']['gpu_allocator_peak_bytes']=max(lane['memory']['gpu_allocator_peak_bytes'],m['device_allocator']['peak_bytes_in_use'])
            lane['memory']['gpu_pool_peak_bytes']=max(lane['memory']['gpu_pool_peak_bytes'],m['device_allocator']['peak_pool_bytes'])
    if history and len(history)==102:
        signatures={field:sorted({r[field] for r in history[2:]})
                    for field in ('rollout_signatures','tell_signatures','observer_signatures')}
        compile_calls=sum(r.get('backend_compile_calls',0) for r in history[2:])
        lane['g2']={'post_warm_generations':100,'signatures':signatures,
                    'backend_compile_calls':compile_calls,
                    'passes':all(len(v)==1 for v in signatures.values()) and compile_calls==0
                             and all('backend_compile_calls' in r for r in history[2:]),
                    'source':f'g4-{profile}/driver-0'}
    else: lane['g2']={'status':'not-run-or-incomplete'}
    pairs=[]
    for trial in range(3):
        c=read(f'g4-{profile}/controller-{trial}/pilot-run/history.json')
        m=read(f'g4-{profile}/driver-{trial}/history.json')
        cp=read(f'g4-{profile}/controller-{trial}/pilot.json')
        mp=read(f'g4-{profile}/driver-{trial}/result.json')
        if not c or not m or not cp or not mp or len(c)!=12 or len(m) not in (12,102): continue
        for field in ('config_sha256','hardware','provenance'):
            assert cp[field]==mp[field],f'Unmatched G4 provenance: {profile}/{trial}/{field}'
        assert cp['seed']==mp['seed']==900
        for a,b in zip(c[:11],m[:11]):
            for key in ('generation','best','mean','valid','species','stagnation_removed','nodes_max','edges_max'):
                if a[key] != b[key]: raise AssertionError(f'Unmatched G4 quality field: {profile}/{trial}/{key}')
        controller=median([r['total_seconds']-r['selection_seconds'] for r in c[2:11]])
        minimal=median([r['total_seconds'] for r in m[2:11]])
        pairs.append({'trial':trial,'controller_seconds':controller,'minimal_seconds':minimal,
                      'ratio':controller/minimal})
    cost_ratio=(median([v['controller_seconds'] for v in pairs])/median([v['minimal_seconds'] for v in pairs])
                if len(pairs)==3 else None)
    lane['g4']={'pairs':pairs,'ratio':cost_ratio,'passes':cost_ratio<=1.2 if cost_ratio is not None else None,
                'median_pair_ratio':median([v['ratio'] for v in pairs]) if len(pairs)==3 else None,
                'sample_generations':'2..10 inclusive; terminal selection and forced terminal checkpoint excluded'}
    status['profiles'][profile]=lane
lines+=['','The target is ≥450 in at least three of five seeds. Each seed has a 7,200-second budget. '
        'G9 pilots use seed 900. Learning runs shared host CPUs with other diagnostic/card runs; '
        'their timings establish feasibility, not an isolated GPU speed comparison.','',
        '## Correctness, compilation and controller cost','',
        '| Profile | Differential max error/threshold | G2 100 post-warm generations | G4 median ratio |',
        '|---|---:|---|---:|']
for profile,lane in status['profiles'].items():
    ratio=lane['differential'].get('max_error_threshold_ratio')
    pairs=lane['g4']['pairs']
    cost=f"{lane['g4']['ratio']:.3f}" if len(pairs)==3 else 'open'
    lines.append(f'| {profile} | {ratio if ratio is not None else "open"} | {lane["g2"].get("passes","open")} | {cost} |')
for name in ('resume-40-100.json','closed-loop-replay-cpu.json','statistics-cpu.json'):
    status[name]=read(f'qualification-current/{name}')
lines+=['','Complete native state, evaluated archive and history quality fields are compared after separate-container '
        '40/100 CPU resume. Standalone closed-loop replay compares all actions, policy states, environment states and rewards '
        'against the independent NumPy reference. See the qualification artifacts for status and numerical errors.','',
        'G4 uses three fresh-process pairs, the same seed 900, dimensions, diagnostics and checkpoint interval. '
        'It compares generations 2–10 inclusive in both paths. Driver trial 0 continues to 102 generations for G2; '
        'the other two stop at 12. Numerical/quality fields must match exactly before a pair is accepted.','',
        '## Ungated diagnostics','']
if all(len(v['g4']['pairs'])==3 for v in status['profiles'].values()):
    position=lines.index('## Ungated diagnostics')
    detail=['G4 compares the median of the three controller medians with the median of the three driver medians. '
            'All pairs, including timing outliers, remain in the raw report.','',
            '| Profile | Trial | Controller median seconds | Driver median seconds | Pair ratio |',
            '|---|---:|---:|---:|---:|']
    for profile,lane in status['profiles'].items():
        for v in lane['g4']['pairs']:
            detail.append(f"| {profile} | {v['trial']} | {v['controller_seconds']:.4f} | {v['minimal_seconds']:.4f} | {v['ratio']:.3f} |")
    lines[position:position]=detail+['']
for task in ('memory','dag-control','synchronous-cartpole','fixed-scenario'):
    values=[]
    for seed in range(100,105):
        selection=read(f'diagnostics/current-{task}/seed-{seed}/selection.json')
        history=read(f'diagnostics/current-{task}/seed-{seed}/history.json')
        if not selection or not history: continue
        values.append({'seed':seed,'heldout_score':selection['heldout_score'],
            'generalization_score':selection.get('generalization_score'),
            'interpolation_by_delay':selection.get('interpolation_by_delay'),
            'generalization_by_delay':selection.get('generalization_by_delay'),
            'stagnation_removed':sum(len(r['stagnation_removed']) for r in history),
            'first_training_threshold_generation':next((r['generation'] for r in history if r['best'] >= (.9 if task=='memory' else 450)),None)})
    status['diagnostics'][task]=values
    lines.append(f'- {task}: '+(', '.join(f"seed {r['seed']} = {r['heldout_score']:g}" for r in values) if len(values)==5 else 'incomplete')+'.')
lines+=['','Memory interpolation delays 1–8 and generalization delays 9–16 are reported separately per delay. '
        'Only final champions receive held-out evaluation: the recorded first training threshold is a training proxy, '
        'not a measured held-out solve generation. Synchronous hidden hops add one observation of latency per hidden node; '
        'the latency fixtures test one and two hops. Comparisons use diagnostic seeds 100–104, separate from acceptance.','']
if all(len(v)==5 for v in status['diagnostics'].values()):
    lines+=['| Seed | DAG return | Synchronous minus DAG return | Varying-scenario stagnation removals | Fixed-scenario removals | Memory interpolation / generalization | First training ≥90% generation (zero based) |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for dag,sync,fixed,memory in zip(status['diagnostics']['dag-control'],
            status['diagnostics']['synchronous-cartpole'],status['diagnostics']['fixed-scenario'],
            status['diagnostics']['memory']):
        lines.append(f"| {dag['seed']} | {dag['heldout_score']:g} | {sync['heldout_score']-dag['heldout_score']:g} | "
            f"{dag['stagnation_removed']} | {fixed['stagnation_removed']} | "
            f"{memory['heldout_score']:g} / {memory['generalization_score']:g} | {memory['first_training_threshold_generation']} |")
    lines+=['','The synchronous comparison changes both propagation timing and permitted genome operators. '
            'It is a direct-genome system comparison; it isolates no encoding advantage. Ceiling CartPole scores '
            'do not establish that latency has no cost on other tasks. Stagnation removal differences are descriptive '
            'over five matched seeds, with no significance claim.']
lines+=['',
        '## Q2 upstream screen','',
        '| Card | Mode | Seed | Result | Warm seconds/generation |','|---|---|---:|---|---:|']
for directory,profile in (('q2-v2','rtx3060'),('q2-complete','rtx3070')):
    for mode in ('feedforward','recurrent'):
        for seed in (700,701,702):
            prefix=f'{directory}/{profile}-{mode}-{seed}'
            v=read(f'{prefix}/result.json')
            failure=read(f'{prefix}/failure.json')
            value={'profile':profile,'mode':mode,'seed':seed,'path':prefix,
                   'result':v,'failure':failure}
            status['q2'].append(value)
            label='timing pass; resource audit open' if v else (failure['type'] if failure else 'not-run')
            warm=f"{v['warm_median_seconds']:.4f}" if v else '—'
            lines.append(f'| {profile} | {mode} | {seed} | {label} | {warm} |')
lines+=['','Both recurrent seed-702 runs refuse a 13,424-connection discovery at the frozen 10,000 cap. '
        '`q2-capacity-v2.json` declares 32,768 slots with refusal on overflow; it has not run. '
        'Q2 remains open: profile peak-resource audit, complete active-size distributions '
        'and synchronized isolated stage timing are still required. Native upstream stage timers are dispatch/incidental-wait '
        'measurements. Successful warm medians are not sequential-rollout budgets or complete Q2 passes. '
        'Full CPU EMR and max-depth-5 cells have not run; neither is inferred from the GPU depth-4 screen.','',
        '### Raw upstream direct/fixed baselines','',
        'These use 64 independent samples, native DAG direct genomes or native fixed HyperNEAT '
        'with reset per sample and K=3 settling. They share the frozen dataset with EMR but differ in capacity, '
        'functions and propagation timing. This is a system comparison, not evidence about encoding alone. '
        'V1 passed raw scores to native evolution; nonfinite direct fitness caused strict JSON failures. '
        'V2 freezes candidate rejection for nonfinite outputs/fitness, -1e30 invalid fitness, counts per generation '
        'and all-invalid abort. It changes no native operators or workload capacities. V1 failed cells remain intact. '
        'V2 uses one initially empty persistent cache per profile/mode: seed 700 is empty-cache, '
        '701/702 retain it in fresh processes. Initial entry counts are recorded; their first-generation times '
        'are not empty-cache compilation measurements.','',
        '| Version | Profile | Mode | Completed / failed seeds | Warm median seconds (range across completed seeds) | Invalid candidate evaluations |',
        '|---|---|---|---|---:|---:|']
status['q2_native']=[]
status['q2_native_v1']=[]
for version in ('v1','v2'):
    for profile in profiles:
        for mode in ('direct','fixed'):
            values=[]
            failed=0
            invalid=[]
            for seed in (700,701,702):
                v=read(f'q2-native-{version}/{profile}-{mode}-{seed}/result.json')
                failure=read(f'q2-native-{version}/{profile}-{mode}-{seed}/failure.json')
                if v:
                    values.append(v['warm_median_seconds'])
                    if 'invalid_candidate_evaluations' in v: invalid.append(v['invalid_candidate_evaluations'])
                    status['q2_native' if version=='v2' else 'q2_native_v1'].append({'profile':profile,'mode':mode,'seed':seed,**v})
                if failure:
                    failed+=1
                    status.setdefault('q2_native_failures',[]).append({'version':version,'profile':profile,'mode':mode,'seed':seed,**failure})
            timing=(f'{median(values):.4f} ({min(values):.4f}–{max(values):.4f})' if values else 'no completed cell')
            invalid_text=', '.join(map(str,invalid)) if invalid else 'not recorded'
            lines.append(f'| {version} | {profile} | {mode} | {len(values)} / {failed} of 3 | {timing} | {invalid_text} |')
native_replay=read('q2-native-v2/cpu-direct-700/replay.json')
status['native_invalid_replay']=native_replay
if native_replay:
    lines+=['','The first-invalid-genome replay reproduces '+str(native_replay['nonfinite_outputs'])+
        ' nonfinite outputs across 64 samples. Active attributes are finite; isolated non-input node IDs are '+
        str(native_replay['isolated_noninput_ids'])+'. The fixture, genes, inputs, outputs and replay result are retained '
        'under `q2-native-v2/cpu-direct-700`.']
dataset_audit_path=out/'results/q2-dataset-audit.json'
if dataset_audit_path.exists():
    dataset_audit=json.loads(dataset_audit_path.read_text())
    status['q2_dataset_audit']={'status':dataset_audit['status'],'datasets':len(dataset_audit['datasets'])}
    lines+=['',str(len(dataset_audit['datasets']))+' persisted Q2 datasets have byte-identical FP32 input/target arrays '
            '(`results/q2-dataset-audit.json`); the audit also checks completed v2 histories, manifest/script identity '
            'and the declared cache policy.']
lines+=['',
        '## Scope and provenance','',
        'Implemented: Q0/Q1 and V0–V5, the M1 one-capacity DAG/synchronous runtime, scalar layered reference, '
        'strict portable import/export, native checkpoints, memory/CartPole evaluation and reducer fixtures. '
        'M2–M5, platform fixed/adaptive decoders, large-fan-in calibration, W8 sequence environment, scale optimizations '
        'and sustained W7 stability are not implemented/qualified. No trading functionality is included.','',
        'Current results live under `results/evidence/*current*` and `g4-*`; every run records its engine and image hash. '
        'Earlier source `e03c0613…` results and checkpoints are retained under `m1-final-*`, with exact source in '
        '`reproduction/m1-e03`. They do not supply acceptance evidence for the final overflow guard. '
        'Other retained development results are identified by their own manifests. '
        'The file index contains SHA-256 hashes, including failed cells and portable/checkpoint artifacts.']
gates=[lane['q1']=='pass' and lane.get('tests',{}).get('failures')==0 and
       lane['differential'].get('status')=='pass' and lane.get('learning',{}).get('passed_seeds',0)>=3
       and lane.get('learning',{}).get('complete_protocol',False)
       for lane in status['profiles'].values()]
status['learning_and_correctness_all_profiles']=all(gates)
for profile,lane in status['profiles'].items():
    memory=lane.get('memory')
    lane['g3']={'passes':False,'host_limit_bytes':6*1024**3,'vram_limit_fraction':.6 if profile!='cpu' else None}
    if memory:
        bounded=(memory['cgroup_peak_bytes']<=6*1024**3 and memory['process_peak_rss_bytes']<=6*1024**3
                 and memory['swap_bytes_max']==0)
        if profile!='cpu':
            total=8192 if profile=='rtx3070' else 12288
            sampled=memory.get('physical_samples')
            bounded=bounded and sampled is not None and sampled['used_mib_max']<=.6*total and sampled['free_mib_min']>=1024
            bounded=bounded and memory['gpu_pool_peak_bytes']<=.6*total*1024**2
        window=status.get('vm_observed_window')
        lane['g3']['passes']=bool(bounded and window and window['swap_out_pages']==0 and window['mem_available_min_bytes']>0)
        lane['g3']['coverage']='per-run cgroup/process/allocator peaks; physical GPU and VM samples in recorded time window'
c=status.get('resume-40-100.json') or {}
replay=status.get('closed-loop-replay-cpu.json') or {}
functional=all(gates) and c.get('status')=='pass' and replay.get('status')=='pass' and all(len(v)==5 for v in status['diagnostics'].values())
status['m1_functional']=functional
status['m1_closed']=bool(functional and all(lane['oracle']=='pass' and lane['g2'].get('passes') and lane['g3']['passes']
    for lane in status['profiles'].values()) and status['profiles']['rtx3070']['g4']['passes'])
status['m1_status']=('closed-first-vertical-slice; H0 is next; M2–M5 not implemented' if status['m1_closed'] else
    'open-required-evidence-or-gate; consult per-profile G1–G4, resume and diagnostic artifacts')
status['q2_status']=('open-capacity-failure-and-resource-stage-audit' if len(status['q2_native'])==18 else
                     'open-capacity-failure-and-incomplete-baselines-resource-audit')
if all(len(status['profiles'][p]['g4']['pairs'])==3 for p in profiles):
    cpu=median([v['controller_seconds'] for v in status['profiles']['cpu']['g4']['pairs']])
    status['w3_gpu_to_cpu_throughput_ratio']={p:cpu/median([v['controller_seconds'] for v in status['profiles'][p]['g4']['pairs']]) for p in profiles[1:]}
    lines+=['','Measured W3 GPU/CPU throughput ratios: '+json.dumps(status['w3_gpu_to_cpu_throughput_ratio'])+'. '
            'M1 reports these ratios without gating; no useful-acceleration claim follows from CUDA placement.']
    status['w3_controller_medians_and_ranges']={p:{'median_seconds':median([v['controller_seconds'] for v in status['profiles'][p]['g4']['pairs']]),
        'range_seconds':[min(v['controller_seconds'] for v in status['profiles'][p]['g4']['pairs']),max(v['controller_seconds'] for v in status['profiles'][p]['g4']['pairs'])]}
        for p in profiles}
    lines+=['','Three-run controller medians and ranges (seconds): '+json.dumps(status['w3_controller_medians_and_ranges'])+'.']
lines+=['','## Gate disposition','',status['m1_status']+'.',
        'Per-profile G3: '+json.dumps({p:v['g3']['passes'] for p,v in status['profiles'].items()})+'. '
        'Host/process/cgroup and allocator peaks are recorded separately. Physical GPU and VM observations use '
        'the recorded sampling window; pre-existing VM swap use is retained and swap-out growth is checked separately. '
        'See JSON for maxima, minimum free memory, thermals, sample counts and window boundaries. '
        'The RTX 3060 drives the host display; the RTX 3070 display is disabled (`results/display-state.csv`). '
        'Physical telemetry includes background display use.']
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'report':str(out/'REPORT.md'),'current_learning_complete':all(gates)}))
