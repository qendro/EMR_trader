"""Report H0/H1 correctness separately from the later M2 acceptance gates."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
p.add_argument('--source',default='/results')
p.add_argument('--output',default='/reports')
p.add_argument('--coefficient',required=True)
args=p.parse_args()
root,out=Path(args.source),Path(args.output)
out.mkdir(parents=True,exist_ok=True)
coefficient=json.loads(Path(args.coefficient).read_text())
status={'schema':1,'packages':['H0','H1'],'coefficient':coefficient,'profiles':{},
    'M2_acceptance':'open: B0, H2 memory/CartPole learning, comparison reports, W8 environment and integrated performance gates not run',
    'Q2':'original capacity failure/resource-stage audit remains open'}
lines=['# H0/H1 correctness and wiring report','',
    'Fixed-coordinate CPPN decoding, persistent layered inference, role-shaped dense blocks, and portable artifacts. '
    'This report does not close M2 or qualify its learning/performance gates. The M1 report is historical evidence '
    'for its separately retained source and image.','',
    '| Profile | Tests passed | 256 / 1024 nodes | Frozen calibration |',
    '|---|---:|---|---|']
complete=True
for profile in ('cpu','rtx3070','rtx3060'):
    xml=root/f'h1-qualification-v1/tests-{profile}.xml'
    case={}
    if xml.exists():
        x=ET.parse(xml).getroot().find('testsuite')
        case['tests']={k:int(x.get(k)) for k in ('tests','failures','errors','skipped')}
    for key,path in (('large',f'h1-audit-v2/{profile}/result.json'),
                     ('calibration',f'h1-calibration-v2/{profile}-validation.json')):
        file=root/path
        case[key]=json.loads(file.read_text()) if file.exists() else None
    case['smoke']={}
    for task in ('cartpole','memory'):
        path=root/f'h1-smoke-v2/{profile}-{task}'
        if (path/'seed-900/selection.json').exists() and (path/'replay.json').exists():
            selected=json.loads((path/'seed-900/selection.json').read_text())
            replay=json.loads((path/'replay.json').read_text())
            summary=json.loads((path/'seed-900/summary.json').read_text())
            assert selected['kind']=='smoke' and summary['provenance']==coefficient['provenance']
            case['smoke'][task]={'heldout':selected['heldout_score'],
                'generalization':selected.get('generalization_score'),'replay':replay['status'],
                'kind':'smoke-not-M2-acceptance','generations':summary['generations']}
    tests=case.get('tests',{})
    passed=bool(tests.get('tests')==54 and all(tests[k]==0 for k in ('failures','errors','skipped')))
    large=case['large']
    if large:
        assert large['provenance']==coefficient['provenance'] and large['coefficient']==coefficient['coefficient']
    ready=bool(passed and large and large['status']=='pass' and case['calibration']
               and case['calibration']['status']=='validation-pass' and len(case['smoke'])==2
               and all(v['replay']=='pass' for v in case['smoke'].values()))
    case['H0_H1_checks_pass']=ready
    complete &= ready
    status['profiles'][profile]=case
    lines.append(f'| {profile} | {tests.get("tests","pending") if passed else "incomplete"} | '
                 f'{large["status"] if large else "pending"} | {case["calibration"]["status"] if case["calibration"] else "pending"} |')
lines+=['','## Arithmetic and failed calibration coverage','',
    'V1 c=30 failed the CPU 1024-node decoded output preactivation (ratio 3.451 of its threshold). '
    'The initial audit had omitted this output check and is superseded. The failure remains in raw evidence. '
    'Before further qualification, v2 added single-candidate hidden products and repeated-term two-output reductions. '
    'Its frozen rule derived c='+str(coefficient['coefficient'])+' from CPU and RTX 3070 only; RTX 3060 takes no part '
    'in choosing it. All c-dependent calibration/large-fixture checks were rerun on every profile. '
    'Small-fan-in and stable-trajectory tolerances, FP32/HIGHEST policy and learning targets were unchanged. '
    'See `docs/h1.md` for the dated rationale and exact rerun scope.','',
    '| Profile | Nodes | Contractive trajectory max absolute error | Decoded trajectory max absolute error | Hidden / output preactivation error-to-threshold |',
    '|---|---:|---:|---:|---:|']
for profile,case in status['profiles'].items():
    if case['large']:
        for row in case['large']['fixtures']:
            lines.append(f'| {profile} | {row["nodes"]} | {row["trajectory_error_max"]:.3g} | {row["decoded_trajectory_error_max"]:.3g} | '
                f'{row["preactivation_error_to_frozen_threshold_ratio"]:.3g} / {row["decoded_output_preactivation_error_to_threshold_ratio"]:.3g} |')
lines+=['','Trajectories are checked at atol=1e-4 and rtol=1e-4. The constant-score decoder fixture has large '
    'identity outputs, so its absolute error must be interpreted with the relative term. The separate contractive '
    'fixture checks unsaturated dynamics. Both run 128-step compiled scans with connected recurrent phenotypes. '
    'Small mixed-delay cases also match the independent scalar FP64 graph oracle.','',
    'At 1024 total nodes (4 inputs, 1018 hidden, 2 outputs), the decoder evaluates 1,043,450 coordinate queries '
    'in 4096-row slabs, producing 1,042,432 edges plus 1018 bias queries. Role-block arrays occupy 5,221,384 host '
    'bytes per phenotype. These are synthetic P=1/B=1 correctness fixtures, not evolved agents or the M2 G5 cell. '
    'Export/reload preserves arrays exactly. Changing weight/mask/observation values adds no compiled trajectory '
    'signature within each fixed shape. This is not the 100-generation M2 G2 gate.','',
    '## Wiring demonstrations','',
    '| Profile | CartPole smoke held-out | Memory smoke interpolation / generalization | Portable replay |',
    '|---|---:|---:|---|']
for profile,case in status['profiles'].items():
    a,b=case['smoke'].get('cartpole'),case['smoke'].get('memory')
    lines.append(f'| {profile} | {a["heldout"] if a else "pending"} | '
        f'{str(b["heldout"])+" / "+str(b["generalization"]) if b else "pending"} | '
        f'{"pass" if a and b and a["replay"]==b["replay"]=="pass" else "pending"} |')
lines+=['','All wiring runs use seed 900 and four generations at P=8. CartPole uses its smoke horizon 64; '
    'memory uses the shared 32-training-episode/17-step protocol. They archive evaluated CPPN genes and decoded '
    'phenotypes before tell. The full controller split-resume/reconstruction is also covered in the 54-test suite. '
    'No smoke score substitutes for five-seed H2 acceptance.','',
    'The next work is B0, then H2 with frozen learning manifests, integrated G9 pilots, the geometric diagnostic, '
    'W8 sequence environment and the system/encoding-isolating reports. Q2’s resource/capacity work remains '
    'a separate prerequisite for A0. No adaptive platform decoder, scale optimizer or M2 completion is claimed.']
status['H0_H1_complete']=complete
(out/'h1-status.json').write_text(json.dumps(status,indent=2)+'\n')
(out/'H1-REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'H0_H1_complete':complete,'report':str(out/'H1-REPORT.md')}))
