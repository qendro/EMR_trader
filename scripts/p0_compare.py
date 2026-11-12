"""Produce a reproducible four-method P0 comparison from frozen run artifacts."""
import argparse
import json
from pathlib import Path

from p0_report import audit


METHODS = (
    ('adaptive', 'p0-partial-acceptance-v1'),
    ('direct', 'p0-partial-direct-acceptance-v1'),
    ('fixed', 'p0-partial-fixed-acceptance-v1'),
    ('weight', 'p0-partial-weight-acceptance-v1'),
)


def compare(root, baseline):
    root=Path(root)
    rows=[]
    for name,directory in METHODS:
        path=root/directory/'experiment.json'
        result=audit(path,baseline)
        experiment=json.loads(path.read_text())
        rows.append({'method':name,'experiment':result['experiment'],
            'config_sha256':result['config_sha256'],
            'engine_sha256':result['engine_sha256'],
            'heldout_means':[r['selection']['heldout_score']
                             for r in sorted(experiment['runs'],key=lambda row:row['seed'])],
            'quality':result['paired_quality'],'raw_seed_passes':result['raw_seed_passes'],
            'passes':result['passes']})
    return {'task':'cartpole-partial','comparison':'shared-heldout-protocol',
            'methods':rows,'any_method_passes':any(row['passes'] for row in rows)}


def markdown(result):
    lines=['# P0 partially observed CartPole comparison','',
        'All four methods used the frozen five evolution seeds, 100 generations, '
        '100 paired held-out episodes and the 500-step normalization scale. '
        'Each method retained its own genome operators and matching G9 pilot.','',
        '| Method | Held-out means by seed | Median paired gain | 95% seed bootstrap interval | Raw target seeds | Research target |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for row in result['methods']:
        quality=row['quality']
        held=', '.join(f'{score:.2f}' for score in row['heldout_means'])
        low,high=quality['bootstrap_95_interval']
        lines.append(f"| {row['method']} | {held} | "
            f"{100*quality['median_improvement']:.2f}% | "
            f"[{100*low:.2f}%, {100*high:.2f}%] | "
            f"{row['raw_seed_passes']}/5 | {'pass' if row['passes'] else 'open'} |")
    lines.extend(['','The research target requires median paired normalized '
        'improvement of at least 20% and a positive lower bootstrap endpoint '
        'resampled over evolution seeds. The raw 120-step seed count is a '
        'separate diagnostic. '
        'A failed method remains in the table.','',
        'The random-action baseline and known finite-difference controller '
        'are recorded separately in the P0 quality report. This comparison '
        'supports method diagnosis; it does not replace M4 scale, stability '
        'or both-card performance gates.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--json-output',required=True)
    parser.add_argument('--markdown-output',required=True)
    args=parser.parse_args()
    result=compare(args.root,args.baseline)
    Path(args.json_output).write_text(json.dumps(result,indent=2)+'\n')
    Path(args.markdown_output).write_text(markdown(result))
