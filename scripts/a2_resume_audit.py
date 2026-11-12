"""Compare trusted local full and split A2 runs at the native state boundary."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from neuroevo.artifacts import load_checkpoint


def compare(full, resumed):
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (full, resumed)]
    if manifests[0]['config'] != manifests[1]['config'] or \
            manifests[0]['provenance'] != manifests[1]['provenance'] or \
            manifests[0]['seed'] != manifests[1]['seed']:
        raise ValueError('A2 resume runs have different manifests')
    m = manifests[0]
    config = {**m['config'], 'run_seed': m['seed']}
    a, b = (load_checkpoint(p/'checkpoint.bin', config, m['provenance'])
            for p in (full, resumed))
    leaves_a, leaves_b = jax.tree.leaves(a['state']), jax.tree.leaves(b['state'])
    exact_state = len(leaves_a) == len(leaves_b) and all(
        np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(leaves_a, leaves_b))
    keys = ('generation', 'best', 'mean', 'valid', 'species', 'nodes_max',
            'edges_max', 'adaptive_graphs')
    quality = [[{k: row[k] for k in keys} for row in saved['history']]
               for saved in (a, b)]
    selection = [json.loads((p/'selection.json').read_text()) for p in (full, resumed)]
    result = {'exact_native_state': exact_state,
              'same_quality_history': quality[0] == quality[1],
              'same_selection': selection[0] == selection[1],
              'generations': len(quality[0]), 'state_leaves': len(leaves_a)}
    if not all(result[k] for k in ('exact_native_state', 'same_quality_history', 'same_selection')):
        raise AssertionError(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('full', type=Path)
    parser.add_argument('resumed', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.full, args.resumed), indent=2))
