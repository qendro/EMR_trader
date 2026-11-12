"""Versioned adaptive 5x5 diagnostic capability boundary."""
import json
from pathlib import Path

import pytest

from neuroevo.adapters.emr import settings, translate
from neuroevo.runner import load_config


VISUAL = json.loads(Path('configs/a2-visual-adaptive-diagnostic-v1.json').read_text())
BASE = json.loads(Path('configs/q2-capacity-v2.json').read_text())


def test_visual_adaptive_config_uses_exact_shared_grid_and_public_api():
    from emr_hyperneat import EMRHyperNEAT

    config = load_config('configs/a2-visual-adaptive-diagnostic-v1.json')
    smoke = load_config('configs/a2-visual-smoke-v1.json')
    assert smoke['adapter']['population'] == 8
    checked = settings(config['adaptive'])
    assert len(checked['input_coords']) == 25
    assert checked['max_depth'] == 1 and not checked['recurrent']
    algo = EMRHyperNEAT()
    algo.create_config(translate(checked, BASE))
    assert len(algo.substrate_input_coords) == 25
    assert len(algo.substrate_output_coords) == 1


@pytest.mark.parametrize('change', [
    {'recurrent': True},
    {'max_depth': 3},
    {'input_coords': [[-1, -1], [1, -1]]},
    {'input_coords': [[-1, -1]] * 25},
    {'output_coords': [[-1, 1], [1, 1]]},
])
def test_visual_adaptive_rejects_other_envelopes(change):
    with pytest.raises(ValueError):
        settings({**VISUAL['adaptive'], **change})
