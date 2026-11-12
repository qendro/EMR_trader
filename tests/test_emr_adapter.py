"""A0 public-config translation and capability rejection."""
import json
from pathlib import Path

import pytest

from neuroevo.adapters.emr import settings, translate


BASE = json.loads(Path('configs/q2-capacity-v2.json').read_text())
CONFIG = json.loads(Path('configs/a0-adapter-v1.json').read_text())


def body(config=CONFIG):
    return translate(config, BASE)['algorithm_params']['emrhyperneat']


def test_a0_translation_and_upstream_public_config():
    from emr_hyperneat import EMRHyperNEAT
    actual = body()
    assert actual['population_size'] == 128
    emr = actual['emr_hyperneat']
    assert emr['max_depth'] == 4 and emr['variance_threshold'] == .03
    assert emr['execution'] == {'mode':'standard'}
    assert emr['recurrence']['max_sparse_conns'] == 32768
    assert emr['recurrence']['sparse_cap_policy'] == 'refuse'
    assert emr['recurrence']['hh_cache_enabled'] is False
    assert actual['substrate']['input_coords'] == CONFIG['input_coords']
    assert actual['substrate']['output_coords'] == CONFIG['output_coords']
    # The public API must accept the translated shape, not just a dictionary test.
    EMRHyperNEAT().create_config(translate(CONFIG, BASE))


def test_every_exposed_a0_setting_changes_the_public_payload():
    from emr_hyperneat import EMRHyperNEAT
    original = body()
    def interpreted(config):
        algorithm = EMRHyperNEAT()
        algorithm.create_config(translate(config, BASE))
        return algorithm
    baseline = interpreted(CONFIG)
    changed = {
        'population': 64,
        'max_depth': 3,
        'variance_threshold': .04,
        'weight_threshold': .2,
        'max_weight': 7.0,
        'band_threshold': .4,
        'max_sparse_conns': 20000,
        'settling_steps': 12,
        'input_coords': [[-1,-1],[1,-1]],
        'output_coords': [[0,1]],
    }
    attributes = {
        'population': lambda a:a.population_size,
        'max_depth': lambda a:a.max_depth,
        'variance_threshold': lambda a:a.variance_threshold,
        'weight_threshold': lambda a:a.weight_threshold,
        'max_weight': lambda a:a.max_weight,
        'band_threshold': lambda a:a.band_threshold,
        'max_sparse_conns': lambda a:a.extended_config.max_sparse_conns,
        'settling_steps': lambda a:a._substrate_activate_time,
        'input_coords': lambda a:a.substrate_input_coords,
        'output_coords': lambda a:a.substrate_output_coords,
    }
    for field, value in changed.items():
        altered = {**CONFIG,field:value}
        payload = body(altered)
        assert payload != original, field
        assert attributes[field](interpreted(altered)) != attributes[field](baseline), field
    feedforward = {key:value for key,value in CONFIG.items()
                   if key not in ('band_threshold','max_sparse_conns','settling_steps')}
    feedforward['recurrent'] = False
    ff = body(feedforward)
    assert ff['emr_hyperneat']['recurrence'] == {
        'enabled':False,'allow_hidden_to_hidden':False,'hh_cache_enabled':False}
    interpreted(feedforward)


@pytest.mark.parametrize('change', [
    {'max_connections': 10000},
    {'hh_cache_enabled': True},
    {'dynamic_functions': {'mode':'enabled'}},
    {'execution_mode': 'streaming'},
    {'max_sparse_conns': 32769},
    {'max_depth': 5},
    {'population': 129},
    {'input_coords': [[-1,-1],[-1,-1]]},
    {'variance_threshold': float('nan')},
    {'schema': True},
])
def test_a0_rejects_unsupported_or_invalid_options(change):
    with pytest.raises(ValueError):
        settings({**CONFIG,**change})


def test_a0_requires_measured_base_manifest():
    with pytest.raises(ValueError,match='measured Q2'):
        translate(CONFIG,{**BASE,'population':64})
