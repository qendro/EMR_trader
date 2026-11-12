"""A0: narrow, explicit translation into the pinned EMR public config API.

The Q2 manifest supplies pinned CPPN operators and unexposed defaults. A0
exposes only options that this initial adaptive envelope can translate. A1
will add phenotype extraction and parity; this module does not run evolution.
"""
import hashlib
import json

import numpy as np


Q2_CANONICAL_SHA256 = 'f5183586324a98081c89736dc43ebdc119343aba7c90024d2da86461522e8bd0'
COMMON = {'schema', 'kind', 'population', 'input_coords', 'output_coords',
          'max_depth', 'variance_threshold', 'weight_threshold', 'max_weight',
          'recurrent'}
RECURRENT = {'band_threshold', 'max_sparse_conns', 'settling_steps'}


def _number(value, name, lower, upper, *, upper_open=False):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f'Invalid adaptive {name}')
    if value < lower or (value >= upper if upper_open else value > upper):
        raise ValueError(f'Adaptive {name} exceeds the declared envelope')
    return float(value)


def _coordinates(value, name, maximum, y):
    if (type(value) is not list or not 1 <= len(value) <= maximum or
            any(type(row) is not list or len(row) != 2 for row in value)):
        raise ValueError(f'Invalid adaptive {name}')
    try:
        points = np.asarray(value, np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Invalid adaptive {name}') from error
    if (not np.isfinite(points).all() or np.any(np.abs(points) > 1) or
            not np.all(points[:, 1] == y) or
            len(np.unique(points, axis=0)) != len(points)):
        raise ValueError(f'Invalid adaptive {name}')
    return points.tolist()


def settings(config):
    if type(config) is not dict or type(config.get('schema')) is not int or \
            config['schema'] != 1 or \
            config.get('kind') != 'adaptive-emr-v1' or type(config.get('recurrent')) is not bool:
        raise ValueError('Unsupported adaptive decoder schema')
    expected = COMMON | (RECURRENT if config['recurrent'] else set())
    if set(config) != expected:
        raise ValueError('Unsupported adaptive option or missing required option')
    for name, maximum in (('population', 128), ('max_depth', 4)):
        if type(config[name]) is not int or not 1 <= config[name] <= maximum:
            raise ValueError(f'Invalid adaptive {name}')
    checked = dict(config)
    checked['input_coords'] = _coordinates(config['input_coords'], 'input coordinates', 4, -1)
    checked['output_coords'] = _coordinates(config['output_coords'], 'output coordinates', 2, 1)
    checked['variance_threshold'] = _number(config['variance_threshold'],
                                            'variance threshold', 0, 1, upper_open=True)
    checked['weight_threshold'] = _number(config['weight_threshold'],
                                          'weight threshold', 0, 1, upper_open=True)
    checked['max_weight'] = _number(config['max_weight'], 'max weight', 0.000001, 8)
    if config['recurrent']:
        checked['band_threshold'] = _number(config['band_threshold'],
                                            'band threshold', 0, 1, upper_open=True)
        for name, maximum in (('max_sparse_conns', 32768), ('settling_steps', 17)):
            if type(config[name]) is not int or not 1 <= config[name] <= maximum:
                raise ValueError(f'Invalid adaptive {name}')
    return checked


def translate(config, q2_manifest):
    """Return exactly the public `EMRHyperNEAT.create_config` input.

    Unknown and ineffective features, including the legacy `max_connections`
    knob, are rejected by ``settings``. They cannot silently enter a run.
    """
    c = settings(config)
    canonical = json.dumps(q2_manifest, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != Q2_CANONICAL_SHA256:
        raise ValueError('A0 requires the measured Q2 base manifest')
    discovery = {key: value for key, value in q2_manifest['discovery'].items()
                 if key != 'execution_mode'}
    discovery.update(max_depth=c['max_depth'],
                     variance_threshold=c['variance_threshold'],
                     weight_threshold=c['weight_threshold'],
                     max_weight=c['max_weight'])
    if c['recurrent']:
        discovery['band_threshold'] = c['band_threshold']
        recurrence = dict(q2_manifest['recurrence'])
        recurrence.pop('max_connections', None)  # Legacy no-op, never exposed by A0.
        recurrence.update(max_sparse_conns=c['max_sparse_conns'],
                          activate_time=c['settling_steps'])
    else:
        recurrence = {'enabled': False, 'allow_hidden_to_hidden': False,
                      'hh_cache_enabled': False}
    discovery.update(execution={'mode': 'standard'},
                     dynamic_functions={'mode': 'disabled'},
                     aggregation={'mode': 'disabled'}, recurrence=recurrence,
                     verbose=False)
    substrate = {'input_coords': c['input_coords'],
                 'output_coords': c['output_coords'],
                 'activate_time': c['settling_steps'] if c['recurrent'] else
                                  q2_manifest['substrate']['settling_steps'],
                 'hidden_activation': 'tanh', 'output_activation': 'sigmoid'}
    cppn = q2_manifest['cppn']
    return {'algorithm_params': {'emrhyperneat': {
        'population_size': c['population'], 'emr_hyperneat': discovery,
        'substrate': substrate, 'neat_mutation': dict(cppn['mutation']),
        'neat_species': dict(cppn['species']),
        'neat_selection': dict(cppn['selection'])}}}
