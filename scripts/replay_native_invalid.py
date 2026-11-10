"""Replay a saved raw-upstream invalid genome; this is not platform inference."""
import argparse
import json
from pathlib import Path
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
from tensorneat.genome import DefaultGenome
from tensorneat.genome.gene import DefaultNode,DefaultConn
from tensorneat.common import State,ACT,AGG
from neuroevo.artifacts import write_json

p=argparse.ArgumentParser()
p.add_argument('fixture')
p.add_argument('--output',required=True)
args=p.parse_args()
with np.load(args.fixture,allow_pickle=False) as data:
    nodes,conns,inputs,expected=(data[name] for name in ('nodes','conns','inputs','outputs'))
genome=DefaultGenome(4,2,max_nodes=50,max_conns=100,
    node_gene=DefaultNode(activation_options=ACT.tanh,aggregation_options=AGG.sum),
    conn_gene=DefaultConn(),output_transform=ACT.sigmoid)
state=State(randkey=jax.random.PRNGKey(700))
transformed=genome.transform(state,jnp.asarray(nodes),jnp.asarray(conns))
actual=np.asarray(jax.jit(jax.vmap(genome.forward,in_axes=(None,None,0)))(state,transformed,jnp.asarray(inputs)))
np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5,equal_nan=True)
assert np.any(~np.isfinite(actual))
active_nodes=nodes[~np.isnan(nodes[:,0])]
active_conns=conns[~np.isnan(conns[:,0])]
isolated=[int(n[0]) for n in active_nodes if n[0]>=4 and not np.any(active_conns[:,1]==n[0])]
assert isolated,'Expected an isolated non-input node in the saved reproducer'
write_json(args.output,{'status':'reproduced-upstream-nonfinite-output','hardware':hardware,
    'samples':len(inputs),'nonfinite_outputs':int((~np.isfinite(actual)).sum()),
    'nonfinite_output_columns':np.flatnonzero((~np.isfinite(actual)).any(axis=0)).tolist(),
    'isolated_noninput_ids':isolated,
    'active_attributes_finite':bool(np.isfinite(active_nodes).all() and np.isfinite(active_conns).all()),
    'interpretation':'native empty-input aggregation convention; platform isolated-bias contract differs'})
