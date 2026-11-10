import json
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.runner import load_config, provenance
from neuroevo.artifacts import save_checkpoint, load_checkpoint, export, reload
from neuroevo.planning import plan
from neuroevo.runtime import compiled_step


def adapter():
    return TensorNEATAdapter(load_config("configs/m1-smoke-v1.json")["adapter"])


def assert_state(a,b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)


def test_upstream_operator_parity_and_association():
    a = adapter()
    state = a.initialize(12)
    batch = a.ask(state,"parity")
    scores = np.arange(8, dtype=np.float32)
    with pytest.raises(ValueError, match="misordered"):
        a.tell(state,batch,batch.ids[::-1],scores)
    with pytest.raises(ValueError, match="All candidates"):
        a.tell(state,batch,batch.ids,np.full(8,np.nan))
    raw = jax.jit(a.algorithm.tell)(state,jnp.asarray(scores))
    actual,_ = a.tell(state,batch,batch.ids,scores)
    assert_state(actual,raw)
    with pytest.raises(ValueError, match="Stale"):
        a.tell(state,batch,batch.ids,scores)
    # Maximizing elite is present unchanged somewhere after reproduction.
    assert any(np.array_equal(batch.nodes[-1], n, equal_nan=True) and
               np.array_equal(batch.conns[-1], c, equal_nan=True)
               for n,c in zip(np.asarray(actual.pop_nodes),np.asarray(actual.pop_conns)))


def test_direct_output_bias_response_and_upstream_parity():
    a = adapter()
    state = a.initialize(18)
    nodes,conns = np.asarray(state.pop_nodes[0]),np.asarray(state.pop_conns[0])
    p = plan(a.decode(nodes,conns))
    for obs in (np.zeros(4,np.float32),np.array([.2,-.1,.05,.3],np.float32)):
        native = a.algorithm.forward(state,a.algorithm.transform(state,(jnp.asarray(nodes),jnp.asarray(conns))),obs)
        ours,_ = compiled_step(p,jnp.zeros(32),obs,profile="dag")
        np.testing.assert_allclose(ours,native,atol=1e-6,rtol=1e-5)
    assert set(p.node_ids[p.node_mask]) == set(nodes[~np.isnan(nodes[:,0]),0])
    for c in conns[~np.isnan(conns[:,0])]:
        lookup = {int(v):i for i,v in enumerate(p.node_ids[p.node_mask])}
        assert p.w_zero[lookup[int(c[0])],lookup[int(c[1])]] == c[2]


def test_complete_cpu_checkpoint_40_100_and_schema(tmp_path):
    a = adapter()
    config, source = a.config, provenance()
    state = a.initialize(19)
    for generation in range(100):
        batch = a.ask(state,"resume")
        state,_ = a.tell(state,batch,batch.ids,np.arange(8,dtype=np.float32))
        if generation == 39:
            save_checkpoint(tmp_path/"state.bin",state,config,source,[],None)
    restored = load_checkpoint(tmp_path/"state.bin",config,source)["state"]
    b = adapter()
    for generation in range(40,100):
        batch = b.ask(restored,"resume")
        restored,_ = b.tell(restored,batch,batch.ids,np.arange(8,dtype=np.float32))
    assert_state(state,restored)
    assert np.max((~np.isnan(np.asarray(state.pop_nodes)[:,:,0])).sum(axis=1)) > 6
    with pytest.raises(ValueError,match="mismatch"):
        load_checkpoint(tmp_path/"state.bin",{**config,"population":9},source)
    raw = bytearray((tmp_path/"state.bin").read_bytes())
    raw[-1] ^= 1
    (tmp_path/"corrupt.bin").write_bytes(raw)
    with pytest.raises(ValueError,match="Corrupt"):
        load_checkpoint(tmp_path/"corrupt.bin",config,source)
    for n,c in zip(np.asarray(state.pop_nodes),np.asarray(state.pop_conns)):
        plan(a.decode(n,c))
    assert a._observe._cache_size() <= 2


def test_export_reloads_in_fresh_process_without_backend(tmp_path):
    a = adapter()
    state = a.initialize(1)
    p = plan(a.decode(state.pop_nodes[0],state.pop_conns[0]))
    export(tmp_path/"phenotype.npz",p,"dag",{"action":"argmax-first-tie"})
    q,meta = reload(tmp_path/"phenotype.npz")
    for x,y in zip(p,q):
        np.testing.assert_array_equal(x,y)
    code = "from neuroevo.cli import replay; replay(__import__('sys').argv[1], {})"
    subprocess.run([sys.executable,"-c",code,str(tmp_path/"phenotype.npz")],check=True)
