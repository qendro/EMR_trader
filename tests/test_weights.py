import json
import jax
import numpy as np
import pytest
from neuroevo.adapters.weights import WeightAdapter,settings
from neuroevo.runner import load_config,run,provenance
from neuroevo.artifacts import save_checkpoint,load_checkpoint,reload


def adapter(task='memory',**changes):
    c=load_config(f'configs/b0-{task}-smoke-v1.json')['adapter']
    return WeightAdapter({**c,**changes})


def test_fixed_gene_layout_topology_and_bounds():
    a=adapter()
    assert a.parameters==32
    weights=np.arange(32,dtype=np.float32)/10
    p=a.decode(weights)
    np.testing.assert_array_equal(p.w_in,weights[:8].reshape(2,4))
    np.testing.assert_array_equal(p.w_delay,weights[8:24].reshape(4,4))
    np.testing.assert_array_equal(p.w_out,weights[24:28].reshape(4,1))
    np.testing.assert_array_equal(p.bias_hidden,weights[28:])
    assert p.mask_in.all() and p.mask_delay.all() and p.mask_out.all() and np.all(p.bias_output==0)
    ff=adapter(recurrent=False)
    assert ff.parameters==16
    p=ff.decode(np.ones(16,np.float32))
    assert not p.mask_delay.any() and np.all(p.w_delay==0)
    with pytest.raises(ValueError,match='bounds'): a.decode(np.full(32,5.01,np.float32))
    with pytest.raises(ValueError,match='genome'): a.decode(np.full(32,np.nan,np.float32))


def test_truncation_maximizes_elites_and_association():
    a=adapter(mutation_rate=0.,survival_fraction=.25)
    state=a.initialize(3)
    batch=a.ask(state,'selection')
    assert not batch.weights.flags.writeable
    with pytest.raises(ValueError,match='misordered'): a.tell(state,batch,batch.ids[::-1],np.arange(8))
    with pytest.raises(ValueError,match='All candidates'): a.tell(state,batch,batch.ids,np.full(8,np.nan))
    with pytest.raises(ValueError,match='shape'): a.tell(state,batch,batch.ids,np.arange(7))
    result,_=a.tell(state,batch,batch.ids,np.arange(8,dtype=np.float32))
    np.testing.assert_array_equal(result.weights[:2],batch.weights[[7,6]])
    assert all(any(np.array_equal(child,parent) for parent in batch.weights[[7,6]]) for child in np.asarray(result.weights)[2:])
    with pytest.raises(ValueError,match='Stale'): a.tell(state,batch,batch.ids,np.arange(8))
    # One valid candidate supplies every elite and parent, excluding NaNs/inf.
    batch=a.ask(result,'selection')
    score=np.full(8,np.nan,np.float32); score[3]=1.; score[5]=np.inf
    result,_=a.tell(result,batch,batch.ids,score)
    np.testing.assert_array_equal(result.weights,np.repeat(batch.weights[3:4],8,axis=0))


def test_gaussian_mutation_changes_parameters_preserves_topology_and_caps():
    a=adapter(mutation_rate=1.,mutation_std=2.)
    state=a.initialize(5)
    batch=a.ask(state,'mutation')
    result,_=a.tell(state,batch,batch.ids,np.arange(8))
    np.testing.assert_array_equal(result.weights[:2],batch.weights[[7,6]])
    assert np.isfinite(result.weights).all() and np.max(np.abs(result.weights))<=5
    assert not any(np.array_equal(result.weights[2],p) for p in batch.weights)
    before,after=a.decode(batch.weights[0]),a.decode(np.asarray(result.weights[2]))
    for name in ('mask_in','mask_delay','mask_out','node_ids','inputs','outputs'):
        np.testing.assert_array_equal(getattr(before,name),getattr(after,name))


def test_weight_state_complete_40_100_resume(tmp_path):
    a=adapter()
    state=a.initialize(71)
    source=provenance()
    for generation in range(100):
        batch=a.ask(state,'resume')
        state,_=a.tell(state,batch,batch.ids,np.arange(8,dtype=np.float32))
        if generation==39: save_checkpoint(tmp_path/'state.bin',state,a.config,source,[])
    b=adapter()
    resumed=load_checkpoint(tmp_path/'state.bin',b.config,source)['state']
    for generation in range(40,100):
        batch=b.ask(resumed,'resume')
        resumed,_=b.tell(resumed,batch,batch.ids,np.arange(8,dtype=np.float32))
    for x,y in zip(jax.tree.leaves(state),jax.tree.leaves(resumed)): np.testing.assert_array_equal(x,y)
    assert a._tell._cache_size()==1


@pytest.mark.parametrize('task',['cartpole','memory'])
def test_weight_controller_archive_resume_reconstruction(task,tmp_path):
    c=load_config(f'configs/b0-{task}-smoke-v1.json')
    run(c,900,tmp_path,{},stop_after=2)
    run(c,900,tmp_path,{},resume=True)
    state=load_checkpoint(tmp_path/'checkpoint.bin',{**c,'run_seed':900},provenance())['state']
    assert int(state.generation)==4
    history=json.loads((tmp_path/'history.json').read_text())
    assert [r['generation'] for r in history]==list(range(4))
    assert all(r['species'] is None and r['stagnation_removed']==[] for r in history)
    p,meta=reload(tmp_path/'champion.npz')
    assert meta['genotype_encoding']=='flat-role-weights-v1' and meta['baseline']==c['adapter']
    with np.load(tmp_path/'champion-genes.npz',allow_pickle=False) as data:
        assert data.files==['weights']
        restored=WeightAdapter(c['adapter']).decode(data['weights'])
    for x,y in zip(p,restored): np.testing.assert_array_equal(x,y)
    saved=load_checkpoint(tmp_path/'checkpoint.bin',{**c,'run_seed':900},provenance())
    assert saved['champion']['id'].startswith('seed-900:3:')
    with np.load(tmp_path/'evaluated-winner-genes.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['weights'],saved['champion']['weights'])


def test_weight_streaming_history_resume_discards_uncheckpointed_tail(tmp_path):
    c={**load_config('configs/b0-memory-smoke-v1.json'),'history_policy':'jsonl'}
    c['checkpoint_every']=2
    run(c,900,tmp_path,{},stop_after=2)
    path=tmp_path/'history.jsonl'
    with path.open('ab') as stream:
        stream.write(b'{"generation":999}\n')
    run(c,900,tmp_path,{},resume=True)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    final=json.loads((tmp_path/'history.json').read_text())
    assert [r['generation'] for r in rows]==list(range(4))
    assert [r['generation'] for r in final]==list(range(4))
    assert all(a['best']==b['best'] for a,b in zip(rows,final))
    assert all('total_seconds' in r for r in rows)


def test_weight_elapsed_stop_selects_last_evaluated_generation(tmp_path):
    c={**load_config('configs/b0-memory-smoke-v1.json'),
       'elapsed_stop_seconds':1e-9,'history_policy':'jsonl'}
    summary,history=run(c,900,tmp_path,{})
    assert summary['stop_reason']=='elapsed'
    assert summary['generations']==1
    assert len(history)==1
    selection=json.loads((tmp_path/'selection.json').read_text())
    assert selection['candidate_id'] in selection['finalist_ids']
    assert (tmp_path/'checkpoint.bin').exists()


@pytest.mark.parametrize('changes',[{'mutation_rate':1.1},{'survival_fraction':0},
    {'profile':'synchronous'},{'elite_count':8},{'recurrent':1},{'hidden':1024},{'weight_bound':6}])
def test_weight_configuration_rejects_unsupported(changes):
    with pytest.raises(ValueError): adapter(**changes)
