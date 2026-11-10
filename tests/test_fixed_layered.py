import json
import math
from dataclasses import replace
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neuroevo.contracts import Graph,stack_plans
from neuroevo.planning import plan,to_graph
from neuroevo.runtime import compiled_step
from neuroevo.reference import step as scalar
from neuroevo.substrate import blocks,from_graph,reference_step
from neuroevo.artifacts import export,reload
from neuroevo.fixed import FixedDecoder,transform,settings
from neuroevo.adapters.cppn import CPPNQuery,query_scores
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.runner import load_config
from neuroevo.evaluation import memory_chunk,memory_initial
from neuroevo.evaluation import make_cartpole_rollout,evaluate_population
from neuroevo.environments import cartpole
from neuroevo.randomness import key
from neuroevo.environments import memory_scenarios


def decoder_config(slab=7,recurrent=True):
    return dict(schema=1,kind='fixed-coordinate-v1',inputs=2,hidden=3,outputs=1,
                recurrent=recurrent,threshold=.3,max_weight=5.,query_slab=slab)


def cppn():
    c=load_config('configs/m1-smoke-v1.json')['adapter']
    c={**c,'inputs':4,'outputs':1,'nodes':32,'edges':128,'profile':'dag'}
    return TensorNEATAdapter(c)


def sample_blocks():
    rng=np.random.default_rng(5)
    wi=rng.uniform(-.3,.3,(2,3)).astype(np.float32)
    wd=rng.uniform(-.1,.1,(3,3)).astype(np.float32)
    wo=rng.uniform(-.3,.3,(3,1)).astype(np.float32)
    return blocks(wi,wd,wo,np.ones(wi.shape,bool),np.eye(3,dtype=bool),np.ones(wo.shape,bool),
                  np.array([.1,-.1,.05],np.float32))


def test_layered_mixed_delays_two_hidden_export_and_input_history(tmp_path):
    g=Graph((0,1,2,3),(0,),(3,),np.array([0,.1,-.1,.2],np.float32),np.ones(4,np.float32),
        np.array([0,1,1,0]),((0,1,1.,0),(1,2,.5,0),(2,3,2.,0),
        (2,1,.75,1),(1,2,-.15,1),(0,1,.2,1)),'layered')
    packed=plan(g)
    export(tmp_path/'mixed.npz',packed,'layered',{'action':'sign-positive-tie'})
    restored,_=reload(tmp_path/'mixed.npz')
    assert restored.mask_zero[1,2] and restored.mask_delay[1,2]
    state=jnp.zeros(32)
    expected=np.zeros(4,np.float64)
    for t in range(128):
        obs=np.array([np.sin(t)*.2],np.float32)
        target,expected=scalar(g,expected,obs)
        output,state=compiled_step(restored,state,obs,profile='layered')
        np.testing.assert_allclose(output,target,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(np.asarray(state)[:4],expected,atol=1e-4,rtol=1e-4)
    dag=replace(g,profile='dag',edges=tuple(e for e in g.edges if not e[3]))
    a,_=compiled_step(plan(dag),jnp.zeros(32),jnp.array([.4]),profile='dag')
    b,_=compiled_step(plan(replace(dag,profile='layered')),jnp.zeros(32),jnp.array([.4]),profile='layered')
    np.testing.assert_array_equal(a,b)


def test_role_blocks_scalar_graph_and_lossless_conversion(tmp_path):
    p=sample_blocks()
    graph=to_graph(p,'layered')
    restored=from_graph(graph)
    for a,b in zip(p,restored): np.testing.assert_array_equal(a,b)
    export(tmp_path/'blocks.npz',p,'layered',{'action':'sign-positive-tie'})
    imported,meta=reload(tmp_path/'blocks.npz')
    assert meta['encoding']=='role-blocks-v1'
    state=jnp.zeros(len(p.node_mask))
    rs=np.zeros(len(p.node_mask),np.float64)
    for t in range(128):
        obs=np.array([np.sin(t)*.2,np.cos(t)*.1],np.float32)
        target,rs=scalar(graph,rs,obs)
        ry,rstate=reference_step(p,np.asarray(state),obs)
        y,state=compiled_step(imported,state,obs,profile='layered')
        np.testing.assert_allclose(y,target,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(np.asarray(state),rs,atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(y,ry,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(state,rstate,atol=1e-6,rtol=1e-5)
    # Immediate hidden edges cannot be silently folded into the delayed role block.
    with pytest.raises(ValueError,match='outside'):
        from_graph(replace(graph,edges=graph.edges+((2,3,.1,0),)))


def test_role_blocks_portable_fresh_process_and_strict_import(tmp_path):
    p=sample_blocks()
    path=tmp_path/'policy.npz'
    export(path,p,'layered',{'action':'sign-positive-tie'})
    script='''
import sys
import jax.numpy as jnp
from neuroevo.artifacts import reload
from neuroevo.runtime import compiled_step
p,m=reload(sys.argv[1])
head,state=compiled_step(p,jnp.zeros(len(p.node_mask)),jnp.zeros(2),profile='layered')
assert head.shape==(1,) and state.shape==(6,)
assert 'tensorneat' not in sys.modules and 'emr_hyperneat' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',script,str(path)],check=True)
    with pytest.raises(ValueError,match='layered'):
        export(path,p,'dag',{'action':'sign-positive-tie'})
    with np.load(path,allow_pickle=False) as f:
        arrays={key:f[key] for key in f.files}
    arrays['w_delay']=arrays['w_delay'].astype(np.float64)
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='array'):
        reload(path)


def test_role_memory_chunk_continuation_reset_isolation_and_dynamic_values():
    p=blocks(np.array([[1.],[0.]],np.float32),np.array([[2.]],np.float32),np.ones((1,1),np.float32),
             np.array([[True],[False]]),np.ones((1,1),bool),np.ones((1,1),bool))
    opposite=p._replace(w_in=-p.w_in)
    packed=stack_plans([p,opposite])
    obs,cues,delays=memory_scenarios(5,'train')
    initial=memory_initial(packed,len(cues))
    whole=memory_chunk(packed,initial,obs,cues,delays,jnp.arange(17),profile='layered')
    partial=memory_chunk(packed,initial,obs[:,:4],cues,delays,jnp.arange(4),profile='layered')
    partial=memory_chunk(packed,partial,obs[:,4:],cues,delays,jnp.arange(4,17),profile='layered')
    for a,b in zip(whole,partial): np.testing.assert_array_equal(a,b)
    assert np.mean(whole.correct[0])==1 and np.mean(whole.correct[1])==0
    reset=memory_chunk(packed,initial,obs,cues,delays,jnp.arange(17),profile='layered',reset_each_step=True)
    assert np.mean(reset.correct[0])<.8
    perm=np.random.default_rng(1).permutation(len(cues))
    permuted=memory_chunk(packed,initial,obs[perm],cues[perm],delays[perm],jnp.arange(17),profile='layered')
    np.testing.assert_array_equal(whole.correct[:,perm],permuted.correct)
    compiled_step(p,jnp.zeros(4),jnp.zeros(2),profile='layered')
    count=compiled_step._cache_size()
    compiled_step(opposite,jnp.zeros(4),jnp.ones(2),profile='layered')
    assert compiled_step._cache_size()==count


def test_role_masks_and_overflow_containment():
    p=sample_blocks()
    bad=p.w_delay.copy()
    bad[~p.mask_delay]=np.nan
    _,state=compiled_step(p._replace(w_delay=bad),jnp.zeros(6),jnp.ones(2),profile='layered')
    assert np.isfinite(state).all()
    huge=np.full((2,3),np.finfo(np.float32).max,np.float32)
    _,state=compiled_step(p._replace(w_in=huge),jnp.zeros(6),jnp.full(2,2.),profile='layered')
    assert not np.isfinite(state).all()
    with pytest.raises(ValueError,match='Nonfinite'):
        blocks(np.full((1,1),np.inf),np.zeros((1,1)),np.zeros((1,1)),
               np.ones((1,1),bool),np.zeros((1,1),bool),np.zeros((1,1),bool))


def test_role_cartpole_population_tail_and_shared_data():
    rng=np.random.default_rng(4)
    wi=rng.uniform(-.1,.1,(4,3)).astype(np.float32)
    wd=np.eye(3,dtype=np.float32)*.1
    wo=rng.uniform(-.2,.2,(3,2)).astype(np.float32)
    p=blocks(wi,wd,wo,np.ones(wi.shape,bool),np.eye(3,dtype=bool),np.ones(wo.shape,bool))
    candidates=[p._replace(bias_output=np.array([j/1000,0],np.float32)) for j in range(35)]
    env,params=cartpole()
    evaluate=make_cartpole_rollout(env,params,32,'layered')
    keys=jnp.stack([key(1,'train',0,e) for e in range(2)])
    shared=jnp.zeros((32,4),jnp.float32)
    one=evaluate_population(candidates,evaluate,keys,shared,1)
    many=evaluate_population(candidates,evaluate,keys,shared,32)
    for a,b in zip(one,many): np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    count=evaluate._cache_size()
    evaluate_population(candidates,evaluate,keys,shared+.001,32)
    assert evaluate._cache_size()==count


def test_fixed_controller_archive_and_resume(tmp_path):
    from neuroevo.runner import run
    from neuroevo.artifacts import load_checkpoint
    from neuroevo.runner import provenance
    c=load_config('configs/h1-fixed-cartpole-smoke-v1.json')
    run(c,900,tmp_path,{},stop_after=2)
    run(c,900,tmp_path,{},resume=True)
    checkpoint=load_checkpoint(tmp_path/'checkpoint.bin',{**c,'run_seed':900},provenance())
    assert int(checkpoint['state'].generation)==4
    rows=json.loads((tmp_path/'history.json').read_text())
    assert [r['generation'] for r in rows]==list(range(4))
    assert all(r['phenotype_nodes_max']==14 and r['cppn_queries_total']>0 for r in rows)
    p,metadata=reload(tmp_path/'champion.npz')
    assert metadata['profile']=='layered' and metadata['decoder']==c['decoder']
    assert p.w_delay.shape==(8,8)
    with np.load(tmp_path/'champion-genes.npz',allow_pickle=False) as data:
        a=TensorNEATAdapter(c['adapter'])
        rebuilt=FixedDecoder(CPPNQuery(a),c['decoder']).decode(data['nodes'],data['conns'])
    for x,y in zip(p,rebuilt): np.testing.assert_array_equal(x,y)


def test_fixed_query_scalar_coordinates_weights_and_slabs():
    a=cppn()
    batch=a.ask(a.initialize(33),'query')
    q=CPPNQuery(a)
    d=FixedDecoder(q,decoder_config())
    p=d.decode(batch.nodes[0],batch.conns[0])
    graph=a.decode(batch.nodes[0],batch.conns[0])
    xs=[(-1.,-1.),(0.,-1.),(1.,-1.)]
    hs=[(-1.,0.),(0.,0.),(1.,0.)]
    ys=[(0.,1.)]
    def expected(sources,targets):
        weights=[]
        for source in sources:
            row=[]
            for target in targets:
                raw,_=scalar(graph,np.zeros(len(graph.node_ids)),np.array([*source,*target],np.float32))
                v=math.tanh(raw[0])
                row.append(math.copysign((abs(v)-.3)/.7*5,v) if abs(v)>.3 else 0.)
            weights.append(row)
        return np.array(weights,np.float64)
    wi=expected(xs,hs)
    np.testing.assert_allclose(p.w_in,wi[:-1],atol=2e-6,rtol=1e-5)
    np.testing.assert_allclose(p.bias_hidden,wi[-1],atol=2e-6,rtol=1e-5)
    np.testing.assert_allclose(p.w_delay,expected(hs,hs),atol=2e-6,rtol=1e-5)
    np.testing.assert_allclose(p.w_out,expected(hs,ys),atol=2e-6,rtol=1e-5)
    for slab in (1,64):
        other=FixedDecoder(q,decoder_config(slab)).decode(batch.nodes[0],batch.conns[0])
        for x,y in zip(p,other): np.testing.assert_allclose(x,y,atol=2e-6,rtol=1e-5)
    prepared=q.prepare(batch.nodes[0],batch.conns[0])
    rows=np.zeros((7,4),np.float32)
    q(prepared,rows)
    before=query_scores._cache_size()
    q(q.prepare(batch.nodes[1],batch.conns[1]),rows+.1)
    assert query_scores._cache_size()==before


def test_fixed_feedforward_upstream_compatible_output():
    from tensorneat.algorithm import NEAT
    from tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT
    from tensorneat.algorithm.hyperneat.substrate import MLPSubstrate
    from tensorneat.genome import DefaultGenome
    from tensorneat.genome.gene import DefaultNode
    from tensorneat.common import State,ACT,AGG
    a=cppn()
    genes=a.ask(a.initialize(71),'parity')
    ours=FixedDecoder(CPPNQuery(a),decoder_config(64,False)).decode(genes.nodes[0],genes.conns[0])
    genome=DefaultGenome(4,1,max_nodes=32,max_conns=128,
        node_gene=DefaultNode(activation_options=ACT.tanh,aggregation_options=AGG.sum),output_transform=ACT.tanh)
    native=HyperNEAT(MLPSubstrate([3,3,1]),NEAT(genome,8,verbose=False,**a.config['species']),
                     activation=ACT.tanh,output_transform=None,activate_time=3)
    state=native.setup(State(randkey=jax.random.PRNGKey(71)))
    phenotype=native.transform(state,(jnp.asarray(genes.nodes[0]),jnp.asarray(genes.conns[0])))
    for obs in (np.array([0.,0.],np.float32),np.array([.2,-.1],np.float32)):
        expected=native.forward(state,phenotype,obs)
        actual,_=compiled_step(ours,jnp.zeros(6),obs,profile='layered')
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-5)


def test_fixed_threshold_boundaries_and_capability_rejection():
    t=np.float32(.3)
    scores=np.array([0,t,-t,np.nextafter(t,np.float32(1)),np.nextafter(-t,np.float32(-1)),1,-1])
    w,mask=transform(scores,.3,5.)
    np.testing.assert_array_equal(mask,[False,False,False,True,True,True,True])
    assert w[-2]==5 and w[-1]==-5 and np.all(w[:3]==0)
    for invalid in (np.nan,np.inf,1.01):
        with pytest.raises(ValueError,match='query score'):
            transform([invalid],.3,5.)
    for change in ({'hidden':1024},{'threshold':1.},{'recurrent':'yes'},{'extra_option':True}):
        with pytest.raises(ValueError): settings({**decoder_config(),**change})
