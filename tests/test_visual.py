import json
import jax.numpy as jnp
import numpy as np
import pytest
from neuroevo.visual import visual_scenarios, input_coordinates, TRAIN_POSITIONS, HELDOUT_POSITIONS
from neuroevo.fixed import FixedDecoder, settings
from neuroevo.runner import load_config
from neuroevo.evaluation import visual_chunk
from neuroevo.contracts import stack_plans
from neuroevo.substrate import blocks
from neuroevo.runtime import compiled_step


def test_visual_protocol_geometry_balance_and_independent_noise():
    assert set(TRAIN_POSITIONS).isdisjoint(HELDOUT_POSITIONS)
    assert len(input_coordinates()) == 25 and len(set(map(tuple, input_coordinates()))) == 25
    for namespace, episodes, positions in [('train',40,TRAIN_POSITIONS),('validation',40,TRAIN_POSITIONS),
                                            ('interpolation',160,HELDOUT_POSITIONS)]:
        obs, labels, sites = visual_scenarios(700,namespace,0,episodes)
        obs, labels, sites = map(np.asarray,(obs,labels,sites))
        for site in range(len(positions)):
            for label in (-1,1):
                chosen=(sites==site)&(labels==label)
                assert chosen.sum()==episodes//(2*len(positions))
                y,x=positions[site]
                signal=np.zeros((5,5),bool)
                for offset in range(3):
                    signal[y+(offset if label==1 else 0),x+(offset if label==-1 else 0)]=True
                assert np.all(obs[chosen][:,signal.ravel()]>.85)
                assert np.all(np.abs(obs[chosen][:,~signal.ravel()])<=.15)
    a,_,_=visual_scenarios(700,'train',0,40)
    b,_,_=visual_scenarios(700,'train',0,40)
    c,_,_=visual_scenarios(700,'train',1,40)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    with pytest.raises(ValueError,match='balance'):
        visual_scenarios(700,'train',0,39)


def test_visual_config_decoder_coordinates_and_interface(tmp_path):
    c=load_config('configs/h2-visual-fixed-diagnostic-v1.json')
    assert c['decoder']['input_coordinates']==input_coordinates()
    assert load_config('configs/h2-visual-direct-diagnostic-v1.json')['adapter']['profile']=='dag'
    assert load_config('configs/h2-visual-b0-diagnostic-v1.json')['adapter']['profile']=='layered'
    class Query:
        def prepare(self,nodes,conns): return None
        def __call__(self,prepared,rows): return np.tanh(rows[:,0]+rows[:,1]-rows[:,2])
    decoder=FixedDecoder(Query(),c['decoder'])
    np.testing.assert_array_equal(decoder.input_coordinates[:-1],np.asarray(input_coordinates(),np.float32))
    np.testing.assert_array_equal(decoder.input_coordinates[-1],[0.,-2.])
    phenotype=decoder.decode(None,None)
    assert phenotype.w_in.shape==(25,12)
    bad=dict(c['decoder'],bias_coordinate=[-1.,-1.])
    with pytest.raises(ValueError,match='coordinates'):
        settings(bad)
    bad=dict(c,decoder=dict(c['decoder'],input_coordinates=list(reversed(input_coordinates()))))
    path=tmp_path/'bad.json'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='frozen'):
        load_config(path)


def test_visual_population_uses_one_step_and_matches_scalar():
    weights=np.zeros((25,2),np.float32)
    weights[0,0]=1.
    weights[10,1]=1.
    one=blocks(weights,np.zeros((2,2),np.float32),np.array([[1.],[-1.]],np.float32),
               np.ones_like(weights,bool),np.zeros((2,2),bool),np.ones((2,1),bool))
    two=one._replace(w_out=-one.w_out)
    plans=stack_plans([one,two])
    obs,labels,_=visual_scenarios(700,'train',0,40)
    scores,valid=visual_chunk(plans,obs,labels,profile='layered')
    assert np.asarray(valid).all()
    for p,actual in zip((one,two),np.asarray(scores)):
        expected=[]
        for x,label in zip(np.asarray(obs),np.asarray(labels)):
            head,_=compiled_step(p,jnp.zeros(28),x,profile='layered')
            expected.append((1 if head[0]>=0 else -1)==label)
        np.testing.assert_array_equal(actual,expected)
    before=visual_chunk._cache_size()
    other, _=visual_chunk(plans,obs+.001,labels,profile='layered')
    assert visual_chunk._cache_size()==before
    assert np.asarray(other).shape==(2,40)
