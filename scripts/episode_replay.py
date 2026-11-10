"""Closed-loop portable CartPole replay, without evolution-library imports."""
import argparse
import sys
from neuroevo.backend import initialize
hardware=initialize()
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.artifacts import reload,write_json
from neuroevo.planning import to_graph
from neuroevo.reference import step as reference
from neuroevo.runtime import compiled_step
from neuroevo.environments import cartpole
from neuroevo.randomness import key

p=argparse.ArgumentParser()
p.add_argument("phenotype")
p.add_argument("--seed",type=int,default=0)
p.add_argument("--output",required=True)
args=p.parse_args()
packed,metadata=reload(args.phenotype)
graph=to_graph(packed,metadata["profile"])
env,params=cartpole()
reset_key,rollout_key=jax.random.split(key(args.seed,"interpolation",0,0))
obs,es=env.reset_env(reset_key,params)
ref_obs,ref_es=env.reset_env(reset_key,params)
state=jnp.zeros(len(packed.node_mask),jnp.float32)
ref_state=np.zeros(len(graph.node_ids),np.float64)
reward_sum=reference_reward=0.
maximum_error=0.
actions=[]
for t in range(500):
    head,state=compiled_step(packed,state,obs,profile=metadata["profile"])
    ref_head,ref_state=reference(graph,ref_state,np.asarray(ref_obs))
    np.testing.assert_allclose(np.asarray(state)[:len(ref_state)],ref_state,atol=1e-4,rtol=1e-4)
    maximum_error=max(maximum_error,float(np.max(np.abs(np.asarray(head)-ref_head))))
    action,ref_action=int(np.argmax(head)),int(np.argmax(ref_head))
    if action != ref_action:
        raise AssertionError("Portable and scalar actions differ")
    step_key=jax.random.fold_in(rollout_key,t)
    obs,es,reward,done,_=env.step_env(step_key,es,action,params)
    ref_obs,ref_es,ref_reward,ref_done,_=env.step_env(step_key,ref_es,ref_action,params)
    np.testing.assert_array_equal(obs,ref_obs)
    assert bool(done)==bool(ref_done)
    reward_sum+=float(reward)
    reference_reward+=float(ref_reward)
    actions.append(action)
    if done:
        break
assert reward_sum==reference_reward
assert "tensorneat" not in sys.modules and "emr_hyperneat" not in sys.modules
write_json(args.output,{"status":"pass","steps":len(actions),"return":reward_sum,
                       "reference_return":reference_reward,"max_head_abs_error":maximum_error,
                       "all_actions_match":True,"candidate_id":metadata.get("candidate_id"),
                       "backend_imports":False,"hardware":hardware})
