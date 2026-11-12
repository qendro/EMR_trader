"""Inspect complete held-out trajectories from exported P0 champions."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.artifacts import reload as reload_artifact
from neuroevo.environments import cartpole
from neuroevo.evaluation import cartpole_observation
from neuroevo.randomness import key
from neuroevo.runtime import step


def trajectory(plan, profile, seed, episode, horizon=500):
    env,params=cartpole()
    root=key(seed,'interpolation',0,episode)
    reset_key,rollout_key=jax.random.split(root)
    observation,state=env.reset_env(reset_key,params)

    @jax.jit
    def execute(initial_observation,initial_state):
        initial=(jnp.zeros(plan.node_mask.shape,jnp.float32),initial_state,
                 initial_observation,jnp.bool_(True),jnp.float32(0))
        def body(carry,time):
            policy,es,obs,active,total=carry
            visible=cartpole_observation(obs,'position-angle')
            logits,new_policy=step(plan,policy,visible,profile=profile)
            action=jnp.argmax(logits).astype(jnp.int32)
            next_obs,next_state,reward,done,_=env.step_env(
                jax.random.fold_in(rollout_key,time),es,action,params)
            keep=lambda new,old:jnp.where(active,new,old)
            result=(obs,visible,logits,action,jnp.where(active,reward,0),active)
            return (keep(new_policy,policy),jax.tree.map(keep,next_state,es),
                    keep(next_obs,obs),active & ~done,
                    total+jnp.where(active,reward,0)),result
        return jax.lax.scan(body,initial,jnp.arange(horizon))

    final,rows=execute(observation,state)
    obs,visible,logits,actions,rewards,active=(np.asarray(x) for x in rows)
    length=int(active.sum())
    return {'seed':seed,'episode':episode,'return':float(np.asarray(final[4])),
            'length':length,'steps':[{'time':t,'state':obs[t].tolist(),
                'policy_observation':visible[t].tolist(),
                'logits':logits[t].tolist(),'action':int(actions[t]),
                'reward':float(rewards[t])} for t in range(length)]}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episodes',type=int,default=3)
    args=parser.parse_args()
    root=Path(args.experiment_root)
    config=json.loads((root/'experiment.json').read_text())['config']
    if config['task']!='cartpole-partial' or args.episodes<1 or \
            args.episodes>config['heldout_episodes']:
        raise ValueError('Invalid P0 trajectory request')
    all_rows=[]
    for seed in config['seeds']:
        directory=root/f'seed-{seed}'
        plan,metadata=reload_artifact(directory/'champion.npz')
        scores=json.loads((directory/'selection.json').read_text())['episode_scores']
        for episode in range(args.episodes):
            row=trajectory(plan,metadata['profile'],seed,episode,config['horizon'])
            if not np.isclose(row['return'],scores[episode],atol=1e-5):
                raise AssertionError('Portable trajectory disagrees with held-out score')
            all_rows.append(row)
    Path(args.output).write_text(json.dumps({'task':config['task'],
        'episodes_per_seed':args.episodes,'trajectories':all_rows},indent=2)+'\n')
