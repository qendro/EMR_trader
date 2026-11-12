"""Diagnostic partial-observation finite-difference CartPole controller."""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from neuroevo.environments import cartpole
from neuroevo.randomness import key


def returns(seed, episodes=100, horizon=500):
    env,params=cartpole()
    roots=jnp.stack([key(seed,'interpolation',0,e) for e in range(episodes)])
    tau=jnp.float32(params.tau)

    @jax.jit
    def evaluate(keys):
        def one(root):
            reset_key,rollout_key=jax.random.split(root)
            obs,state=env.reset_env(reset_key,params)
            initial=(state,obs,obs[0],obs[2],jnp.bool_(False),
                     jnp.bool_(True),jnp.float32(0))
            def body(carry,time):
                es,current,old_x,old_theta,has_history,active,total=carry
                x,theta=current[0],current[2]
                xdot=jnp.where(has_history,(x-old_x)/tau,0)
                thetadot=jnp.where(has_history,(theta-old_theta)/tau,0)
                control=theta+jnp.float32(.2)*thetadot + \
                        jnp.float32(.01)*x+jnp.float32(.01)*xdot
                action=(control>0).astype(jnp.int32)
                nxt_obs,nxt,reward,done,_=env.step_env(
                    jax.random.fold_in(rollout_key,time),es,action,params)
                choose=lambda a,b:jnp.where(active,a,b)
                return (jax.tree.map(choose,nxt,es),choose(nxt_obs,current),
                        choose(x,old_x),choose(theta,old_theta),
                        has_history|active,active & ~done,
                        total+jnp.where(active,reward,0)),None
            return jax.lax.scan(body,initial,jnp.arange(horizon))[0][-1]
        return jax.vmap(one)(keys)
    return np.asarray(evaluate(roots),np.float32)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    samples={str(seed):returns(seed).tolist() for seed in range(5)}
    result={'label':'known-fixed-controller-diagnostic-not-evolved',
            'observations':'position-angle-plus-one-step-finite-difference',
            'seeds':samples,'means':{seed:float(np.mean(values))
                                    for seed,values in samples.items()}}
    with open(args.output,'w') as stream:
        json.dump(result,stream,indent=2)
    print(result['means'])
