import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.environments import cartpole
from neuroevo.evaluation import make_cartpole_rollout,evaluate_population
from neuroevo.adapters.tensorneat import TensorNEATAdapter
from neuroevo.runner import load_config
from neuroevo.planning import plan
from neuroevo.randomness import key


def test_cartpole_scalar_microbatch_tails_shared_values():
    cfg = load_config("configs/m1-smoke-v1.json")["adapter"]
    a = TensorNEATAdapter(cfg)
    s = a.initialize(4)
    plans = [plan(a.decode(s.pop_nodes[i%8],s.pop_conns[i%8])) for i in range(35)]
    env,params = cartpole()
    f = make_cartpole_rollout(env,params,horizon=32)
    keys = jnp.stack([key(4,"train",0,i) for i in range(2)])
    shared = jnp.zeros((32,4),jnp.float32)
    one = evaluate_population(plans,f,keys,shared,1)
    batch = evaluate_population(plans,f,keys,shared,32)
    for x,y in zip(one,batch):
        np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
    permutation = np.random.default_rng(3).permutation(35)
    reordered = evaluate_population([plans[i] for i in permutation],f,keys[::-1],shared,32)
    for x,y in zip(batch,reordered):
        np.testing.assert_allclose(x[permutation,::-1],y,atol=1e-6,rtol=1e-5)
    count = f._cache_size()
    evaluate_population(plans,f,keys,shared.at[31].set(.1),32)
    assert f._cache_size() == count
    assert np.all(batch[0] == batch[1])  # reward exactly once per transition
    assert np.all(batch[1] <= 32)
    assert np.all(batch[2] ^ batch[3])


def test_terminal_reward_once_and_horizon_truncation():
    # Same Gymnax state, force physical termination in one step.
    cfg=load_config("configs/m1-smoke-v1.json")["adapter"]
    a=TensorNEATAdapter(cfg)
    s=a.initialize(6)
    p=plan(a.decode(s.pop_nodes[0],s.pop_conns[0]))
    env,params=cartpole()
    keys=jnp.stack([key(6,"train")])
    class OneStep:
        def reset_env(self, reset_key, ep):
            _, es = env.reset_env(reset_key, ep)
            es = es.replace(x=0.,theta=0.,theta_dot=100.)
            return env.get_obs(es),es
        step_env = env.step_env
    result=evaluate_population([p],make_cartpole_rollout(OneStep(),params,32),keys,jnp.zeros((32,4)),1)
    assert result[0][0,0] == result[1][0,0] == 1
    assert result[2][0,0] and not result[3][0,0]
    huge=params.replace(theta_threshold_radians=1e6,x_threshold=1e6)
    result=evaluate_population([p],make_cartpole_rollout(env,huge,1),keys,jnp.zeros((1,4)),1)
    assert result[3][0,0] and not result[2][0,0]


def test_seed_namespaces():
    assert not np.array_equal(key(0,"train"),key(0,"validation"))
    assert not np.array_equal(key(0,"interpolation"),key(0,"generalization"))
    np.testing.assert_array_equal(key(0,"train",7,11,9),key(0,"train",7,11,9))
