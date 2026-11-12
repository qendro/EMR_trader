"""P0 partially observed control baseline and paired seed-level analysis."""
import jax
import jax.numpy as jnp
import numpy as np

from .environments import cartpole
from .randomness import key


def random_action_returns(seed, episodes=100, horizon=500):
    """Match the finalist's held-out resets; random actions use separate keys."""
    if episodes < 1 or horizon < 1:
        raise ValueError('Invalid random-action protocol')
    env, params = cartpole()
    roots = jnp.stack([key(seed, 'interpolation', 0, e) for e in range(episodes)])

    @jax.jit
    def evaluate(keys):
        def episode(root):
            reset_key, rollout_key = jax.random.split(root)
            _, state = env.reset_env(reset_key, params)
            def body(carry, time):
                current, active, total = carry
                action = jax.random.bernoulli(jax.random.fold_in(rollout_key, time)).astype(jnp.int32)
                _, nxt, reward, done, _ = env.step_env(
                    jax.random.fold_in(rollout_key, time), current, action, params)
                return (jax.tree.map(lambda a, b: jnp.where(active, a, b), nxt, current),
                        active & ~done, total + jnp.where(active, reward, 0)), None
            return jax.lax.scan(body, (state, jnp.bool_(True), jnp.float32(0)),
                                jnp.arange(horizon))[0][2]
        return jax.vmap(episode)(keys)
    return np.asarray(evaluate(roots), np.float32)


def paired_quality(policy_returns, random_returns, *, scale=500., bootstrap_samples=20000,
                   bootstrap_seed=2026):
    """Resample independent evolution seeds, preserving their episode pairs."""
    policy = np.asarray(policy_returns, np.float64)
    baseline = np.asarray(random_returns, np.float64)
    if (policy.ndim != 2 or policy.shape != baseline.shape or policy.shape[0] < 5 or
            policy.shape[1] < 100 or not np.isfinite(policy).all() or
            not np.isfinite(baseline).all() or not np.isfinite(scale) or scale <= 0 or
            type(bootstrap_samples) is not int or bootstrap_samples < 1000):
        raise ValueError('P0 requires at least five seeds and 100 paired held-out episodes')
    per_seed=(policy-baseline).mean(axis=1)/scale
    rng=np.random.default_rng(bootstrap_seed)
    indices=rng.integers(0,len(per_seed),size=(bootstrap_samples,len(per_seed)))
    boot=np.median(per_seed[indices],axis=1)
    low,high=np.percentile(boot,[2.5,97.5])
    return {'seeds':len(per_seed),'episodes_per_seed':policy.shape[1],
            'normalization_scale':float(scale),'per_seed_improvement':per_seed.tolist(),
            'median_improvement':float(np.median(per_seed)),
            'bootstrap_95_interval':[float(low),float(high)],
            'bootstrap_unit':'evolution-seed','bootstrap_samples':bootstrap_samples,
            'passes':bool(np.median(per_seed)>=.2 and low>0)}
