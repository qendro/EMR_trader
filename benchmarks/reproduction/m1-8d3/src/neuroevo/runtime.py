"""M1 JAX DAG and synchronous steps. All topology arrays are dynamic arguments."""
from functools import partial
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")


def step(p, state, observation, *, profile):
    if profile not in ("dag", "synchronous"):
        raise ValueError("Layered JAX execution is gated on M2")
    wz = jnp.where(p.mask_zero, p.w_zero, 0)
    old = jnp.where(p.node_mask, state, 0).at[p.inputs].set(observation)
    def activate(z):
        return jnp.where(p.activation == 1, jnp.tanh(z), z)
    if profile == "synchronous":
        z = p.bias + p.response * jnp.matmul(old, wz, precision=jax.lax.Precision.HIGHEST)
        # Saturating tanh must not conceal an FP32 pre-activation overflow.
        values = jnp.where(jnp.isfinite(z), activate(z), jnp.nan)
        new = jnp.where(p.node_mask, values, 0).at[p.inputs].set(observation)
    else:
        initial = jnp.zeros_like(old).at[p.inputs].set(observation)
        def body(carry):
            idx, values = carry
            node = p.order[idx]
            z = p.bias[node] + p.response[node] * jnp.dot(
                values, wz[:, node], precision=jax.lax.Precision.HIGHEST)
            activated = jnp.where(p.activation[node] == 1, jnp.tanh(z), z)
            activated = jnp.where(jnp.isfinite(z), activated, jnp.nan)
            return idx + 1, values.at[node].set(activated)
        _, new = jax.lax.while_loop(lambda c: c[0] < p.count, body, (jnp.int32(0), initial))
    return new[p.outputs], new


compiled_step = partial(jax.jit, static_argnames=("profile",))(step)


def population_step(plans, states, observations, *, profile):
    return jax.vmap(lambda p, s, x: jax.vmap(
        lambda ss, xx: step(p, ss, xx, profile=profile))(s, x))(plans, states, observations)
