"""FP32 graph and single-hidden-population steps with dynamic topology values."""
from functools import partial
import jax
import jax.numpy as jnp
from .contracts import BlockPlan

jax.config.update("jax_default_matmul_precision", "float32")


def step(p, state, observation, *, profile):
    if isinstance(p, BlockPlan):
        if profile != "layered":
            raise ValueError("Role blocks require the layered profile")
        return block_step(p, state, observation)
    if profile not in ("dag", "synchronous", "layered"):
        raise ValueError("Unsupported execution profile")
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
        if profile == "layered":
            # Delayed input edges, if present, read the previous observation too.
            previous = jnp.where(p.node_mask, state, 0)
            wd = jnp.where(p.mask_delay, p.w_delay, 0)
        def body(carry):
            idx, values = carry
            node = p.order[idx]
            summed = jnp.dot(values, wz[:, node], precision=jax.lax.Precision.HIGHEST)
            if profile == "layered":
                summed = summed + jnp.dot(previous, wd[:, node], precision=jax.lax.Precision.HIGHEST)
            z = p.bias[node] + p.response[node] * summed
            activated = jnp.where(p.activation[node] == 1, jnp.tanh(z), z)
            activated = jnp.where(jnp.isfinite(z), activated, jnp.nan)
            return idx + 1, values.at[node].set(activated)
        _, new = jax.lax.while_loop(lambda c: c[0] < p.count, body, (jnp.int32(0), initial))
    return new[p.outputs], new


def block_preactivation(p, state, observation):
    i, h = p.w_in.shape
    previous = jnp.where(p.node_mask[i:i+h], state[i:i+h], 0)
    wi = jnp.where(p.mask_in, p.w_in, 0)
    wd = jnp.where(p.mask_delay, p.w_delay, 0)
    return (jnp.matmul(observation, wi, precision=jax.lax.Precision.HIGHEST)
         + jnp.matmul(previous, wd, precision=jax.lax.Precision.HIGHEST) + p.bias_hidden)


def block_step(p, state, observation):
    z=block_preactivation(p,state,observation)
    wo = jnp.where(p.mask_out, p.w_out, 0)
    hidden = jnp.where(jnp.isfinite(z), jnp.tanh(z), jnp.nan)
    output = jnp.matmul(hidden, wo, precision=jax.lax.Precision.HIGHEST) + p.bias_output
    new = jnp.concatenate((observation, hidden, output))
    return output, new


compiled_step = partial(jax.jit, static_argnames=("profile",))(step)


def population_step(plans, states, observations, *, profile):
    return jax.vmap(lambda p, s, x: jax.vmap(
        lambda ss, xx: step(p, ss, xx, profile=profile))(s, x))(plans, states, observations)
