"""S1 recurrent edge-list prototype; dense input/output blocks stay unchanged."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class EdgeBlock(NamedTuple):
    w_in: object
    w_out: object
    mask_in: object
    mask_out: object
    source: object
    target: object
    weight: object
    valid: object
    bias_hidden: object
    bias_output: object
    node_mask: object


def checked_edges(source, target, weight, hidden):
    """One edge per ordered pair; never silently sum duplicate genes."""
    src,dst,w=(np.asarray(v) for v in (source,target,weight))
    if (src.ndim!=1 or dst.shape!=src.shape or w.shape!=src.shape or
            src.dtype!=np.int32 or dst.dtype!=np.int32 or w.dtype!=np.float32 or
            type(hidden) is not int or hidden < 0 or
            np.any(src<0) or np.any(src>=hidden) or
            np.any(dst<0) or np.any(dst>=hidden) or
            not np.isfinite(w).all() or np.any(w==0)):
        raise ValueError('Invalid recurrent edge list')
    pairs=np.stack((src,dst),axis=1) if len(src) else np.empty((0,2),np.int32)
    if len(np.unique(pairs,axis=0))!=len(src):
        raise ValueError('Duplicate recurrent edge pair')
    return src.copy(),dst.copy(),w.copy()


def from_blocks(plan, edge_capacity):
    """Losslessly convert the delayed block to one padded edge-list bucket."""
    h=plan.w_delay.shape[0]
    if type(edge_capacity) is not int or edge_capacity < 0:
        raise ValueError('Invalid recurrent edge capacity')
    positions=np.argwhere(plan.mask_delay)
    edges=len(positions)
    if edges>edge_capacity:
        raise OverflowError('Recurrent edge list exceeds declared capacity')
    src=positions[:,0].astype(np.int32)
    dst=positions[:,1].astype(np.int32)
    weight=np.asarray(plan.w_delay[src,dst],np.float32)
    src,dst,weight=checked_edges(src,dst,weight,h)
    width=min(edge_capacity,1 << (edges-1).bit_length()) if edges else 0
    source=np.zeros(width,np.int32)
    target=np.zeros(width,np.int32)
    values=np.zeros(width,np.float32)
    valid=np.zeros(width,bool)
    source[:edges],target[:edges],values[:edges],valid[:edges]=src,dst,weight,True
    return EdgeBlock(plan.w_in.copy(),plan.w_out.copy(),plan.mask_in.copy(),
                     plan.mask_out.copy(),source,target,values,valid,
                     plan.bias_hidden.copy(),plan.bias_output.copy(),
                     plan.node_mask.copy())


def edge_step(plan,state,observation):
    """One persistent layered step with recurrent scatter-add in edge order."""
    inputs,hidden=plan.w_in.shape
    previous=jnp.where(plan.node_mask[inputs:inputs+hidden],
                       state[inputs:inputs+hidden],0)
    wi=jnp.where(plan.mask_in,plan.w_in,0)
    wo=jnp.where(plan.mask_out,plan.w_out,0)
    drive=jnp.matmul(observation,wi,precision=jax.lax.Precision.HIGHEST)
    if hidden:
        src=jnp.where(plan.valid,plan.source,0)
        dst=jnp.where(plan.valid,plan.target,0)
        contributions=jnp.where(plan.valid,previous[src]*plan.weight,0)
        recurrent=jnp.zeros(hidden,jnp.float32).at[dst].add(contributions)
    else:
        recurrent=jnp.zeros(0,jnp.float32)
    z=drive+recurrent+plan.bias_hidden
    values=jnp.where(jnp.isfinite(z),jnp.tanh(z),jnp.nan)
    output=jnp.matmul(values,wo,precision=jax.lax.Precision.HIGHEST)+plan.bias_output
    new=jnp.concatenate((observation,values,output))
    return output,new


compiled_edge_step=jax.jit(edge_step)
