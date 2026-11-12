"""Concrete single-hidden-population dense schema; no genome dependencies."""
import numpy as np
from .contracts import BlockPlan, Graph


def blocks(w_in, w_delay, w_out, mask_in, mask_delay, mask_out,
           bias_hidden=None, bias_output=None):
    wi, wd, wo = (np.asarray(v) for v in (w_in,w_delay,w_out))
    if wi.ndim != 2 or wd.ndim != 2 or wo.ndim != 2:
        raise ValueError("Block weights must be matrices")
    i,h = wi.shape
    o = wo.shape[1]
    if min(i,o) < 1 or wd.shape != (h,h) or wo.shape[0] != h:
        raise ValueError("Inconsistent block dimensions")
    n=i+h+o
    weights, masks = [], []
    for value, mask in zip((wi,wd,wo),(mask_in,mask_delay,mask_out)):
        mask=np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != value.shape:
            raise ValueError("Invalid block mask")
        with np.errstate(over='ignore',invalid='ignore'):
            value=np.where(mask,value,0).astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite active block weight or FP32 overflow")
        weights.append(value)
        masks.append(mask.copy())
    biases=[]
    for value,size in ((bias_hidden,h),(bias_output,o)):
        with np.errstate(over='ignore',invalid='ignore'):
            value=np.zeros(size,np.float32) if value is None else np.asarray(value,np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError("Invalid block bias")
        biases.append(value.copy())
    return BlockPlan(*weights,*masks,*biases,np.ones(n,bool),np.arange(i,dtype=np.int32),
                     np.arange(i+h,n,dtype=np.int32),np.arange(n,dtype=np.int32))


def validate(p):
    checked=blocks(p.w_in,p.w_delay,p.w_out,p.mask_in,p.mask_delay,p.mask_out,
                   p.bias_hidden,p.bias_output)
    for actual,expected in zip(p,checked):
        if np.asarray(actual).dtype != expected.dtype or not np.array_equal(actual,expected):
            raise ValueError("Invalid role-block array or ordering")
    return checked


def to_graph(p):
    """Explicit small-fixture conversion, never a decoder intermediate."""
    i,h=p.w_in.shape
    o=p.w_out.shape[1]
    edges=[]
    for w,m,source,target,delay in ((p.w_in,p.mask_in,0,i,0),
            (p.w_delay,p.mask_delay,i,i,1),(p.w_out,p.mask_out,i,i+h,0)):
        for a,b in np.argwhere(m):
            edges.append((int(source+a),int(target+b),float(w[a,b]),delay))
    return Graph(tuple(range(i+h+o)),tuple(range(i)),tuple(range(i+h,i+h+o)),
        np.concatenate((np.zeros(i,np.float32),p.bias_hidden,p.bias_output)),
        np.ones(i+h+o,np.float32),np.array([0]*i+[1]*h+[0]*o,np.int32),tuple(edges),'layered')


def from_graph(g):
    """Convert canonical single-hidden graphs without losing either delay set."""
    from .planning import plan
    i,o=len(g.inputs),len(g.outputs)
    h=len(g.node_ids)-i-o
    n=i+h+o
    if (g.profile!='layered' or g.k!=1 or h<0 or g.node_ids!=tuple(range(n))
            or g.inputs!=tuple(range(i)) or g.outputs!=tuple(range(i+h,n))
            or not np.array_equal(g.response,np.ones(n))
            or not np.array_equal(g.activation,np.array([0]*i+[1]*h+[0]*o))
            or np.any(np.asarray(g.bias)[:i]!=0)):
        raise ValueError("Graph is outside the single-hidden role schema")
    p=plan(g,n,max(1,len(g.edges)))
    allowed_zero=np.zeros((n,n),bool)
    allowed_zero[:i,i:i+h]=True
    allowed_zero[i:i+h,i+h:]=True
    allowed_delay=np.zeros((n,n),bool)
    allowed_delay[i:i+h,i:i+h]=True
    if np.any(p.mask_zero & ~allowed_zero) or np.any(p.mask_delay & ~allowed_delay):
        raise ValueError("Graph has edges outside the single-hidden role schema")
    return blocks(p.w_zero[:i,i:i+h],p.w_delay[i:i+h,i:i+h],p.w_zero[i:i+h,i+h:],
        p.mask_zero[:i,i:i+h],p.mask_delay[i:i+h,i:i+h],p.mask_zero[i:i+h,i+h:],
        g.bias[i:i+h],g.bias[i+h:])


def reference_step(p,state,observation):
    """Independent FP64 role equations; scalar graph oracle checks small fixtures."""
    i,h=p.w_in.shape
    if np.shape(state)!=(len(p.node_mask),) or np.shape(observation)!=(i,):
        raise ValueError('Invalid block reference state/observation shape')
    if not np.isfinite(state).all() or not np.isfinite(observation).all():
        raise ValueError('Nonfinite block reference input')
    x=np.asarray(observation,np.float32).astype(np.float64)
    old=np.asarray(state,np.float64)
    wi=np.where(p.mask_in,p.w_in,0).astype(np.float64)
    wd=np.where(p.mask_delay,p.w_delay,0).astype(np.float64)
    wo=np.where(p.mask_out,p.w_out,0).astype(np.float64)
    z=x@wi+old[i:i+h]@wd+p.bias_hidden.astype(np.float64)
    hidden=np.tanh(z)
    output=hidden@wo+p.bias_output.astype(np.float64)
    new=np.concatenate((x,hidden,output))
    if not np.isfinite(z).all() or not np.isfinite(new).all():
        raise ValueError("Nonfinite block reference state/pre-activation")
    return output,new
