"""H0 query bridge: native genes -> small direct plan -> independent samples."""
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.planning import plan
from neuroevo.runtime import step


@jax.jit
def query_scores(p,rows):
    def one(row):
        output,state=step(p,jnp.zeros(p.node_mask.shape,jnp.float32),row,profile='dag')
        return jnp.where(jnp.all(jnp.isfinite(state)),jnp.tanh(output[0]),jnp.nan)
    return jax.vmap(one)(rows)


class CPPNQuery:
    def __init__(self,adapter):
        if (adapter.config['inputs'],adapter.config['outputs'],adapter.config['profile']) != (4,1,'dag'):
            raise ValueError('Fixed CPPN requires four coordinate inputs, one output, DAG')
        self.adapter=adapter

    def prepare(self,nodes,conns):
        return plan(self.adapter.decode(nodes,conns),self.adapter.config['nodes'],self.adapter.config['edges'])

    def __call__(self,p,rows):
        return np.asarray(jax.device_get(query_scores(p,np.asarray(rows,np.float32))))
