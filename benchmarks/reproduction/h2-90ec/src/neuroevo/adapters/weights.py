"""B0 only: fixed-topology truncation selection and bounded Gaussian mutation."""
from dataclasses import dataclass
import hashlib
import math
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from neuroevo.randomness import key
from neuroevo.substrate import blocks


class WeightState(NamedTuple):
    generation: object
    weights: object
    randkey: object


@dataclass(frozen=True)
class WeightBatch:
    token: str
    ids: tuple[str,...]
    weights: np.ndarray


def settings(config):
    names={'schema','kind','population','inputs','hidden','outputs','profile','recurrent',
           'init_std','mutation_std','mutation_rate','survival_fraction','elite_count','weight_bound'}
    if set(config)!=names or type(config['schema']) is not int or config['schema']!=1 or config['kind']!='fixed-weight-v1':
        raise ValueError('Unsupported fixed-weight adapter configuration')
    for field in ('population','inputs','hidden','outputs','elite_count'):
        if type(config[field]) is not int or config[field]<1:
            raise ValueError('Invalid fixed-weight dimensions/elite count')
    if config['population']<2 or config['elite_count']>=config['population']:
        raise ValueError('Fixed-weight population must leave nonelite offspring')
    if config['inputs']+config['hidden']+config['outputs']>1024:
        raise ValueError('Fixed-weight phenotype exceeds 1024-node envelope')
    if config['profile']!='layered' or type(config['recurrent']) is not bool:
        raise ValueError('Fixed weights require explicit layered recurrence')
    for field in ('init_std','mutation_std','mutation_rate','survival_fraction','weight_bound'):
        if type(config[field]) not in (int,float) or not np.isfinite(config[field]):
            raise ValueError('Invalid fixed-weight mutation/distribution setting')
    if (config['init_std']<=0 or config['mutation_std']<0 or not 0<=config['mutation_rate']<=1
            or not 0<config['survival_fraction']<=1 or not 0<config['weight_bound']<=5):
        raise ValueError('Unsupported fixed-weight initialization/mutation bounds')
    return dict(config)


class WeightAdapter:
    def __init__(self,config):
        self.config=settings(config)
        c=self.config
        self.i,self.h,self.o=c['inputs'],c['hidden'],c['outputs']
        self.sizes=(self.i*self.h,self.h*self.h if c['recurrent'] else 0,self.h*self.o,self.h)
        self.parameters=sum(self.sizes)
        survivors=math.ceil(c['population']*c['survival_fraction'])
        elite=c['elite_count']
        def reproduce(state,scores):
            order=jnp.argsort(-scores,stable=True)
            finite=jnp.isfinite(scores)
            count=jnp.sum(finite)
            pool=jnp.minimum(jnp.int32(survivors),count)
            next_key,parents_key,noise_key,mask_key=jax.random.split(state.randkey,4)
            ranks=jax.random.randint(parents_key,(c['population']-elite,),0,pool)
            parents=state.weights[order[ranks]]
            noise=jax.random.normal(noise_key,parents.shape,dtype=jnp.float32)*jnp.float32(c['mutation_std'])
            mutate=jax.random.uniform(mask_key,parents.shape)<jnp.float32(c['mutation_rate'])
            children=jnp.clip(parents+jnp.where(mutate,noise,0),-c['weight_bound'],c['weight_bound'])
            # If fewer candidates are valid than the elite count, repeat valid elites.
            elites=state.weights[order[jnp.minimum(jnp.arange(elite),count-1)]]
            return WeightState(state.generation+1,jnp.concatenate((elites,children)),next_key)
        self._tell=jax.jit(reproduce)
        self.pending=None

    def initialize(self,seed):
        next_key,initial=jax.random.split(key(seed,'evolution'))
        values=jax.random.normal(initial,(self.config['population'],self.parameters),dtype=jnp.float32)
        values=jnp.clip(values*jnp.float32(self.config['init_std']),-self.config['weight_bound'],self.config['weight_bound'])
        return WeightState(jnp.int32(0),values,next_key)

    @staticmethod
    def token(ids,weights):
        return hashlib.sha256('|'.join(ids).encode()+np.asarray(weights).tobytes()).hexdigest()

    def ask(self,state,run_id):
        values=np.asarray(jax.device_get(state.weights)).copy()
        ids=tuple(f'{run_id}:{int(state.generation)}:{slot}' for slot in range(len(values)))
        token=self.token(ids,values)
        values.flags.writeable=False
        self.pending=(token,ids)
        return WeightBatch(token,ids,values)

    def tell(self,state,batch,ids,scores):
        if self.pending!=(batch.token,tuple(ids)) or tuple(ids)!=batch.ids:
            raise ValueError('Stale or misordered fitness association')
        if self.token(batch.ids,np.asarray(state.weights))!=batch.token or self.token(batch.ids,batch.weights)!=batch.token:
            raise ValueError('Population changed after ask')
        scores=np.asarray(scores,np.float32)
        if scores.shape!=(len(batch.ids),):
            raise ValueError('Wrong fitness vector shape')
        if not np.isfinite(scores).any():
            raise ValueError('All candidates invalid; refusing reproduction')
        # -inf remains nonfinite for the valid-parent count and sorts last.
        result=self._tell(state,jnp.asarray(np.where(np.isfinite(scores),scores,-np.inf)))
        jax.block_until_ready(result)
        self.pending=None
        return result,[]

    def decode(self,weights):
        weights=np.asarray(weights)
        if weights.shape!=(self.parameters,) or weights.dtype!=np.float32 or not np.isfinite(weights).all():
            raise ValueError('Invalid fixed-weight genome')
        if np.any(np.abs(weights)>self.config['weight_bound']):
            raise ValueError('Fixed-weight genome exceeds declared bounds')
        a,b,c,d=np.split(weights,np.cumsum(self.sizes)[:-1])
        wd=b.reshape(self.h,self.h) if self.config['recurrent'] else np.zeros((self.h,self.h),np.float32)
        return blocks(a.reshape(self.i,self.h),wd,c.reshape(self.h,self.o),
            np.ones((self.i,self.h),bool),np.full((self.h,self.h),self.config['recurrent'],bool),
            np.ones((self.h,self.o),bool),d)

    def interpretation(self):
        return {'baseline':self.config,'genotype_encoding':'flat-role-weights-v1',
            'weight_order':['input-hidden','hidden-hidden-delayed-if-enabled','hidden-output','hidden-bias'],
            'weight_segment_sizes':list(self.sizes),'state_order':'input-hidden-output',
            'role_activation':'tanh-hidden-identity-output','response':'one','output_bias':'zero',
            'topology':'fixed dense input-hidden/hidden-output; fixed delayed hidden-hidden if enabled'}
