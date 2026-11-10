"""H0 fixed-coordinate development directly into role-shaped dense blocks."""
import numpy as np
from .substrate import blocks


def settings(config):
    required={'schema','kind','inputs','hidden','outputs','recurrent','threshold','max_weight','query_slab'}
    if set(config)!=required or type(config['schema']) is not int or config['schema']!=1 or config['kind']!='fixed-coordinate-v1':
        raise ValueError('Unsupported fixed decoder configuration')
    for name in ('inputs','hidden','outputs','query_slab'):
        if type(config[name]) is not int or config[name]<1:
            raise ValueError('Invalid fixed decoder dimensions/slab')
    if config['inputs']+config['hidden']+config['outputs']>1024:
        raise ValueError('Fixed phenotype capacity exceeds the H1 1024-node envelope')
    if config['query_slab']>65536:
        raise ValueError('Query slab exceeds the bounded 65536-row envelope')
    if type(config['recurrent']) is not bool:
        raise ValueError('Recurrence must be explicit')
    for name in ('threshold','max_weight'):
        if type(config[name]) not in (int,float) or not np.isfinite(np.float32(config[name])):
            raise ValueError('Invalid fixed weight transform')
    if not 0<=config['threshold']<1 or config['max_weight']<=0:
        raise ValueError('Invalid fixed weight threshold/range')
    return dict(config)


def coordinates(size,y):
    # Match the pinned MLPSubstrate lattice, rounded once to storage FP32.
    x=np.array([0.] if size==1 else [-1.+2./(size-1)*j for j in range(size)],np.float32)
    return np.column_stack((x,np.full(size,y,np.float32))).astype(np.float32)


def transform(scores,threshold,max_weight):
    scores=np.asarray(scores,np.float32)
    if not np.isfinite(scores).all() or np.any(np.abs(scores)>1):
        raise ValueError('Invalid bounded CPPN query score')
    threshold,max_weight=np.float32(threshold),np.float32(max_weight)
    mask=np.abs(scores)>threshold
    # Threshold equality produces zero and is not an expressed connection.
    weights=np.where(mask,np.sign(scores)*(np.abs(scores)-threshold)/np.float32(1-threshold)*max_weight,0)
    if not np.isfinite(weights).all():
        raise ValueError('Fixed weight transform overflow')
    return weights.astype(np.float32),mask


class FixedDecoder:
    def __init__(self,query,config):
        self.config=settings(config)
        self.query=query
        c=self.config
        # Last virtual input is constant one. Its decoded row becomes hidden bias.
        self.input_coordinates=coordinates(c['inputs']+1,-1.)
        self.hidden_coordinates=coordinates(c['hidden'],0.)
        self.output_coordinates=coordinates(c['outputs'],1.)
        self.last_stats=None

    def _block(self,p,source,target):
        c=self.config
        count=len(source)*len(target)
        weights=np.zeros(count,np.float32)
        masks=np.zeros(count,bool)
        slab=c['query_slab']
        for start in range(0,count,slab):
            take=min(slab,count-start)
            indices=np.arange(start,start+take)
            rows=np.concatenate((source[indices//len(target)],target[indices%len(target)]),axis=1)
            if take<slab:
                rows=np.concatenate((rows,np.repeat(rows[:1],slab-take,axis=0)))
            scores=self.query(p,rows)[:take]
            w,m=transform(scores,c['threshold'],c['max_weight'])
            weights[start:start+take],masks[start:start+take]=w,m
        return weights.reshape(len(source),len(target)),masks.reshape(len(source),len(target))

    def decode(self,nodes,conns):
        p=self.query.prepare(nodes,conns)
        wi,mi=self._block(p,self.input_coordinates,self.hidden_coordinates)
        wo,mo=self._block(p,self.hidden_coordinates,self.output_coordinates)
        h=self.config['hidden']
        if self.config['recurrent']:
            wd,md=self._block(p,self.hidden_coordinates,self.hidden_coordinates)
        else:
            wd,md=np.zeros((h,h),np.float32),np.zeros((h,h),bool)
        result=blocks(wi[:-1],wd,wo,mi[:-1],md,mo,wi[-1])
        queries=wi.size+wo.size+(wd.size if self.config['recurrent'] else 0)
        self.last_stats={'queries':queries,'expressed_edges':int(mi[:-1].sum()+md.sum()+mo.sum()),
            'bias_queries':h,'expressed_biases':int(mi[-1].sum()),
            'phenotype_nodes':len(result.node_mask),'query_slab':self.config['query_slab'],
            'encoding':'role-blocks-v1','host_array_bytes':sum(v.nbytes for v in result)}
        return result

    def interpretation(self):
        return {'decoder':self.config,'query_order':['source-x','source-y','target-x','target-y'],
            'input_coordinates_including_virtual_bias':self.input_coordinates.tolist(),
            'hidden_coordinates':self.hidden_coordinates.tolist(),'output_coordinates':self.output_coordinates.tolist(),
            'bias':'last virtual input is one; its weights become hidden biases; output bias zero',
            'cppn_output':'tanh of direct identity head','weight_transform':'sign(q)*(abs(q)-threshold)/(1-threshold)*max_weight when abs(q)>threshold; else zero',
            'role_activation':'tanh-hidden-identity-output','response':'one','state_order':'input-hidden-output',
            'delays':'input-hidden/output immediate; hidden-hidden unit-delay'}
