"""S0 lossless index maps, stable buckets and permutation-safe routing."""
import numpy as np
import pytest

from neuroevo.adapters.emr_phenotype import EvaluatedPhenotype
from neuroevo.packing import BucketFamily, index_map, route
from neuroevo.substrate import blocks, reference_step


def phenotype(hidden, index):
    wi=np.arange(2*hidden,dtype=np.float32).reshape(2,hidden)/10+.1
    wd=np.eye(hidden,dtype=np.float32)*.2
    wo=np.ones((hidden,1),np.float32)*.3
    p=blocks(wi,wd,wo,wi!=0,wd!=0,wo!=0)
    grid=np.arange(index,index+hidden,dtype=np.int32)
    return EvaluatedPhenotype(p,grid,np.zeros((hidden,2),np.float32),3,0)


def test_s0_index_map_and_capacity_promotion():
    compact,inverse=index_map(np.array([1,4,9],np.int32),12)
    np.testing.assert_array_equal(inverse[compact],np.arange(3))
    assert np.count_nonzero(inverse==-1)==9
    with pytest.raises(ValueError):
        index_map(np.array([4,1],np.int32),12)
    with pytest.raises(ValueError):
        index_map(np.array([1,1],np.int32),12)
    family=BucketFamily.create(2,1,23)
    assert family.widths==(0,4,8,16,20)
    assert [family.width(n) for n in (0,1,4,5,9,20)]==[0,4,4,8,16,20]
    with pytest.raises(OverflowError):
        family.width(21)
    wider=family.promote(43)
    assert wider.widths==(0,4,8,16,20,32,40)
    assert all(wider.width(n)==family.width(n) for n in range(21))


def test_s0_padded_carry_matches_compact_and_permutation_restores_ids():
    family=BucketFamily.create(2,1,23)
    phenotypes=[phenotype(n,0) for n in (0,3,7,11,20)]
    plans=[family.pack(p) for p in phenotypes]
    assert [p.w_in.shape[1] for p in plans]==[0,4,8,16,20]
    assert len(route(plans))==5
    for original,padded in zip(phenotypes,plans):
        state=np.zeros(len(original.plan.node_mask),np.float64)
        padded_state=np.zeros(len(padded.node_mask),np.float64)
        for observation in (np.array([.3,-.2]),np.array([-.7,.6])):
            head,state=reference_step(original.plan,state,observation)
            packed_head,padded_state=reference_step(padded,padded_state,observation)
            np.testing.assert_allclose(packed_head,head,atol=1e-6)
            h=len(original.grid_indices)
            np.testing.assert_allclose(padded_state[2:2+h],state[2:2+h],atol=1e-6)
    permutation=(4,1,3,0,2)
    shuffled=[plans[i] for i in permutation]
    recovered=np.empty(5,np.int32)
    for _,indices in route(shuffled):
        for index in indices:
            recovered[index]=permutation[index]
    np.testing.assert_array_equal(recovered,permutation)
