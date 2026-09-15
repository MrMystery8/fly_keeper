import numpy as np
from workspace.gpu.metal_reference import MetalReference
def cpu_compact(active,flags):return np.asarray(active,dtype=np.int32)[np.asarray(flags,dtype=bool)]
def test_unsorted_stable_order():assert cpu_compact([91,7,500,2,44],[1,0,1,1,0]).tolist()==[91,500,2]
def test_random_stable_order():
 rng=np.random.default_rng(7)
 with MetalReference(np.array([0,0],np.int64),np.empty(0,np.int32),np.empty(0,np.float32)) as metal:
  for n in range(1,200):
   active=rng.permutation(n);flags=rng.integers(0,2,n)
   expected=cpu_compact(active,flags)
   assert np.array_equal(metal.compact(active,flags),expected)

def test_gpu_compaction_unsorted_and_empty():
 with MetalReference(np.array([0,0],np.int64),np.empty(0,np.int32),np.empty(0,np.float32)) as metal:
  assert metal.compact(np.array([91,7,500,2,44],np.int32),np.array([1,0,1,1,0],np.int32)).tolist()==[91,500,2]
  assert metal.compact(np.empty(0,np.int32),np.empty(0,np.int32)).tolist()==[]
