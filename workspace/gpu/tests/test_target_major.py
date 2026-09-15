import numpy as np
def target_major(ptr,post,weight):
 order=np.argsort(post,kind='stable');off=np.r_[0,np.cumsum(np.bincount(post[order],minlength=len(ptr)-1))]
 src=np.searchsorted(ptr,order,side='right')-1;return off,src,weight[order],order
def test_convergent_inputs_preserve_original_ordinal_order():
 ptr=np.array([0,2,3,3]);post=np.array([2,2,2]);weight=np.array([1.,1e20,-1e20],np.float32)
 off,src,w,ord=target_major(ptr,post,weight)
 assert ord.tolist()==[0,1,2] and src.tolist()==[0,0,1]
 # Same source/edge traversal sequence for target 2, so float32 sum is exact-equivalent.
 assert np.float32(sum(w))==np.float32(sum(weight))
