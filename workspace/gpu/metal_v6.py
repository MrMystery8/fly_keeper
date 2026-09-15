import ctypes as C
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;LIB=ROOT/'build/libmalecns_metal_v6.dylib';METALLIB=ROOT/'build/MaleCNS_v6.metallib'
class MetalSparseV6:
 def __init__(self,ptr,post,weight,width=1024):
  self.ptr=np.ascontiguousarray(ptr,np.int64);self.post=np.ascontiguousarray(post,np.int32);self.weight=np.ascontiguousarray(weight,np.float32);self.n=len(self.ptr)-1;self.lib=C.CDLL(str(LIB));self.lib.metal_create_v6.restype=C.c_void_p;self.lib.metal_create_v6.argtypes=[C.c_int,C.c_longlong,C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.c_int];self.lib.metal_destroy_v6.argtypes=[C.c_void_p];self.lib.metal_load_v6.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_step_v6.argtypes=[C.c_void_p,C.c_int];self.lib.metal_step_v6.restype=C.c_int;self.lib.metal_set_drive_v6.argtypes=[C.c_void_p,C.c_void_p];self.lib.metal_zero_counts_v6.argtypes=[C.c_void_p];self.lib.metal_read_v6.argtypes=[C.c_void_p,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p];self.ctx=self.lib.metal_create_v6(self.n,len(self.post),self.ptr.ctypes.data,self.post.ctypes.data,self.weight.ctypes.data,str(METALLIB).encode(),width)
  if not self.ctx:raise RuntimeError('Metal sparse-v6 context creation failed')
 def close(self):
  if self.ctx:self.lib.metal_destroy_v6(self.ctx);self.ctx=None
 def __enter__(self):return self
 def __exit__(self,*_):self.close()
 def load(self,s):
  f=[np.ascontiguousarray(s[k]) for k in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last')];clock=np.asarray([s['clock']],np.int64);self._state=f
  if not self.lib.metal_load_v6(self.ctx,*[x.ctypes.data for x in f],clock.ctypes.data):raise RuntimeError('Metal v6 load failed')
 def set_drive(self,d):
  d=np.ascontiguousarray(d,np.float32)
  if not self.lib.metal_set_drive_v6(self.ctx,d.ctypes.data):raise RuntimeError('Metal v6 drive failed')
 def zero_spike_counts(self):self.lib.metal_zero_counts_v6(self.ctx)
 def step_batch(self,n):
  if not self.lib.metal_step_v6(self.ctx,int(n)):raise RuntimeError('Metal v6 step failed')
 def read(self,indices):
  i=np.ascontiguousarray(indices,np.int32);c=np.empty(len(i),np.int32);v=np.empty(len(i),np.float32)
  if not self.lib.metal_read_v6(self.ctx,i.ctypes.data,len(i),c.ctypes.data,v.ctypes.data):raise RuntimeError('Metal v6 read failed')
  return c,v
