"""ctypes harness for the separately-built deterministic-v2 Metal backend."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LIB=ROOT/'build/libmalecns_metal_v2.dylib'
METALLIB=ROOT/'build/MaleCNS_v2.metallib'

class MetalDeterministicV2:
    def __init__(self,ptr,post,weight,width=64):
        self.ptr=np.ascontiguousarray(ptr,np.int64);self.post=np.ascontiguousarray(post,np.int32);self.weight=np.ascontiguousarray(weight,np.float32);self.n=len(self.ptr)-1
        self.lib=C.CDLL(str(LIB)); self.lib.metal_create_deterministic_v2.restype=C.c_void_p
        self.lib.metal_create_deterministic_v2.argtypes=[C.c_int,C.c_longlong,C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.c_int]
        for name in ('metal_destroy_v2','metal_step_v2','metal_step_v2_batch','metal_set_drive_v2','metal_zero_spike_counts_v2','metal_load_state_v2','metal_copy_state_v2'):
            getattr(self.lib,name)
        self.lib.metal_destroy_v2.argtypes=[C.c_void_p]; self.lib.metal_step_v2.argtypes=[C.c_void_p];self.lib.metal_step_v2.restype=C.c_int
        self.lib.metal_step_v2_batch.argtypes=[C.c_void_p,C.c_int];self.lib.metal_step_v2_batch.restype=C.c_int
        self.lib.metal_set_drive_v2.argtypes=[C.c_void_p,C.c_void_p];self.lib.metal_set_drive_v2.restype=C.c_int;self.lib.metal_zero_spike_counts_v2.argtypes=[C.c_void_p];self.lib.metal_zero_spike_counts_v2.restype=C.c_int
        self.lib.metal_load_state_v2.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_load_state_v2.restype=C.c_int;self.lib.metal_copy_state_v2.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_copy_state_v2.restype=C.c_int
        self.ctx=self.lib.metal_create_deterministic_v2(self.n,len(self.post),self.ptr.ctypes.data,self.post.ctypes.data,self.weight.ctypes.data,str(METALLIB).encode(),width)
        if not self.ctx: raise RuntimeError('Metal deterministic-v2 context creation failed')
    def close(self):
        if self.ctx:self.lib.metal_destroy_v2(self.ctx);self.ctx=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def load(self,s):
        fields=tuple(s[k] for k in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last'));arrays=[np.ascontiguousarray(x) for x in fields];clock=np.asarray([s['clock']],np.int64);self._state_arrays=arrays
        if not self.lib.metal_load_state_v2(self.ctx,*[x.ctypes.data for x in arrays],clock.ctypes.data):raise RuntimeError('Metal v2 state load failed')
    def copy(self):
        n=self.n;s={'v':np.empty(n,np.float32),'g':np.empty(n,np.float32),'refractory':np.empty(n,np.int16),'drive':np.empty(n,np.float32),'previous_drive':np.empty(n,np.float32),'queue':np.empty((19,n),np.int32),'queue_count':np.empty(19,np.int32),'counts':np.empty(n,np.int32),'active':np.empty(n,np.int32),'flags':np.empty(n,np.uint8),'nactive':np.empty(1,np.int32),'last':np.empty(n,np.int64),'clock':0};fields=tuple(s[k] for k in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last'));clock=np.zeros(1,np.int64)
        if not self.lib.metal_copy_state_v2(self.ctx,*[x.ctypes.data for x in fields],clock.ctypes.data):raise RuntimeError('Metal v2 state copy failed')
        s['clock']=int(clock[0]);return s
    def step(self):
        if not self.lib.metal_step_v2(self.ctx):raise RuntimeError('Metal v2 step failed')
    def step_batch(self,steps):
        if not self.lib.metal_step_v2_batch(self.ctx,int(steps)):raise RuntimeError('Metal v2 batch step failed')
    def set_drive(self,drive):
        d=np.ascontiguousarray(drive,np.float32)
        if d.shape!=(self.n,) or not self.lib.metal_set_drive_v2(self.ctx,d.ctypes.data):raise RuntimeError('Metal v2 drive update failed')
    def zero_spike_counts(self):
        if not self.lib.metal_zero_spike_counts_v2(self.ctx):raise RuntimeError('Metal v2 spike-count reset failed')
