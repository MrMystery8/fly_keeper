"""Persistent, whole-batch deterministic Metal backend."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LIB=ROOT/'build/libmalecns_metal_v4.dylib';METALLIB=ROOT/'build/MaleCNS_v4.metallib'

class _Stats(C.Structure):
    _fields_=[('clock',C.c_int64),('n',C.c_uint32),('slots',C.c_uint32),('delay',C.c_uint32),('rfc',C.c_uint32),('dt',C.c_float),('total_sources',C.c_uint64),('total_edges',C.c_uint64),('total_active_visits',C.c_uint64),('max_sources',C.c_uint32),('max_edges',C.c_uint32),('max_active',C.c_uint32)]

class MetalBatchedV4:
    def __init__(self,ptr,post,weight,width=256):
        self.ptr=np.ascontiguousarray(ptr,np.int64);self.post=np.ascontiguousarray(post,np.int32);self.weight=np.ascontiguousarray(weight,np.float32);self.n=len(self.ptr)-1
        self.lib=C.CDLL(str(LIB));self.lib.metal_create_deterministic_v4.restype=C.c_void_p;self.lib.metal_create_deterministic_v4.argtypes=[C.c_int,C.c_longlong,C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p,C.c_int]
        self.lib.metal_destroy_v4.argtypes=[C.c_void_p];self.lib.metal_step_v4.argtypes=[C.c_void_p];self.lib.metal_step_v4.restype=C.c_int;self.lib.metal_step_v4_batch.argtypes=[C.c_void_p,C.c_int];self.lib.metal_step_v4_batch.restype=C.c_int
        self.lib.metal_set_drive_v4.argtypes=[C.c_void_p,C.c_void_p];self.lib.metal_set_drive_v4.restype=C.c_int;self.lib.metal_zero_spike_counts_v4.argtypes=[C.c_void_p];self.lib.metal_zero_spike_counts_v4.restype=C.c_int
        self.lib.metal_load_state_v4.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_load_state_v4.restype=C.c_int;self.lib.metal_copy_state_v4.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_copy_state_v4.restype=C.c_int
        self.lib.metal_read_v4.argtypes=[C.c_void_p,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p];self.lib.metal_read_v4.restype=C.c_int;self.lib.metal_stats_v4.argtypes=[C.c_void_p,C.POINTER(_Stats)];self.lib.metal_stats_v4.restype=C.c_int
        self.ctx=self.lib.metal_create_deterministic_v4(self.n,len(self.post),self.ptr.ctypes.data,self.post.ctypes.data,self.weight.ctypes.data,str(METALLIB).encode(),width)
        if not self.ctx:raise RuntimeError('Metal batched-v4 context creation failed')
    def close(self):
        if self.ctx:self.lib.metal_destroy_v4(self.ctx);self.ctx=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def load(self,s):
        fields=tuple(s[k] for k in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last'));arrays=[np.ascontiguousarray(x) for x in fields];clock=np.asarray([s['clock']],np.int64);self._state_arrays=arrays
        if not self.lib.metal_load_state_v4(self.ctx,*[x.ctypes.data for x in arrays],clock.ctypes.data):raise RuntimeError('Metal v4 state load failed')
    def copy(self):
        n=self.n;s={'v':np.empty(n,np.float32),'g':np.empty(n,np.float32),'refractory':np.empty(n,np.int16),'drive':np.empty(n,np.float32),'previous_drive':np.empty(n,np.float32),'queue':np.empty((19,n),np.int32),'queue_count':np.empty(19,np.int32),'counts':np.empty(n,np.int32),'active':np.empty(n,np.int32),'flags':np.empty(n,np.uint8),'nactive':np.empty(1,np.int32),'last':np.empty(n,np.int64),'clock':0};fields=tuple(s[k] for k in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last'));clock=np.zeros(1,np.int64)
        if not self.lib.metal_copy_state_v4(self.ctx,*[x.ctypes.data for x in fields],clock.ctypes.data):raise RuntimeError('Metal v4 state copy failed')
        s['clock']=int(clock[0]);return s
    def step(self):
        if not self.lib.metal_step_v4(self.ctx):raise RuntimeError('Metal v4 step failed')
    def step_batch(self,steps):
        if not self.lib.metal_step_v4_batch(self.ctx,int(steps)):raise RuntimeError('Metal v4 batch step failed')
    def set_drive(self,drive):
        d=np.ascontiguousarray(drive,np.float32)
        if d.shape!=(self.n,) or not self.lib.metal_set_drive_v4(self.ctx,d.ctypes.data):raise RuntimeError('Metal v4 drive update failed')
    def zero_spike_counts(self):
        if not self.lib.metal_zero_spike_counts_v4(self.ctx):raise RuntimeError('Metal v4 spike-count reset failed')
    def read(self,indices):
        idx=np.ascontiguousarray(indices,np.int32);counts=np.empty(len(idx),np.int32);voltage=np.empty(len(idx),np.float32)
        if not self.lib.metal_read_v4(self.ctx,idx.ctypes.data,len(idx),counts.ctypes.data,voltage.ctypes.data):raise RuntimeError('Metal v4 read failed')
        return counts,voltage
    def stats(self):
        s=_Stats()
        if not self.lib.metal_stats_v4(self.ctx,C.byref(s)):raise RuntimeError('Metal v4 stats failed')
        return {name:getattr(s,name) for name,_ in s._fields_}
