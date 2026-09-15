"""Narrow ctypes harness for the non-selectable serial Metal parity oracle."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LIB=ROOT/'build/libmalecns_metal_bridge.dylib'
METALLIB=ROOT/'build/MaleCNS.metallib'

class MetalReference:
    def __init__(self,ptr,post,weight):
        self.ptr=np.ascontiguousarray(ptr,dtype=np.int64);self.post=np.ascontiguousarray(post,dtype=np.int32);self.weight=np.ascontiguousarray(weight,dtype=np.float32);self.n=len(self.ptr)-1
        self.lib=C.CDLL(str(LIB));self.lib.metal_create_deterministic_v1.restype=C.c_void_p
        self.lib.metal_create_deterministic_v1.argtypes=[C.c_int,C.c_longlong,C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p]
        self.lib.metal_destroy.argtypes=[C.c_void_p]
        self.lib.metal_step_reference_serial.argtypes=[C.c_void_p];self.lib.metal_step_reference_serial.restype=C.c_int
        self.lib.metal_set_drive.argtypes=[C.c_void_p,C.c_void_p];self.lib.metal_set_drive.restype=C.c_int
        self.lib.metal_zero_spike_counts.argtypes=[C.c_void_p];self.lib.metal_zero_spike_counts.restype=C.c_int
        self.lib.metal_load_state.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_load_state.restype=C.c_int
        self.lib.metal_copy_state.argtypes=[C.c_void_p]+[C.c_void_p]*13;self.lib.metal_copy_state.restype=C.c_int
        self.lib.metal_compact_test.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p];self.lib.metal_compact_test.restype=C.c_int
        self.ctx=self.lib.metal_create_deterministic_v1(self.n,len(self.post),self.ptr.ctypes.data,self.post.ctypes.data,self.weight.ctypes.data,str(METALLIB).encode())
        if not self.ctx: raise RuntimeError('Metal reference context creation failed')
    def close(self):
        if self.ctx:self.lib.metal_destroy(self.ctx);self.ctx=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def compact(self,active,flags):
        active=np.ascontiguousarray(active,dtype=np.int32);flags=np.ascontiguousarray(flags,dtype=np.int32);out=np.zeros(len(active),np.int32);count=np.zeros(1,np.uint32)
        if not self.lib.metal_compact_test(self.ctx,active.ctypes.data,flags.ctypes.data,len(active),out.ctypes.data,count.ctypes.data):raise RuntimeError('Metal compaction failed')
        return out[:count[0]].copy()
    def load(self,s):
        n=self.n
        fields=(s['v'],s['g'],s['refractory'],s['drive'],s['previous_drive'],s['queue'],s['queue_count'],s['counts'],s['active'],s['flags'],s['nactive'],s['last'])
        arrays=[np.ascontiguousarray(x) for x in fields];clock=np.asarray([s['clock']],np.int64)
        self._state_arrays=arrays
        if not self.lib.metal_load_state(self.ctx,*[x.ctypes.data for x in arrays],clock.ctypes.data):raise RuntimeError('Metal state load failed')
    def copy(self):
        n=self.n;s={'v':np.empty(n,np.float32),'g':np.empty(n,np.float32),'refractory':np.empty(n,np.int16),'drive':np.empty(n,np.float32),'previous_drive':np.empty(n,np.float32),'queue':np.empty((19,n),np.int32),'queue_count':np.empty(19,np.int32),'counts':np.empty(n,np.int32),'active':np.empty(n,np.int32),'flags':np.empty(n,np.uint8),'nactive':np.empty(1,np.int32),'last':np.empty(n,np.int64),'clock':0}
        fields=(s['v'],s['g'],s['refractory'],s['drive'],s['previous_drive'],s['queue'],s['queue_count'],s['counts'],s['active'],s['flags'],s['nactive'],s['last']);clock=np.zeros(1,np.int64)
        if not self.lib.metal_copy_state(self.ctx,*[x.ctypes.data for x in fields],clock.ctypes.data):raise RuntimeError('Metal state copy failed')
        s['clock']=int(clock[0]);return s
    def step(self):
        if not self.lib.metal_step_reference_serial(self.ctx):raise RuntimeError('Metal reference step failed')
    def set_drive(self,drive):
        drive=np.ascontiguousarray(drive,dtype=np.float32)
        if drive.shape!=(self.n,) or not self.lib.metal_set_drive(self.ctx,drive.ctypes.data):raise RuntimeError('Metal drive update failed')
    def zero_spike_counts(self):
        if not self.lib.metal_zero_spike_counts(self.ctx):raise RuntimeError('Metal spike-count reset failed')
