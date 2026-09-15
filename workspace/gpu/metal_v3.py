"""Independent ctypes harness for deterministic-v3 worklist primitives."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LIB=ROOT/'build/libmalecns_metal_v3.dylib'
METALLIB=ROOT/'build/MaleCNS_v3.metallib'

class MetalV3Primitives:
    def __init__(self):
        self.lib=C.CDLL(str(LIB));self.lib.metal_v3_primitive_create.restype=C.c_void_p;self.lib.metal_v3_primitive_create.argtypes=[C.c_char_p]
        self.lib.metal_v3_primitive_destroy.argtypes=[C.c_void_p]
        self.lib.metal_v3_sort64.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p];self.lib.metal_v3_sort64.restype=C.c_int
        self.lib.metal_v3_segment64.argtypes=[C.c_void_p,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p];self.lib.metal_v3_segment64.restype=C.c_int
        self.ctx=self.lib.metal_v3_primitive_create(str(METALLIB).encode())
        if not self.ctx: raise RuntimeError('Metal v3 primitive context creation failed')
    def close(self):
        if self.ctx:self.lib.metal_v3_primitive_destroy(self.ctx);self.ctx=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def sort64(self,keys,payload):
        keys=np.ascontiguousarray(keys,dtype=np.uint64);payload=np.ascontiguousarray(payload,dtype=np.uint32)
        if keys.ndim!=1 or payload.shape!=keys.shape: raise ValueError('keys and payload must be matching vectors')
        out_keys=np.empty_like(keys);out_payload=np.empty_like(payload)
        if not self.lib.metal_v3_sort64(self.ctx,keys.ctypes.data,payload.ctypes.data,len(keys),out_keys.ctypes.data,out_payload.ctypes.data):raise RuntimeError('Metal v3 radix sort failed')
        return out_keys,out_payload
    def segments(self,keys):
        keys=np.ascontiguousarray(keys,dtype=np.uint64);target=np.empty(len(keys),np.uint32);start=np.empty(len(keys),np.uint32);end=np.empty(len(keys),np.uint32);count=np.zeros(1,np.uint32)
        if not self.lib.metal_v3_segment64(self.ctx,keys.ctypes.data,len(keys),target.ctypes.data,start.ctypes.data,end.ctypes.data,count.ctypes.data):raise RuntimeError('Metal v3 segmentation failed')
        return target[:count[0]].copy(),start[:count[0]].copy(),end[:count[0]].copy()
