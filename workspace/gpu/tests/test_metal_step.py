"""Per-step CPU/native versus one-thread Metal reference parity tests."""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from doom.native import _f
from workspace.gpu.metal_reference import MetalReference

def make_state(ptr,post,weight,active=(),v=None):
    n=len(ptr)-1
    return {'ptr':np.asarray(ptr,np.int64),'post':np.asarray(post,np.int32),'weight':np.asarray(weight,np.float32),
      'v':np.full(n,-52,np.float32) if v is None else np.asarray(v,np.float32).copy(),'g':np.zeros(n,np.float32),
      'refractory':np.zeros(n,np.int16),'drive':np.zeros(n,np.float32),'previous_drive':np.zeros(n,np.float32),
      'queue':np.zeros((19,n),np.int32),'queue_count':np.zeros(19,np.int32),'counts':np.zeros(n,np.int32),
      'active':np.pad(np.asarray(active,np.int32),(0,n-len(active))),'flags':np.isin(np.arange(n),active).astype(np.uint8),
      'nactive':np.asarray([len(active)],np.int32),'last':np.full(n,-1,np.int64),'clock':0}

def cpu_step(s):
    clock=np.asarray([s['clock']],np.int64)
    arrays=[s[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
    _f(len(s['v']),*[x.ctypes.data for x in arrays],1,.1,*[s[k].ctypes.data for k in ('counts','active','flags','nactive','last')])
    s['clock']=int(clock[0])

def assert_same(cpu,gpu):
    for key in ('v','g','refractory','drive','previous_drive','queue','queue_count','counts','active','flags','nactive','last'):
        assert np.array_equal(cpu[key],gpu[key]),key
    assert cpu['clock']==gpu['clock']

def paired_steps(s,steps):
    with MetalReference(s['ptr'],s['post'],s['weight']) as metal:
        metal.load(s)
        for _ in range(steps):
            cpu_step(s);metal.step();assert_same(s,metal.copy())

def test_one_source_one_target_delayed_delivery():
    s=make_state([0,1,1],[1],[1.0],active=[0],v=[-44,-52])
    paired_steps(s,19)
    assert s['g'][1]>0

def test_dynamic_convergent_source_order_is_preserved():
    # Queue insertion rank intentionally differs from numerical source order:
    # source 2 delivers before source 0 into target 3.
    s=make_state([0,1,1,2,2],[3,3],[0.125,0.375])
    s['queue'][0,:2]=[2,0];s['queue_count'][0]=2
    paired_steps(s,1)
    assert s['g'][3]==np.float32(np.float32(.375)+np.float32(.125))

def test_refractory_target_and_ring_wraparound():
    s=make_state([0,1,1],[1],[1.0],active=[0],v=[-44,-52])
    s['refractory'][1]=22
    paired_steps(s,57)
    assert s['clock']==57

def test_random_small_graphs_thousand_steps():
    rng=np.random.default_rng(317)
    n=10
    # Unique-target source rows; this is a synthetic semantic test only.
    rows=[];post=[];weight=[]
    for src in range(n):
        targets=rng.choice(n,size=3,replace=False);rows.append(len(post));post.extend(targets);weight.extend(rng.uniform(.01,.2,3))
    rows.append(len(post))
    s=make_state(rows,post,weight,active=[7,2,9,1],v=np.where(np.arange(n)%3==0,-44.0,-52.0))
    paired_steps(s,1000)
