import subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
subprocess.run([str(ROOT/'upstream/doomfly/.venv-neural/bin/python'),'-m','workspace.gpu.build_accelerators'],cwd=ROOT,check=True)
from workspace.gpu.metal_v3 import MetalV3Primitives

def expected(keys,payload):
    order=np.argsort(keys,kind='stable');return keys[order],payload[order]

@pytest.mark.parametrize('keys',[
    np.array([],np.uint64),np.array([9],np.uint64),np.arange(257,dtype=np.uint64),
    np.arange(513,dtype=np.uint64)[::-1],np.array([(17<<32)|i for i in range(511)],np.uint64),
    np.array([(0xffffffff<<32)|i for i in range(513)],np.uint64),
])
def test_v3_radix_shapes(keys):
    payload=np.arange(len(keys),dtype=np.uint32)[::-1]
    with MetalV3Primitives() as gpu: got=gpu.sort64(keys,payload)
    want=expected(keys,payload);assert np.array_equal(got[0],want[0]);assert np.array_equal(got[1],want[1])

def test_v3_radix_random_and_adversarial_digits():
    rng=np.random.default_rng(9801)
    cases=[]
    for n in (2,15,255,256,257,1023,4097):
        target=rng.integers(0,2**32,size=n,dtype=np.uint64);rank=rng.integers(0,2**32,size=n,dtype=np.uint64);cases.append((target<<np.uint64(32))|rank)
    cases += [np.asarray([(i%16)<<60 | (4096-i) for i in range(4096)],np.uint64),np.asarray([((i%3)<<32)|i for i in range(4096)],np.uint64)]
    with MetalV3Primitives() as gpu:
        for keys in cases:
            payload=rng.permutation(len(keys)).astype(np.uint32);got=gpu.sort64(keys,payload);want=expected(keys,payload)
            assert np.array_equal(got[0],want[0]);assert np.array_equal(got[1],want[1])

@pytest.mark.parametrize('targets',[[],[7],[1,1,1],[1,1,7,7,9],[0,0,2,4,4,4,0xffffffff]])
def test_v3_target_segments(targets):
    keys=np.asarray([(int(t)<<32)|i for i,t in enumerate(targets)],np.uint64)
    with MetalV3Primitives() as gpu: got=gpu.segments(keys)
    if not targets:
        assert all(len(x)==0 for x in got);return
    changes=np.r_[True,np.diff(np.asarray(targets,dtype=np.uint64))!=0] if targets else np.array([],bool)
    starts=np.flatnonzero(changes).astype(np.uint32);ends=np.r_[starts[1:],len(targets)].astype(np.uint32)
    assert np.array_equal(got[0],np.asarray(targets,dtype=np.uint32)[starts]);assert np.array_equal(got[1],starts);assert np.array_equal(got[2],ends)
