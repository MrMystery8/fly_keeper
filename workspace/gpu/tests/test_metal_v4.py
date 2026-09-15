import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
subprocess.run([str(ROOT/'upstream/doomfly/.venv-neural/bin/python'),'-m','workspace.gpu.build_accelerators'],cwd=ROOT,check=True)
from workspace.gpu.metal_v4 import MetalBatchedV4
from workspace.gpu.run_fixture_parity import cpu_tick,first_difference,snapshot,GRAPH
from workspace.gpu.tests.test_metal_step import make_state
from doom.native import NativeBrain, _f

def _cpu_batch(s,steps):
    clock=np.asarray([s['clock']],np.int64)
    arrays=[s[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
    _f(len(s['v']),*[x.ctypes.data for x in arrays],steps,.1,*[s[k].ctypes.data for k in ('counts','active','flags','nactive','last')])
    s['clock']=int(clock[0])

def test_v4_synthetic_tick_exact():
    rng=np.random.default_rng(417);n=701;ptr=[0];post=[];weight=[]
    for source in range(n):
        degree=257 if source==0 else 3
        post.extend(rng.choice(n,size=degree,replace=False));weight.extend(rng.uniform(.01,.2,degree));ptr.append(len(post))
    s=make_state(ptr,post,weight,active=[7,2,9,1],v=np.where(np.arange(n)%3==0,-44.,-52.));s['queue'][0,:2]=[2,0];s['queue_count'][0]=2;s['refractory'][11]=22
    with MetalBatchedV4(s['ptr'],s['post'],s['weight']) as gpu:
        gpu.load(s)
        for _ in range(100):cpu_tick(s);gpu.step();assert first_difference(s,gpu.copy()) is None

def test_v4_full_graph_batch_matches_native():
    brain=NativeBrain(GRAPH);s=snapshot(brain)
    with MetalBatchedV4(brain.ptr,brain.post,brain.weight) as gpu:
        gpu.load(s);_cpu_batch(s,50);gpu.step_batch(50)
        assert first_difference(s,gpu.copy()) is None

def test_v4_recorded_retinal_batches_match_native():
    brain=NativeBrain(GRAPH);s=snapshot(brain)
    frames=np.load(ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz')['luminance']
    with MetalBatchedV4(brain.ptr,brain.post,brain.weight) as gpu:
        gpu.load(s)
        for frame in frames:
            brain.luminance+=(1-np.exp(-2))*(np.clip(frame,0,1)-brain.luminance)
            brain.drive.fill(0);brain.drive[brain.lamina]=12.;brain.drive[brain.retina]=30*brain.luminance/(.02+brain.luminance)
            brain.counts.fill(0);gpu.set_drive(brain.drive);gpu.zero_spike_counts()
            _cpu_batch(s,200);gpu.step_batch(200)
            assert first_difference(s,gpu.copy()) is None
