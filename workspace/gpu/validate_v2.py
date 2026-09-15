"""Automated staged CPU / serial-oracle / deterministic-v2 regression ladder.

The runner stops at the first state mismatch and prints its exact field/index.
Expensive stages are opt-in only through ``--through`` so a fixed passing
prefix is not repeatedly rerun while investigating a later failure.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from doom.native import NativeBrain
from workspace.gpu.metal_reference import MetalReference
from workspace.gpu.metal_v2 import MetalDeterministicV2
from workspace.gpu.run_fixture_parity import snapshot,cpu_tick,first_difference,GRAPH
from workspace.gpu.tests.test_metal_step import make_state

STAGES=('synthetic','full-1','full-10','full-100','fixture-cases','recorded')
def compare(cpu,serial,v2,step,label):
    a=serial.copy();b=v2.copy();bad=first_difference(cpu,a) or first_difference(cpu,b)
    if bad: raise AssertionError({'label':label,'step':step,'difference':bad})
def synthetic():
    # Includes convergent dynamic source ranks, long rows, refractory and
    # same-clock materialization via the fixed native step contract.
    rng=np.random.default_rng(317);n=701;ptr=[0];post=[];weight=[]
    for source in range(n):
        degree=257 if source==0 else 3
        targets=rng.choice(n,size=degree,replace=False);post.extend(targets);weight.extend(rng.uniform(.01,.2,degree));ptr.append(len(post))
    s=make_state(ptr,post,weight,active=[7,2,9,1],v=np.where(np.arange(n)%3==0,-44.,-52.));s['queue'][0,:2]=[2,0];s['queue_count'][0]=2;s['refractory'][11]=22
    with MetalReference(s['ptr'],s['post'],s['weight']) as serial,MetalDeterministicV2(s['ptr'],s['post'],s['weight']) as v2:
        serial.load(s);v2.load(s)
        for i in range(1000): cpu_tick(s);serial.step();v2.step();compare(s,serial,v2,i+1,'synthetic')
def full_steps(count):
    b=NativeBrain(GRAPH);s=snapshot(b)
    with MetalReference(b.ptr,b.post,b.weight) as serial,MetalDeterministicV2(b.ptr,b.post,b.weight) as v2:
        serial.load(s);v2.load(s)
        for i in range(count): cpu_tick(s);serial.step();v2.step();compare(s,serial,v2,i+1,f'full-{count}')
def fixture_case(luminance,ticks,label):
    b=NativeBrain(GRAPH);s=snapshot(b)
    with MetalReference(b.ptr,b.post,b.weight) as serial,MetalDeterministicV2(b.ptr,b.post,b.weight) as v2:
        serial.load(s);v2.load(s)
        for frame in [luminance]:
            alpha=1-np.exp(-ticks*.1/10);b.luminance += alpha*(np.clip(frame,0,1)-b.luminance);b.drive.fill(0);b.drive[b.lamina]=12.;b.drive[b.retina]=30*b.luminance/(.02+b.luminance);serial.set_drive(b.drive);v2.set_drive(b.drive)
            for i in range(ticks): cpu_tick(s);serial.step();v2.step();compare(s,serial,v2,i+1,label)
def run(through):
    stop=STAGES.index(through)
    for i,stage in enumerate(STAGES[:stop+1]):
        if stage=='synthetic': synthetic()
        elif stage.startswith('full-'): full_steps(int(stage.split('-')[1]))
        elif stage=='fixture-cases':
            f=np.load(ROOT/'workspace/gpu/fixtures/asymmetric_retinal_fixtures.npz')
            for name,lum in zip(f['names'],f['luminance']): fixture_case(lum,1000,str(name))
        elif stage=='recorded':
            f=np.load(ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz');b=NativeBrain(GRAPH);s=snapshot(b)
            with MetalReference(b.ptr,b.post,b.weight) as serial,MetalDeterministicV2(b.ptr,b.post,b.weight) as v2:
                serial.load(s)
                for frame_no,frame in enumerate(f['luminance']):
                    b.luminance+=(1-np.exp(-2))*(np.clip(frame,0,1)-b.luminance);b.drive.fill(0);b.drive[b.lamina]=12.;b.drive[b.retina]=30*b.luminance/(.02+b.luminance);serial.set_drive(b.drive);v2.set_drive(b.drive)
                    for tick in range(200): cpu_tick(s);serial.step();v2.step();compare(s,serial,v2,frame_no*200+tick+1,'recorded')
        print(f'PASS {stage}',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--through',choices=STAGES,default='full-100');a=p.parse_args();run(a.through)
