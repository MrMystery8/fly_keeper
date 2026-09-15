"""End-to-end CPU versus batched-v4 benchmarks on blank and retinal workloads."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'upstream/doomfly'),str(ROOT/'workspace')]
from doom.native import NativeBrain, _f
from workspace.gpu.metal_v4 import MetalBatchedV4
from workspace.gpu.run_fixture_parity import GRAPH,snapshot

def cpu_batch(s,steps):
    clock=np.asarray([s['clock']],np.int64);a=[s[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
    _f(len(s['v']),*[x.ctypes.data for x in a],steps,.1,*[s[k].ctypes.data for k in ('counts','active','flags','nactive','last')]);s['clock']=int(clock[0])

def result(label,steps,elapsed,**extra):
    return {'backend':label,'steps':steps,'wall_seconds':elapsed,'us_per_step':elapsed*1e6/steps,'steps_per_second':steps/elapsed,'realtime_x':steps*.0001/elapsed,**extra}

def apply_frame(brain,frame):
    brain.luminance+=(1-math.exp(-2))*(np.clip(frame,0,1)-brain.luminance);brain.drive.fill(0);brain.drive[brain.lamina]=12.;brain.drive[brain.retina]=30*brain.luminance/(.02+brain.luminance)

def memory_bytes(n,e):
    # All persistent Metal allocations, excluding small ObjC driver objects.
    return (n+1)*8+e*8+n*(4*4+2+8+19*4+4+4+1+1)+19*4+4+8192+72

def run(blank_steps,widths,fixture_repeats):
    rows=[];b=NativeBrain(GRAPH);s=snapshot(b);t=time.perf_counter();cpu_batch(s,blank_steps);elapsed=time.perf_counter()-t;rows.append(result('CPU native blank batch',blank_steps,elapsed))
    for width in widths:
        b=NativeBrain(GRAPH);s=snapshot(b)
        with MetalBatchedV4(b.ptr,b.post,b.weight,width) as gpu:
            gpu.load(s);gpu.step_batch(10) # compile/cache warmup on a disposable state
        b=NativeBrain(GRAPH);s=snapshot(b)
        with MetalBatchedV4(b.ptr,b.post,b.weight,width) as gpu:
            gpu.load(s);t=time.perf_counter();gpu.step_batch(blank_steps);elapsed=time.perf_counter()-t
            rows.append(result(f'Metal batched v4 blank ({width})',blank_steps,elapsed,width=width,stats=gpu.stats()))
    frames=np.load(ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz')['luminance'];frames=np.tile(frames,(fixture_repeats,1));steps=len(frames)*200
    b=NativeBrain(GRAPH);s=snapshot(b);t=time.perf_counter()
    for frame in frames:apply_frame(b,frame);b.counts.fill(0);cpu_batch(s,200)
    cpu_elapsed=time.perf_counter()-t;rows.append(result('CPU native retinal/FlyKeeper cadence',steps,cpu_elapsed,frames=len(frames)))
    best_width=widths[0]
    for width in widths:
        b=NativeBrain(GRAPH);s=snapshot(b)
        with MetalBatchedV4(b.ptr,b.post,b.weight,width) as gpu:
            gpu.load(s);t=time.perf_counter()
            for frame in frames:apply_frame(b,frame);gpu.set_drive(b.drive);gpu.zero_spike_counts();gpu.step_batch(200);gpu.read([0])
            elapsed=time.perf_counter()-t;rows.append(result(f'Metal batched v4 retinal/FlyKeeper cadence ({width})',steps,elapsed,width=width,frames=len(frames),stats=gpu.stats(),speedup_vs_cpu=cpu_elapsed/elapsed,memory_bytes=memory_bytes(b.n,len(b.post))))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--blank-steps',type=int,default=1000);p.add_argument('--widths',default='128,256,512,1024');p.add_argument('--fixture-repeats',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.blank_steps,[int(x) for x in a.widths.split(',')],a.fixture_repeats);print(json.dumps(r,indent=2));a.output and a.output.write_text(json.dumps(r,indent=2)+'\n')
