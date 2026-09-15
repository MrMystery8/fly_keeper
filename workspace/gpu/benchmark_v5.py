"""Benchmark and readout comparison for the throughput-first Metal backend."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'upstream/doomfly'),str(ROOT/'workspace')]
from doom.native import NativeBrain
from workspace.gpu.benchmark_v4 import apply_frame,cpu_batch,memory_bytes,result
from workspace.gpu.run_fixture_parity import GRAPH,snapshot
from workspace.gpu.metal_v5 import MetalFastV5

READOUT_IDS=(10162,10059)
def action(counts):
    left,right=map(int,counts);return 'LEFT' if left>right and left else 'RIGHT' if right>left and right else 'STAY'

def run(widths,repeats):
    frames=np.load(ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz')['luminance'];frames=np.tile(frames,(repeats,1));steps=len(frames)*200
    cpu=NativeBrain(GRAPH);s=snapshot(cpu);indices=np.asarray([np.flatnonzero(cpu.ids==x)[0] for x in READOUT_IDS],np.int32);cpu_rows=[];t=time.perf_counter()
    for frame in frames:
        apply_frame(cpu,frame);cpu.counts.fill(0);cpu_batch(s,200);c=cpu.counts[indices].copy();cpu_rows.append((c,cpu.v[indices].copy(),action(c)))
    cpu_elapsed=time.perf_counter()-t;out=[result('CPU native retinal/FlyKeeper cadence',steps,cpu_elapsed,frames=len(frames))]
    for width in widths:
        driver=NativeBrain(GRAPH);base=snapshot(driver);rows=[];t=time.perf_counter()
        with MetalFastV5(driver.ptr,driver.post,driver.weight,width) as gpu:
            # Warm the real propagation path; first-use driver/JIT costs are
            # not part of sustained simulation throughput.
            gpu.load(base);apply_frame(driver,frames[0]);gpu.set_drive(driver.drive);gpu.step_batch(200)
            gpu.load(base);driver.luminance.fill(0);driver.drive.fill(0)
            t=time.perf_counter()
            for frame in frames:
                apply_frame(driver,frame);gpu.set_drive(driver.drive);gpu.zero_spike_counts();gpu.step_batch(200);c,v=gpu.read(indices);rows.append((c.copy(),v.copy(),action(c)))
        elapsed=time.perf_counter()-t;count_diff=np.asarray([r[0]-c[0] for r,c in zip(rows,cpu_rows)]);voltage_diff=np.asarray([r[1]-c[1] for r,c in zip(rows,cpu_rows)])
        out.append(result(f'Metal fast v5 retinal/FlyKeeper cadence ({width})',steps,elapsed,width=width,frames=len(frames),speedup_vs_cpu=cpu_elapsed/elapsed,memory_bytes=memory_bytes(driver.n,len(driver.post))+driver.n*7,readout_exact_frames=int(sum(np.array_equal(r[0],c[0]) for r,c in zip(rows,cpu_rows))),action_mismatches=int(sum(r[2]!=c[2] for r,c in zip(rows,cpu_rows))),max_abs_readout_spike_difference=int(np.max(np.abs(count_diff))),max_abs_readout_voltage_difference_mv=float(np.max(np.abs(voltage_diff)))))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--widths',default='128,256,512,1024');p.add_argument('--fixture-repeats',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args();r=run([int(x) for x in a.widths.split(',')],a.fixture_repeats);print(json.dumps(r,indent=2));a.output and a.output.write_text(json.dumps(r,indent=2)+'\n')
