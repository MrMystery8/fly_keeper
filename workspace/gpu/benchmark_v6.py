"""Sustained sparse-v6 benchmark with CPU readout/action comparison."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'upstream/doomfly'),str(ROOT/'workspace')]
from doom.native import NativeBrain
from workspace.gpu.benchmark_v4 import apply_frame,cpu_batch,memory_bytes,result
from workspace.gpu.benchmark_v5 import action,READOUT_IDS
from workspace.gpu.run_fixture_parity import GRAPH,snapshot
from workspace.gpu.metal_v6 import MetalSparseV6
def run(widths,repeats):
 frames=np.load(ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz')['luminance'];frames=np.tile(frames,(repeats,1));steps=len(frames)*200;cpu=NativeBrain(GRAPH);s=snapshot(cpu);idx=np.asarray([np.flatnonzero(cpu.ids==x)[0] for x in READOUT_IDS],np.int32);ref=[];t=time.perf_counter()
 for frame in frames:apply_frame(cpu,frame);cpu.counts.fill(0);cpu_batch(s,200);c=cpu.counts[idx].copy();ref.append((c,cpu.v[idx].copy(),action(c)))
 ce=time.perf_counter()-t;out=[result('CPU native retinal/FlyKeeper cadence',steps,ce,frames=len(frames))]
 for width in widths:
  b=NativeBrain(GRAPH);base=snapshot(b)
  with MetalSparseV6(b.ptr,b.post,b.weight,width) as gpu:
   gpu.load(base);apply_frame(b,frames[0]);gpu.set_drive(b.drive);gpu.step_batch(200);gpu.load(base);b.luminance.fill(0);b.drive.fill(0);rows=[];t=time.perf_counter()
   for frame in frames:apply_frame(b,frame);gpu.set_drive(b.drive);gpu.zero_spike_counts();gpu.step_batch(200);c,v=gpu.read(idx);rows.append((c.copy(),v.copy(),action(c)))
   elapsed=time.perf_counter()-t
  cd=np.asarray([a[0]-z[0] for a,z in zip(rows,ref)]);vd=np.asarray([a[1]-z[1] for a,z in zip(rows,ref)]);out.append(result(f'Metal sparse v6 ({width})',steps,elapsed,width=width,frames=len(frames),speedup_vs_cpu=ce/elapsed,memory_bytes=memory_bytes(b.n,len(b.post))+b.n*16,readout_exact_frames=int(sum(np.array_equal(a[0],z[0]) for a,z in zip(rows,ref))),action_mismatches=int(sum(a[2]!=z[2] for a,z in zip(rows,ref))),max_abs_readout_spike_difference=int(np.max(np.abs(cd))),max_abs_readout_voltage_difference_mv=float(np.max(np.abs(vd)))))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--widths',default='256,512,1024');p.add_argument('--fixture-repeats',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args();r=run([int(x) for x in a.widths.split(',')],a.fixture_repeats);print(json.dumps(r,indent=2));a.output and a.output.write_text(json.dumps(r,indent=2)+'\n')
