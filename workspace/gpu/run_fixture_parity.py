"""CPU/native versus serial-Metal parity using recorded retinal luminance only.

No FlyKeeper environment or renderer is instantiated here. The NPZ is the
sole sensory input, while the documented NativeBrain retinal/lamina transform
is reproduced before both implementations advance one 0.1-ms tick at a time.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from doom.native import NativeBrain, _f
from workspace.gpu.metal_reference import MetalReference

GRAPH=ROOT/'upstream/doomfly/outputs/doom/malecns_v1/graph.npz'
FIXTURE=ROOT/'workspace/gpu/fixtures/flykeeper_retinal_sequence.npz'

def cpu_tick(s):
    clock=np.asarray([s['clock']],np.int64)
    arrays=[s[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
    _f(len(s['v']),*[x.ctypes.data for x in arrays],1,.1,*[s[k].ctypes.data for k in ('counts','active','flags','nactive','last')])
    s['clock']=int(clock[0])

def snapshot(brain):
    return {'ptr':brain.ptr,'post':brain.post,'weight':brain.weight,'v':brain.v,'g':brain.g,'refractory':brain.refractory,
      'drive':brain.drive,'previous_drive':brain.previous_drive,'queue':brain.queue,'queue_count':brain.queue_count,
      'counts':brain.counts,'active':brain.active,'flags':brain.active_flag,'nactive':brain.nactive,'last':brain.last,'clock':brain.cursor}

def first_difference(cpu,gpu):
    for key in ('v','g','refractory','queue','queue_count','counts','active','flags','nactive','last'):
        if not np.array_equal(cpu[key],gpu[key]):
            a,b=cpu[key],gpu[key];where=np.flatnonzero(np.asarray(a).ravel()!=np.asarray(b).ravel())
            return {'field':key,'flat_index':int(where[0]) if len(where) else None,'cpu':np.asarray(a).ravel()[where[0]].item() if len(where) else None,'metal':np.asarray(b).ravel()[where[0]].item() if len(where) else None}
    return None

def run(max_steps,fixture=FIXTURE,frame_index=None,repeat_frames=1):
    f=np.load(fixture);lum=f['luminance'];
    if frame_index is not None: lum=lum[frame_index:frame_index+1]
    if repeat_frames>1: lum=np.tile(lum,(repeat_frames,1))
    dt=float(f['dt_ms']);ticks=int(round(dt/.1));b=NativeBrain(GRAPH);s=snapshot(b);total=0
    with MetalReference(b.ptr,b.post,b.weight) as metal:
        metal.load(s)
        for frame in lum:
            # Exact NativeBrain fixed-baseline visual current transformation.
            alpha=1-np.exp(-ticks*.1/10);b.luminance+=alpha*(np.clip(frame,0,1)-b.luminance)
            b.drive.fill(0);b.drive[b.lamina]=12.;b.drive[b.retina]=30*b.luminance/(.02+b.luminance)
            metal.set_drive(b.drive);b.counts.fill(0);metal.zero_spike_counts()
            for _ in range(ticks):
                if total>=max_steps:return {'ok':True,'steps':total,'fixture_frames':int(np.ceil(total/ticks))}
                cpu_tick(s);metal.step();total+=1;got=metal.copy();bad=first_difference(s,got)
                if bad:return {'ok':False,'steps':total,'fixture_frames':int(np.ceil(total/ticks)),'difference':bad}
    return {'ok':True,'steps':total,'fixture_frames':len(lum)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-steps',type=int,default=100);p.add_argument('--fixture',type=Path,default=FIXTURE);p.add_argument('--frame-index',type=int);p.add_argument('--repeat-frames',type=int,default=1);p.add_argument('--result',type=Path);a=p.parse_args();result=run(a.max_steps,a.fixture,a.frame_index,a.repeat_frames)
    if a.result:a.result.write_text(json.dumps(result,indent=2)+'\n')
    print(result);raise SystemExit(0 if result['ok'] else 1)
