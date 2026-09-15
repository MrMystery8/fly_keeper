"""Repeatable wall-clock benchmark for fixed-weight CPU, serial Metal and v2."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'upstream/doomfly'))
from doom.native import NativeBrain
from workspace.gpu.run_fixture_parity import snapshot,cpu_tick,GRAPH
from workspace.gpu.metal_reference import MetalReference
from workspace.gpu.metal_v2 import MetalDeterministicV2

def time_steps(label,steps,fn):
    t=time.perf_counter()
    for _ in range(steps): fn()
    elapsed=time.perf_counter()-t
    return {'backend':label,'steps':steps,'wall_seconds':elapsed,'us_per_step':elapsed*1e6/steps,'steps_per_second':steps/elapsed,'realtime_x':(steps*.0001)/elapsed}
def run(steps,widths):
    b=NativeBrain(GRAPH);base=snapshot(b);results=[]
    s={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()};results.append(time_steps('CPU native',steps,lambda:cpu_tick(s)))
    with MetalReference(b.ptr,b.post,b.weight) as m:
        m.load(base);results.append(time_steps('Metal reference serial',steps,m.step))
    for width in widths:
        with MetalDeterministicV2(b.ptr,b.post,b.weight,width) as m:
            m.load(base);results.append({**time_steps(f'Metal deterministic v2 ({width})',steps,m.step),'width':width})
    degrees=np.diff(b.ptr)
    results.append({'workload':{'mean_delayed_sources_per_step':0.0,'median_delayed_sources_per_step':0.0,'p95_delayed_sources_per_step':0.0,'max_delayed_sources_per_step':0,'mean_source_out_degree':float(degrees.mean()),'median_source_out_degree':float(np.median(degrees)),'p95_source_out_degree':float(np.quantile(degrees,.95)),'max_source_out_degree':int(degrees.max()),'metal_memory_bytes':int((len(b.ptr)*8+len(b.post)*4+len(b.weight)*4)+(len(b.v)*4*4+len(b.refractory)*2+len(b.last)*8+19*len(b.v)*4+19*4+len(b.v)*4*3+len(b.post)))}})
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);p.add_argument('--widths',default='32,64,128,256');p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.steps,[int(x) for x in a.widths.split(',')]);print(json.dumps(r,indent=2));a.output and a.output.write_text(json.dumps(r,indent=2)+'\n')
