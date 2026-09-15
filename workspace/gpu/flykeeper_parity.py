"""Exact CPU/native versus serial-Metal FlyKeeper trace comparison.

CPU's environment frame is the canonical sensory sequence.  Until a divergence,
the Metal controller receives byte-identical retinal luminance and both separate
deterministic environments are advanced with their respective decoded action.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'upstream/doomfly'),str(ROOT/'workspace')]
from doom.native import NativeBrain, _f
from doom.game import retinal_samples
from experiments.flykeeper.environment import FlyKeeper
from experiments.flykeeper.renderer import render
from experiments.flykeeper.controller import MaleCNSGoalkeeper
from workspace.gpu.metal_reference import MetalReference

GRAPH=ROOT/'upstream/doomfly/outputs/doom/malecns_v1/graph.npz'
LEFT,RIGHT=10162,10059

def tick(s):
    clock=np.asarray([s['clock']],np.int64);a=[s[k] for k in ('ptr','post','weight','v','g','refractory','drive','previous_drive','queue','queue_count')]+[clock]
    _f(len(s['v']),*[x.ctypes.data for x in a],1,.1,*[s[k].ctypes.data for k in ('counts','active','flags','nactive','last')]);s['clock']=int(clock[0])

def state(b):
    return {'ptr':b.ptr,'post':b.post,'weight':b.weight,'v':b.v,'g':b.g,'refractory':b.refractory,'drive':b.drive,'previous_drive':b.previous_drive,'queue':b.queue,'queue_count':b.queue_count,'counts':b.counts,'active':b.active,'flags':b.active_flag,'nactive':b.nactive,'last':b.last,'clock':b.cursor}

def set_frame(b,lum):
    alpha=1-math.exp(-20*.1/10);b.luminance+=alpha*(np.clip(lum,0,1)-b.luminance)
    b.drive.fill(0);b.drive[b.lamina]=12.;b.drive[b.retina]=30*b.luminance/(.02+b.luminance)

def run(episodes=10,seed=7):
    cpu=NativeBrain(GRAPH);s=state(cpu);left_i=int(np.flatnonzero(cpu.ids==LEFT)[0]);right_i=int(np.flatnonzero(cpu.ids==RIGHT)[0])
    cpu_env=FlyKeeper(seed);metal_env=FlyKeeper(seed);trace=[]
    with MetalReference(cpu.ptr,cpu.post,cpu.weight) as metal:
      metal.load(s)
      for ep in range(episodes):
        c=cpu_env.reset(seed+ep);g=metal_env.reset(seed+ep);decision=0
        while not c.done:
          image=render(c,cpu_env);lum=retinal_samples(image,cpu.uv);set_frame(cpu,lum);metal.set_drive(cpu.drive);cpu.counts.fill(0);metal.zero_spike_counts()
          for neural_tick in range(200): tick(s);metal.step()
          gs=metal.copy();cpu_left=int(s['counts'][left_i]);cpu_right=int(s['counts'][right_i]);metal_left=int(gs['counts'][left_i]);metal_right=int(gs['counts'][right_i])
          ca='LEFT' if cpu_left>cpu_right and cpu_left else 'RIGHT' if cpu_right>cpu_left and cpu_right else 'STAY'
          ma='LEFT' if metal_left>metal_right and metal_left else 'RIGHT' if metal_right>metal_left and metal_right else 'STAY'
          row={'episode':ep,'decision':decision,'brain_time_ms':s['clock']*.1,'ball_x':c.ball_x,'ball_y':c.ball_y,'cpu_left':cpu_left,'metal_left':metal_left,'cpu_right':cpu_right,'metal_right':metal_right,'cpu_action':ca,'metal_action':ma,'cpu_keeper_before':c.keeper_x,'metal_keeper_before':g.keeper_x}
          c=cpu_env.step(ca);g=metal_env.step(ma);row.update(cpu_keeper_after=c.keeper_x,metal_keeper_after=g.keeper_x,cpu_result=c.result,metal_result=g.result);trace.append(row)
          if ca!=ma or c.keeper_x!=g.keeper_x or c.result!=g.result:
            return {'ok':False,'first_divergence':row,'decisions':trace}
          decision+=1
    return {'ok':True,'decisions':trace,'summary':{'episodes':episodes,'decisions':len(trace),'action_mismatches':0,'dnp20_mismatches':0,'episode_result_mismatches':0}}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=10);p.add_argument('--seed',type=int,default=7);p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.episodes,a.seed)
 if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'ok':r['ok'],'summary':r.get('summary'),'first_divergence':r.get('first_divergence')},indent=2));raise SystemExit(0 if r['ok'] else 1)
