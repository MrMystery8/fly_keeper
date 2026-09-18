"""Matched fullframe binocular action-policy evaluation with movement metrics."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path:sys.path.insert(0,str(ROOT/"workspace"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import ArcadeController,Arcade2AxisDecoder,run_episode
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_visual
from experiments.arcade_demo.arcade_action_policy import TinyPolicy,BinocularActionPolicyBridge
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
OUT=ROOT/"workspace"/"outputs"/"arcade_demo"; GROUPS=("left","center","right"); HEIGHTS=(("low",0.),("mid",.5),("high",.85))
def cells(n,seed):return [(seed+i*9+j*3+k,g,h,hf) for i,g in enumerate(GROUPS) for j,(h,hf) in enumerate(HEIGHTS) for k in range(n)]
def movement(trace):
 y=np.array([r["fly_y"] for r in trace]);z=np.array([r["fly_z"] for r in trace]); states=[r.get("movement_state") for r in trace]
 airborne=("TAKEOFF","FLIGHT","DIVE")
 # The trace is sampled after each 20 ms world step; the short TAKEOFF state
 # often advances to FLIGHT within that interval, so FLIGHT entry is the
 # robust observable takeoff event (not merely the transient label).
 return dict(mean_abs_lateral_displacement=round(float(np.abs(y-y[0]).mean()),4),peak_lateral_displacement=round(float(np.abs(y-y[0]).max()),4),mean_vertical_displacement=round(float(np.abs(z-z[0]).mean()),4),peak_vertical_displacement=round(float(np.abs(z-z[0]).max()),4),takeoff_rate=round(float(any(s in airborne for s in states)),3),airborne_steps=int(sum(s in airborne for s in states)),substantial_movement_events=int(sum(s in ("FLIGHT","GROUND_CORRECTION") and (i==0 or states[i-1]!=s) for i,s in enumerate(states))))
def evaluate(c,policy_file=None,residual_scale=.35,eye_condition="both",mode="residual"):
 brain=MaleCNSBrain(backend="metal"); results=defaultdict(lambda:[0,0]); ms=[]
 try:
  for seed,g,h,hf in c:
   world=ArcadeGoalkeeperWorld(seed=seed); visual,_=load_visual("arcade_binocular_lateral_hybrid.npz",ArcadeDNBasis()); bridge=visual
   if policy_file:
    pol,_=TinyPolicy.load(OUT/policy_file);bridge=BinocularActionPolicyBridge(visual,pol,residual_scale=residual_scale,mode=mode)
   ctrl=ArcadeController(brain,BinocularVisionBridge(world.fly,brain,condition=eye_condition,retina_map="fullframe"),Arcade2AxisDecoder(),bridge)
   out,tr=run_episode(world,ctrl,world.sample_shot(g,height_frac=hf),collect_trace=True); results[(g,h)][0]+=out["result"]=="SAVE";results[(g,h)][1]+=1;ms.append(movement(tr))
 finally: brain.close()
 n=sum(v[1] for v in results.values());return dict(overall=round(sum(v[0] for v in results.values())/n,3),n=n,by_cell={f"{g}/{h}":f"{v[0]}/{v[1]}" for (g,h),v in sorted(results.items())},movement={k:round(float(np.mean([m[k] for m in ms])),4) for k in ms[0]})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--per-cell",type=int,default=3);p.add_argument("--base-seed",type=int,default=130000);p.add_argument("--policy",required=True);p.add_argument("--residual-scale",type=float,default=.35);p.add_argument("--mode",choices=("residual","full","blend"),default="residual");p.add_argument("--eye-condition",choices=("both","left_blind","right_blind","both_blind"),default="both");a=p.parse_args();c=cells(a.per_cell,a.base_seed);out={"protocol":{"retina_map":"fullframe","eye_condition":a.eye_condition,"matched_reset_corrected":True,"n_shots":len(c),"policy_mode":a.mode,"residual_scale":a.residual_scale},"binocular_supervised":evaluate(c,eye_condition=a.eye_condition),"binocular_action_policy":evaluate(c,a.policy,a.residual_scale,a.eye_condition,a.mode)};(OUT/"arcade_binocular_action_policy_eval.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
