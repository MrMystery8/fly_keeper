"""Dependency-free bounded residual policy search for Arcade.

This is a small evolution-strategy RL fine-tuner, used when Torch/PPO is not
available in the project environment.  It optimizes only four residual policy
parameters against terminal SAVE=+10 / GOAL=-10 reward on curriculum shots.
The visual policy, MaleCNS, DNs and force-driven body stay frozen.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path:sys.path.insert(0,str(ROOT/"workspace"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import ArcadeController,Arcade2AxisDecoder,run_episode
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_visual
from experiments.arcade_demo.arcade_action_policy import TinyPolicy,BinocularActionPolicyBridge
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
OUT=ROOT/"workspace"/"outputs"/"arcade_demo"

class ESBridge(BinocularActionPolicyBridge):
    """Frozen supervised residual plus four bounded trainable ES parameters."""
    def __init__(self, visual, policy, theta):
        super().__init__(visual,policy,residual_scale=.2,mode="residual");self.theta=np.asarray(theta,float)
    def command(self,world):
        lat,vert=super().command(world)
        # theta=[lat gain, vert gain, lat bias, vert bias], all bounded.
        action=np.clip(np.array([self.theta[0]*lat+self.theta[2],self.theta[1]*vert+self.theta[3]]),[-1,0],[1,1])
        self.visual_bridge._ema=action;self.previous=action;return float(action[0]),float(action[1])

def score(theta,seeds,groups=("left","right"),height=.5,policy_file="arcade_binocular_action_policy.npz"):
    brain=MaleCNSBrain(backend="metal"); total=0.
    try:
      policy,_=TinyPolicy.load(OUT/policy_file)
      for seed in seeds:
       for group in groups:
        w=ArcadeGoalkeeperWorld(seed=seed); visual,_=load_visual("arcade_binocular_lateral_hybrid.npz",ArcadeDNBasis()); bridge=ESBridge(visual,policy,theta)
        shot=w.sample_shot(group,height_frac=height)
        # Privileged target is read for reward shaping only, before rollout.
        # It is never exposed to ESBridge or saved in an observation.
        w.reset(shot); target=np.asarray(teacher_command(w)[0],float)
        c=ArcadeController(brain,BinocularVisionBridge(w.fly,brain,condition="both",retina_map="fullframe"),Arcade2AxisDecoder(),bridge);out,_=run_episode(w,c,shot)
        terminal=10 if out["result"]=="SAVE" else -10
        # Small terminal-subordinate shaping supplies ES signal on early MID
        # curriculum failures; clip prevents it from outweighing a save.
        action=bridge.previous
        shaping=1.5*(1.-min(1.,float(np.abs(action-target).mean())))
        total += terminal+shaping
    finally:brain.close()
    return total/(len(seeds)*len(groups))

def train(generations=8,population=8,seed=20260917):
    rng=np.random.default_rng(seed); mean=np.array([1.,1.,0.,0.]);sigma=np.array([.15,.15,.08,.08]);history=[];seeds=list(range(230000,230003))
    for generation in range(generations):
      candidates=np.clip(mean+rng.normal(size=(population,4))*sigma,[.4,.4,-.4,-.25],[1.6,1.6,.4,.25]);rewards=np.array([score(t,seeds) for t in candidates]);best=np.argsort(rewards)[-max(2,population//3):];mean=candidates[best].mean(0);sigma=np.maximum(.03,sigma*.8);history.append(dict(generation=generation,best=float(rewards.max()),mean=float(rewards.mean()),theta=mean.tolist()))
      print(history[-1],flush=True)
    result=dict(method="bounded ES residual RL",reward="SAVE +10, GOAL -10 plus <=+1.5 teacher-distance shaping",curriculum="left/right mid only",theta=mean.tolist(),history=history);(OUT/"arcade_binocular_residual_es.json").write_text(json.dumps(result,indent=2));return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--generations",type=int,default=8);p.add_argument("--population",type=int,default=8);a=p.parse_args();train(a.generations,a.population)
