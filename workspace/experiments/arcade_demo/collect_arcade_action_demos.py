"""Collect causal, fullframe binocular demonstrations for action-policy IMT.

The collector records only policy-visible observations.  ``teacher_command``
uses simulator state, but solely to write the supervised target after the
observation has been formed; no truth field is persisted in the archive.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path: sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path: sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import Arcade2AxisDecoder, DECISION_MS, DECISION_S, MAX_DECISIONS
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.arcade_action_policy import OBS_NAMES

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS=("left", "center", "right"); HEIGHTS=(("low",0.0),("mid",.5),("high",.85))

def _obs(world, bridge, base, previous):
    fly=world.fly; vel=fly.data.qvel[fly.root_dofadr:fly.root_dofadr+3]
    stand=float(getattr(fly,"_stand_z",None) or fly.position[2])
    return np.asarray((bridge.last_pL,bridge.last_pR,base[1],previous[0],previous[1],fly.position[1],fly.position[2]-stand,vel[1],vel[2],float(getattr(fly,"_airborne",False))),np.float32)

def collect(per_cell=10, base_seed=120000, out="arcade_binocular_action_demos.npz", target_mode="residual"):
    brain=MaleCNSBrain(backend="metal"); rows=[]; targets=[]; episodes=[]; manifest=[]; seed=base_seed
    try:
      for group in GROUPS:
       for hname,hf in HEIGHTS:
        for _ in range(per_cell):
          world=ArcadeGoalkeeperWorld(seed=seed); bridge,_=load("arcade_binocular_lateral_hybrid.npz",ArcadeDNBasis()); vision=BinocularVisionBridge(world.fly,brain,condition="both",retina_map="fullframe"); decoder=Arcade2AxisDecoder(); shot=world.sample_shot(group,height_frac=hf); world.reset(shot); brain.reset(); bridge.reset(); previous=np.zeros(2)
          n=0
          while world.result is None and n<MAX_DECISIONS:
            vision.perceive(); bridge.inject(brain); brain.step(DECISION_MS); bridge.observe(brain)
            activity=brain.read(decoder.readout_ids()); command,_=decoder.decode(activity); base=np.asarray(bridge.command(),float)
            teacher=np.asarray(teacher_command(world)[0],float)
            target = teacher if target_mode == "full" else np.clip(teacher-base,-1.,1.)
            rows.append(_obs(world,bridge,base,previous)); targets.append(target); episodes.append(seed)
            world.fly.set_command(command["forward"],command.get("turn",0.),command["gait_on"],lateral=command.get("lateral",0.),vertical=command.get("vertical",0.)); world.step(DECISION_S); previous=base; n+=1
          manifest.append(dict(seed=seed,group=group,height=hname,decisions=n,result=world.result)); seed+=1
    finally: brain.close()
    np.savez(OUT/out,observations=np.asarray(rows,np.float32),targets=np.asarray(targets,np.float32),episode_seeds=np.asarray(episodes,np.int64),observation_names=np.asarray(OBS_NAMES))
    (OUT/out.replace(".npz","_manifest.json")).write_text(json.dumps(dict(model="structured fullframe binocular hybrid",retina_map="fullframe",target_mode=target_mode,teacher="privileged labels only",episodes=manifest,n_steps=len(rows)),indent=2))
    print(f"wrote {len(rows)} causal steps from {len(manifest)} episodes to {OUT/out}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--per-cell",type=int,default=10);p.add_argument("--base-seed",type=int,default=120000);p.add_argument("--out",default="arcade_binocular_action_demos.npz");p.add_argument("--target-mode",choices=("residual","full"),default="residual");a=p.parse_args();collect(a.per_cell,a.base_seed,a.out,a.target_mode)
