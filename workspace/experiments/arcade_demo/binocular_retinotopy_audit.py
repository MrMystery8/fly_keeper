"""Geometric/rendered audit of eye-camera orientation and receptor routing."""
from __future__ import annotations
import json,sys
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'workspace') not in sys.path:sys.path.insert(0,str(ROOT/'workspace'))
if str(ROOT/'upstream'/'doomfly') not in sys.path:sys.path.insert(0,str(ROOT/'upstream'/'doomfly'))
from adapters.brain import MaleCNSBrain
from doom.game import retinal_samples
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_S
from experiments.arcade_demo.binocular_vision import load_eye_manifest
OUT=ROOT/'workspace'/'outputs'/'arcade_demo'
def centroid(diff):
 w=diff.mean(2); y,x=np.indices(w.shape); tot=w.sum();return None if tot<1e-6 else [float((x*w).sum()/tot),float((y*w).sum()/tot),float(tot)]
def render_without_ball(w,r,cam):
 q=w.data.qpos.copy();q[w._ball_qadr:w._ball_qadr+3]=[100,100,100];mujoco.mj_forward(w.model,w.data);r.update_scene(w.data,camera=cam);im=r.render().copy();w.data.qpos[:]=q;mujoco.mj_forward(w.model,w.data);return im
def run(seed=140000,steps=12):
 b=MaleCNSBrain(backend='cpu');man=load_eye_manifest();uv=man['uv'];lm=man['left_mask'];rm=man['right_mask'];cams={}
 # Camera quaternions are recorded from loaded MuJoCo model, not assumed.
 w=ArcadeGoalkeeperWorld(seed=seed);r=mujoco.Renderer(w.model,height=96,width=160)
 for name in ('eye_left','eye_right'):
  cid=mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_CAMERA,name);cams[name]=dict(pos=w.model.cam_pos[cid].tolist(),quat=w.model.cam_quat[cid].tolist(),fovy=float(w.model.cam_fovy[cid]))
 rows=[];s=seed
 for group in ('left','center','right'):
  w=ArcadeGoalkeeperWorld(seed=s);w.reset(w.sample_shot(group,height_frac=.5));r=mujoco.Renderer(w.model,height=96,width=160)
  for _ in range(steps):w.fly.set_command(0,0,0,lateral=0,vertical=0);w.step(DECISION_S)
  entry={'group':group,'world_ball_y':float(w._observe_ball()['pos'][1]),'eyes':{}}
  for cam,mask in (('eye_left',lm),('eye_right',rm)):
   bg=render_without_ball(w,r,cam);r.update_scene(w.data,camera=cam);im=r.render().copy();d=np.abs(im.astype(float)-bg.astype(float));c=centroid(d);lum=retinal_samples(im,uv);base=retinal_samples(bg,uv);delta=np.maximum(lum-base,0);weighted=np.average(uv[mask],axis=0,weights=delta[mask]+1e-8)
   entry['eyes'][cam]=dict(ball_image_centroid_px=c,receptor_delta_uv_centroid=weighted.tolist(),receptor_delta_mean=float(delta[mask].mean()),receptor_delta_sum=float(delta[mask].sum()))
  rows.append(entry);s+=1
 b.close();out=dict(camera_geometry=cams,retinal_mapping='left rootSide receptors sample eye_left; right rootSide receptors sample eye_right',uv_orientation='u increases image-right and v increases image-down in retinal_samples',shots=rows)
 (OUT/'binocular_retinotopy_audit.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
