"""Playable fullframe-binocular Arcade goalkeeper.

Launch from the repository root:
    PYTHONPATH=workspace upstream/doomfly/.venv-neural/bin/python \
      -m experiments.arcade_demo.play_arcade_goalkeeper

Controls: LEFT/RIGHT choose shot side, UP/DOWN choose height, SPACE fires,
R resets the score, Q/ESC quits.  Every shot uses the selected two-eye
MaleCNS -> action policy -> real DN -> force-driven Arcade body path.
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path:sys.path.insert(0,str(ROOT/"workspace"))
if str(ROOT/"upstream"/"doomfly") not in sys.path:sys.path.insert(0,str(ROOT/"upstream"/"doomfly"))

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import ArcadeController,Arcade2AxisDecoder,DECISION_MS,DECISION_S,MAX_DECISIONS
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_visual
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import EarlyIntentBridge, ExecPolicy


class ArcadeGame:
    W,H=1280,860
    SIDES=("left","center","right")
    HEIGHTS=(("low",0.0),("mid",.5),("high",.85))
    # Named MuJoCo cameras are deterministic.  They deliberately include
    # opposite field angles so the player can choose the useful perspective.
    CAMERAS=(("presentation","CLASSIC GAME VIEW"),("track1","TRACK 1"),("track2","TRACK 2"),("track3","HIGH SIDE"),
             ("side","SIDE"),("hero","HERO"),("back","REVERSE"))

    def __init__(self, policy="arcade_early_intent_vert_refined_rl.npz", sense_steps=8):
        import pygame,mujoco
        self.pg,self.mujoco=pygame,mujoco;pygame.init()
        self.screen=pygame.display.set_mode((self.W,self.H));pygame.display.set_caption("Arcade Binocular Fly Goalkeeper (Early-Intent RL)")
        self.font=pygame.font.SysFont("Menlo,Consolas,monospace",18);self.small=pygame.font.SysFont("Menlo,Consolas,monospace",15);self.big=pygame.font.SysFont("Menlo,Consolas,monospace",32,bold=True)
        self.clock=pygame.time.Clock();self.policy_file=policy;self.sense_steps=int(sense_steps);self.seed=400000
        self.side_i=1;self.height_i=1;self.camera_i=0;self.active=False;self.decisions=0;self.last=None;self.saves=0;self.goals=0
        self.last_command={};self.last_diag={}
        self.map_ids=[];self.map_xy=np.empty((0,2));self.map_kind=[];self.map_spikes=np.empty(0,dtype=int)
        self.brain=MaleCNSBrain(backend="metal");self.world=ArcadeGoalkeeperWorld(seed=self.seed)
        self.renderer=mujoco.Renderer(self.world.model,height=400,width=640)
        self.presentation_camera=self._make_presentation_camera()
        self._build_controller()

    def _make_presentation_camera(self):
        """The original interactive game's goal-line presentation camera."""
        cam=self.mujoco.MjvCamera()
        cam.type=self.mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=(0.9,0.0,0.35)
        cam.distance=5.2
        cam.azimuth=180.0
        cam.elevation=-18.0
        return cam

    def _build_controller(self):
        visual,_=load_visual("arcade_binocular_lateral_hybrid.npz",ArcadeDNBasis())
        est,_=EarlyIntentEstimator.load()
        policy,_=ExecPolicy.load(ROOT/"workspace/outputs/arcade_demo"/self.policy_file)
        self.bridge=EarlyIntentBridge(visual,est,policy=policy,sense_steps=self.sense_steps)
        self.vision=BinocularVisionBridge(self.world.fly,self.brain,condition="both",retina_map="fullframe")
        self.decoder=Arcade2AxisDecoder()
        self.controller=ArcadeController(self.brain,self.vision,self.decoder,self.bridge)
        self._build_brain_map()

    def _build_brain_map(self):
        """Cache real annotated soma positions for the live visual-to-DN path."""
        import pandas as pd
        visual_ids=list(self.controller.bridge.body_ids)
        dn_ids=list(self.controller.decoder.readout_ids())
        ids=list(dict.fromkeys(visual_ids+dn_ids))
        ann_path=ROOT/"upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
        ann=pd.read_feather(ann_path,columns=["bodyId","somaLocation"]).set_index("bodyId")
        xy=[];kept=[];kind=[]
        for body_id in ids:
            loc=ann.loc[body_id,"somaLocation"] if body_id in ann.index else None
            if isinstance(loc,str):
                try: loc=ast.literal_eval(loc)
                except (SyntaxError,ValueError): loc=None
            if loc is None or not hasattr(loc,"__len__") or len(loc)<3: continue
            loc=np.asarray(loc,float)
            if not np.isfinite(loc).all(): continue
            # X–Z is an anatomical soma projection, not a fabricated layout.
            kept.append(body_id);xy.append((loc[0],loc[2]));kind.append("DN" if body_id in dn_ids else "VIS")
        self.map_ids=kept;self.map_xy=np.asarray(xy,float);self.map_kind=kind
        self.map_spikes=np.zeros(len(kept),dtype=int)

    def _update_brain_map(self):
        if not self.map_ids:return
        activity=self.brain.read(self.map_ids)
        self.map_spikes=np.asarray([activity[body_id]["spikes"] for body_id in self.map_ids],dtype=int)

    def fire(self):
        if self.active:return
        self.seed+=1;self.world=ArcadeGoalkeeperWorld(seed=self.seed);self.renderer.close();self.renderer=self.mujoco.Renderer(self.world.model,height=400,width=640)
        self._build_controller();side=self.SIDES[self.side_i];_,hf=self.HEIGHTS[self.height_i]
        self.world.reset(self.world.sample_shot(side,height_frac=hf));self.brain.reset();self.controller.reset();self.active=True;self.decisions=0;self.last=None;self.last_command={};self.last_diag={}

    def step(self):
        if not self.active:return
        self.vision.perceive()
        self.bridge.inject(self.brain)
        step_info=self.brain.step(DECISION_MS)
        self.bridge.observe(self.brain)
        activity=self.brain.read(self.decoder.readout_ids())
        _,diag=self.decoder.decode(activity)
        g=self.bridge.gait_on()
        u_lat,u_vert=self.bridge.command(self.world)
        self.last_command={"forward":0.,"lateral":u_lat,"vertical":u_vert,"gait_on":g}
        diag["bridge_u_lat"]=u_lat
        diag["bridge_u_vert"]=u_vert
        diag["intent_lat"]=self.bridge.intent.intent_lat
        diag["confidence"]=self.bridge.intent.confidence
        diag["sensing"]=self.bridge.sensing
        diag["spikes"]=step_info.get("spikes",0)
        self.last_diag=diag
        self._update_brain_map()
        self.world.fly.set_command(0.,0.,g,lateral=u_lat,vertical=u_vert)
        self.world.step(DECISION_S);self.decisions+=1
        if self.world.result is not None or self.decisions>=MAX_DECISIONS:
            self.active=False;self.last=self.world.result or "GOAL"
            if self.last=="SAVE":self.saves+=1
            else:self.goals+=1

    def draw(self):
        # A named camera gives stable framing while C/1–6 lets the player pick
        # the field angle.  The MuJoCo free camera is not used because its pose
        # is arbitrary between scenes.
        camera,camera_label=self.CAMERAS[self.camera_i]
        render_camera=self.presentation_camera if camera=="presentation" else camera
        self.renderer.update_scene(self.world.data,camera=render_camera);rgb=self.renderer.render()
        surf=self.pg.surfarray.make_surface(np.transpose(rgb,(1,0,2)));surf=self.pg.transform.scale(surf,(840,525))
        self.screen.fill((8,13,22));self.pg.draw.rect(self.screen,(26,43,61),(20,100,850,535),border_radius=8);self.screen.blit(surf,(25,105))
        side=self.SIDES[self.side_i].upper();height=self.HEIGHTS[self.height_i][0].upper()
        self._text("ARCADE BINOCULAR EARLY-INTENT GOALKEEPER",24,18,self.big,(125,220,255))
        self._text("FULLFRAME EYES  →  MaleCNS  →  EARLY INTENT  →  RL POLICY  →  REAL DNs  →  PHYSICS",26,60,self.small,(205,222,240))
        status="SHOT ACTIVE" if self.active else (self.last or "READY")
        self._draw_dashboard(side,height,status,camera_label)
        self._draw_brain_map()
        self._text(f"Aim: {side}    Height: {height}    Status: {status}",24,655,self.font,(255,230,130))
        self._text(f"Saves {self.saves}   Goals {self.goals}     [SPACE] fire   [←/→] aim   [↑/↓] height   [C / 1–7] camera   [R] reset   [Q] quit",24,710,self.font,(220,230,240))
        self._text("Every save is a physical interception; contact alone does not count.",24,740,self.small,(146,168,190))
        self.pg.display.flip()

    def _draw_dashboard(self,side,height,status,camera_label):
        x,y,w,h=890,100,370,535
        self.pg.draw.rect(self.screen,(17,29,43),(x,y,w,h),border_radius=8)
        self.pg.draw.rect(self.screen,(50,85,110),(x,y,w,h),2,border_radius=8)
        self._text("LIVE TELEMETRY",x+16,y+16,self.font,(125,220,255))
        color=(116,240,160) if self.active else ((255,205,100) if status=="READY" else (255,130,130))
        self._text(f"● {status}",x+16,y+48,self.font,color)
        phase_str="SENSING" if self.last_diag.get("sensing",False) else "EXECUTION"
        rows=[
            ("SHOT",f"{side} / {height}"),
            ("PHASE",f"{phase_str} (step {self.decisions:02d}/{MAX_DECISIONS})"),
            ("EARLY INTENT",f"{self.last_diag.get('intent_lat',0.):+.3f} (c={self.last_diag.get('confidence',0.):.2f})"),
            ("EYES",f"L {self.last_diag.get('left_lum_mean',0):.3f}   R {self.last_diag.get('right_lum_mean',0):.3f}"),
            ("RETINA SPIKES",str(self.last_diag.get("spikes",0))),
            ("LATERAL CMD",f"{self.last_command.get('lateral',0):+.3f}"),
            ("VERTICAL CMD",f"{self.last_command.get('vertical',0):+.3f}"),
            ("GAIT DRIVE",f"{self.last_command.get('gait_on',0):.1f}"),
        ]
        pos=self.world.fly.position;vel=self.world.fly.velocity
        rows += [("FLY POSITION",f"y {pos[1]:+.3f}  z {pos[2]:.3f}"),("FLY VELOCITY",f"y {vel[1]:+.3f}  z {vel[2]:+.3f}"),("MOVEMENT",str(getattr(self.world.fly,"state","grounded")).upper())]
        yy=y+86
        for label,value in rows:
            self._text(label,x+16,yy,self.small,(138,167,194));self._text(value,x+184,yy,self.small,(235,242,248));yy+=34
        self.pg.draw.line(self.screen,(50,85,110),(x+16,yy+4),(x+w-16,yy+4),1)
        self._text("MODEL",x+16,yy+19,self.small,(138,167,194))
        self._text("early-intent RL • 53.7% honest saves",x+16,yy+42,self.small,(180,205,225))
        self._text("CAMERA",x+16,yy+69,self.small,(138,167,194))
        self._text(f"{camera_label}  •  [C] next  •  [1–7] choose",x+16,yy+92,self.small,(180,205,225))

    def _draw_brain_map(self):
        x,y,w,h=890,655,370,180
        self.pg.draw.rect(self.screen,(17,29,43),(x,y,w,h),border_radius=8)
        self.pg.draw.rect(self.screen,(50,85,110),(x,y,w,h),2,border_radius=8)
        self._text("LIVE BRAIN MAP",x+14,y+12,self.font,(125,220,255))
        self._text("annotated soma locations • X–Z projection",x+14,y+36,self.small,(138,167,194))
        if len(self.map_xy)==0:
            self._text("No annotated pathway somas available",x+14,y+78,self.small,(220,230,240));return
        pad=18;top=y+58;bottom=y+h-14
        lo=self.map_xy.min(0);span=np.maximum(self.map_xy.max(0)-lo,1.0)
        pts=(self.map_xy-lo)/span
        for (px,py),spikes,kind in zip(pts,self.map_spikes,self.map_kind):
            sx=int(x+pad+px*(w-2*pad));sy=int(bottom-py*(bottom-top))
            if spikes:
                color=(255,190,75) if kind=="DN" else (100,235,255)
                radius=min(6,2+int(spikes))
            else:
                color=(116,74,45) if kind=="DN" else (40,78,99);radius=2
            self.pg.draw.circle(self.screen,color,(sx,sy),radius)
        active=int((self.map_spikes>0).sum());total=len(self.map_ids)
        self._text(f"{active}/{total} firing now",x+w-145,y+12,self.small,(235,242,248))
        self._text("● visual pathway",x+14,y+h-20,self.small,(100,235,255))
        self._text("● descending neurons",x+175,y+h-20,self.small,(255,190,75))

    def _text(self,text,x,y,font,color):self.screen.blit(font.render(text,True,color),(x,y))

    def run(self):
        running=True
        while running:
            for event in self.pg.event.get():
                if event.type==self.pg.QUIT:running=False
                if event.type==self.pg.KEYDOWN:
                    if event.key in (self.pg.K_ESCAPE,self.pg.K_q):running=False
                    elif event.key==self.pg.K_SPACE:self.fire()
                    elif event.key==self.pg.K_LEFT:self.side_i=max(0,self.side_i-1)
                    elif event.key==self.pg.K_RIGHT:self.side_i=min(2,self.side_i+1)
                    elif event.key==self.pg.K_UP:self.height_i=min(2,self.height_i+1)
                    elif event.key==self.pg.K_DOWN:self.height_i=max(0,self.height_i-1)
                    elif event.key==self.pg.K_c:self.camera_i=(self.camera_i+1)%len(self.CAMERAS)
                    elif self.pg.K_1<=event.key<=self.pg.K_7:self.camera_i=event.key-self.pg.K_1
                    elif event.key==self.pg.K_r:self.saves=self.goals=0;self.last=None
            # Simulate at 50 Hz, render at 60 Hz.
            self.step();self.draw();self.clock.tick(50)
        self.renderer.close();self.brain.close();self.pg.quit()

def main():
    import argparse
    p = argparse.ArgumentParser(description="Playable Arcade Binocular Fly Goalkeeper")
    p.add_argument("--policy", default="arcade_early_intent_vert_refined_rl.npz", help="ExecPolicy npz file in workspace/outputs/arcade_demo")
    p.add_argument("--sense-steps", type=int, default=8, help="Pre-motion sensing decision steps")
    args = p.parse_args()
    ArcadeGame(policy=args.policy, sense_steps=args.sense_steps).run()

if __name__=="__main__":main()
