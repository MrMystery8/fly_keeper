"""Polished, responsive display layer for the frozen binocular goalkeeper.

The sports camera and all HUD state are display-only.  The controller continues
to consume only BinocularVisionBridge's two physical sensor-camera paths.
"""
from __future__ import annotations
import ast, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
for p in (ROOT/"workspace",ROOT/"upstream"/"doomfly"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from adapters.brain import MaleCNSBrain
from embodiment.mujoco_world import ShotSpec,GOAL_HALF_WIDTH,GOAL_HEIGHT
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld, LATERAL_AIM, height_for_frac
from experiments.arcade_demo.arcade_runtime import ArcadeController,Arcade2AxisDecoder,DECISION_MS,DECISION_S,MAX_DECISIONS
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_visual
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import EarlyIntentBridge,ExecPolicy

# Display-only sports camera controls.
GAME_CAM_DISTANCE=6.4; GAME_CAM_HEIGHT=1.35; GAME_CAM_SIDE_OFFSET=1.15
GAME_CAM_LOOK_AT_HEIGHT=.42; GAME_CAM_FOV=42
SPEED_MIN,SPEED_MAX=4.5,6.0  # Exact frozen training/evaluation support.

@dataclass
class ShotControlState:
    aim_y:float=0.; height_frac:float=.5; sweep:float=0.; stage:str="direction"
    @property
    def aim_z(self): return height_for_frac(self.height_frac)
    @property
    def power(self): return .35+.65*self.height_frac
    @property
    def speed(self): return SPEED_MIN+(self.power-.35)/.65*(SPEED_MAX-SPEED_MIN)

class ArcadeGame:
    CAMERAS=(("presentation","SPORTS CAMERA"),("track1","TRACK 1"),("track2","TRACK 2"),("track3","HIGH SIDE"),("side","SIDE"),("hero","HERO"),("back","REVERSE"))
    BG=(5,10,20); PANEL=(13,27,43); BORDER=(38,72,96); CYAN=(96,224,245); VIOLET=(173,126,255); ORANGE=(255,174,83); AMBER=(255,202,91); GREEN=(99,239,159); RED=(255,102,112); TEXT=(231,240,247); MUTED=(134,161,185)
    def __init__(self,policy="arcade_early_intent_vert_refined_rl.npz",sense_steps=8):
        import pygame,mujoco
        self.pg,self.mujoco=pygame,mujoco; pygame.init()
        self.screen=pygame.display.set_mode((1280,800),pygame.RESIZABLE); pygame.display.set_caption("Fruit Fly Goalkeeper — MaleCNS Early-Intent Controller")
        self.clock=pygame.time.Clock(); self.policy_file=policy; self.sense_steps=int(sense_steps); self.seed=400000; self.camera_i=0; self.show_sports_preview=False; self.research=False; self.fullscreen=False
        self.active=False; self.aftermath=0.; self.decisions=0; self.last=None; self.result_age=9.; self.saves=self.goals=self.misses=0; self.last_command={}; self.last_diag={}; self.eye_info={}; self.last_luminance=None; self.shots=0; self.command_history=[]; self.dn_history=[]; self.last_shot={}; self.sim_seconds=0.; self.wall_seconds=0.; self.realtime_x=0.
        self.shot=ShotControlState(); self.display_aim=0.; self.display_height=.5; self._last_mouse=None
        self.map_ids=[]; self.map_xy=np.empty((0,2)); self.map_kind=[]; self.map_spikes=np.empty(0,int)
        self.brain=MaleCNSBrain(backend="metal"); self.world=ArcadeGoalkeeperWorld(seed=self.seed)
        # The raw MuJoCo model leaves the ball at an origin pose, directly in
        # front of the V2 camera.  Seed the *idle display only* with a normal
        # supported shot pose so the opening screen is a readable pitch/goal.
        # `fire()` always replaces this world and resets a fresh episode.
        self.world.reset(self.world.sample_shot("center",height_frac=.5))
        self.world.model.geom_rgba[self.world._ball_gid,3]=0.0
        self.renderer=mujoco.Renderer(self.world.model,height=540,width=900)
        self.presentation_camera=self._make_game_camera(); self.sports_preview_camera=self._make_sports_preview_camera(); self._build_controller(); self._fonts()
    def _fonts(self):
        self.font=self.pg.font.SysFont("Arial",17); self.small=self.pg.font.SysFont("Arial",13); self.tiny=self.pg.font.SysFont("Consolas",12); self.big=self.pg.font.SysFont("Arial",42,bold=True); self.title=self.pg.font.SysFont("Arial",20,bold=True)
    def _make_game_camera(self):
        # Keep the familiar V2 framing, but lower the display viewpoint to a
        # more natural striker/eye-level angle.  This camera never enters the
        # visual input passed to the frozen controller.
        cam=self.mujoco.MjvCamera(); cam.type=self.mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:]=(.9,0.,.35); cam.distance=5.2; cam.azimuth=180.; cam.elevation=-12.; return cam
    def _make_sports_preview_camera(self):
        # Display-only rear goalkeeper view: camera sits behind the fly, looking
        # through it toward the goal and approaching ball path.
        cam=self.mujoco.MjvCamera(); cam.type=self.mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:]=(.35,0.,GAME_CAM_LOOK_AT_HEIGHT); cam.distance=7.0; cam.azimuth=0.; cam.elevation=-13.; return cam
    def _build_controller(self):
        visual,_=load_visual("arcade_binocular_lateral_hybrid.npz",ArcadeDNBasis()); est,_=EarlyIntentEstimator.load(); policy,_=ExecPolicy.load(ROOT/"workspace/outputs/arcade_demo"/self.policy_file)
        self.bridge=EarlyIntentBridge(visual,est,policy=policy,sense_steps=self.sense_steps); self.vision=BinocularVisionBridge(self.world.fly,self.brain,condition="both",retina_map="fullframe"); self.decoder=Arcade2AxisDecoder(); self.controller=ArcadeController(self.brain,self.vision,self.decoder,self.bridge); self._build_brain_map()
    def _build_brain_map(self):
        import pandas as pd
        dn=list(self.decoder.readout_ids()); ids=list(dict.fromkeys(list(self.bridge.body_ids)+dn)); ann=pd.read_feather(ROOT/"upstream/doomfly/connectome_data/malecns_v1/annotations.feather",columns=["bodyId","somaLocation"]).set_index("bodyId"); kept=[]; xy=[]; kind=[]
        for bid in ids:
            loc=ann.loc[bid,"somaLocation"] if bid in ann.index else None
            if isinstance(loc,str):
                try: loc=ast.literal_eval(loc)
                except (SyntaxError,ValueError): loc=None
            if loc is not None and hasattr(loc,"__len__") and len(loc)>=3 and np.isfinite(np.asarray(loc,float)).all(): kept.append(bid); xy.append((loc[0],loc[2])); kind.append("DN" if bid in dn else "VIS")
        self.map_ids=kept; self.map_xy=np.asarray(xy,float); self.map_kind=kind; self.map_spikes=np.zeros(len(kept),int)
    def fire(self):
        if self.active:return
        self.seed+=1; self.renderer.close(); self.vision._renderer.close(); self.world=ArcadeGoalkeeperWorld(seed=self.seed); self.renderer=self.mujoco.Renderer(self.world.model,height=540,width=900); self._build_controller()
        # Both axes and speed remain inside the original champion's supported
        # physical envelope.  This state is UI-only; it is never controller input.
        group="left" if self.shot.aim_y>.18 else ("right" if self.shot.aim_y<-.18 else "center")
        target=ShotSpec(group,float(self.shot.aim_y),float(self.shot.speed),float(self.shot.aim_z)); self.last_shot={"aim":self.shot.aim_y,"height":self.shot.height_frac,"speed":self.shot.speed}
        self.world.reset(target); self.brain.reset(); self.controller.reset(); self.active=True; self.aftermath=0.; self.decisions=0; self.last=None; self.last_command={}; self.last_diag={}; self.result_age=0.; self.shot.stage="direction"; self.sim_seconds=0.; self.wall_seconds=0.; self.realtime_x=0.

    def _time_ratio(self,wall_dt):
        """Track simulated seconds per wall-clock second for this round."""
        self.sim_seconds+=DECISION_S; self.wall_seconds+=max(float(wall_dt),1e-6)
        self.realtime_x=self.sim_seconds/self.wall_seconds
    def _reset_round(self):
        """Create the next clean READY round while preserving match scores."""
        self.seed+=1; self.renderer.close(); self.vision._renderer.close(); self.world=ArcadeGoalkeeperWorld(seed=self.seed)
        self.world.reset(self.world.sample_shot("center",height_frac=.5)); self.world.model.geom_rgba[self.world._ball_gid,3]=0.0
        self.renderer=self.mujoco.Renderer(self.world.model,height=540,width=900); self._build_controller(); self.brain.reset(); self.controller.reset()
        self.active=False; self.aftermath=0.; self.decisions=0; self.last=None; self.result_age=9.; self.last_command={}; self.last_diag={}; self.eye_info={}; self.last_luminance=None; self.command_history=[]; self.dn_history=[]; self.last_shot={}; self.shot=ShotControlState(); self.display_aim=0.; self.display_height=.5; self.sim_seconds=0.; self.wall_seconds=0.; self.realtime_x=0.
    def reset_game(self):
        """Reset the scoreboard and restore a clean READY round."""
        self.saves=self.goals=self.misses=self.shots=0; self._reset_round()
    def step(self, wall_dt=DECISION_S):
        if not self.active:return
        if self.aftermath>0.:
            # Let MuJoCo resolve the real deflection and fly momentum after a
            # SAVE; the neural episode has already reached its terminal state.
            self.world.fly.set_command(0.,0.,0.,lateral=0.,vertical=0.)
            self.world.step(DECISION_S); self._time_ratio(wall_dt); self.aftermath-=wall_dt
            if self.aftermath<=0.: self._reset_round()
            return
        self.last_luminance,self.eye_info=self.vision.perceive(); self.bridge.inject(self.brain); info=self.brain.step(DECISION_MS); self.bridge.observe(self.brain); act=self.brain.read(self.decoder.readout_ids()); _,diag=self.decoder.decode(act); sensing=self.bridge.sensing; ul,uv=self.bridge.command(self.world); gait=self.bridge.gait_on()
        self.last_command={"lateral":ul,"vertical":uv,"gait_on":gait}; diag.update(intent_lat=self.bridge.intent.intent_lat,confidence=self.bridge.intent.confidence,sensing=sensing,spikes=info.get("spikes",0)); self.last_diag=diag
        self.command_history=(self.command_history+[ul])[-42:]; self.dn_history=(self.dn_history+[(diag["left_spikes"],diag["right_spikes"])])[-42:]
        if self.map_ids: r=self.brain.read(self.map_ids); self.map_spikes=np.asarray([r[i]["spikes"] for i in self.map_ids],int)
        self.world.fly.set_command(0.,0.,gait,lateral=ul,vertical=uv); self.world.step(DECISION_S); self._time_ratio(wall_dt); self.decisions+=1
        if self.world.result is not None or self.decisions>=MAX_DECISIONS:
            self.last=self.world.result or "GOAL"
            self.shots+=1
            if self.last=="SAVE":self.saves+=1
            elif self.last=="GOAL":self.goals+=1
            else:self.misses+=1
            self.aftermath=3.0
    def _layout(self):
        w,h=self.screen.get_size(); top=max(58,int(h*.075)); foot=max(48,int(h*.07)); split=w-max(300,int(w*(.37 if self.research else .28))); return w,h,top,foot,split
    def _text(self,t,x,y,font,color):self.screen.blit(font.render(str(t),True,color),(int(x),int(y)))
    def _fit(self,t,font,max_w):
        text=str(t)
        if font.size(text)[0]<=max_w:return text
        while text and font.size(text+"…")[0]>max_w:text=text[:-1]
        return text+"…"
    def _text_fit(self,t,x,y,font,color,max_w):self._text(self._fit(t,font,max_w),x,y,font,color)
    def _right_text(self,t,right,y,font,color):
        image=font.render(str(t),True,color); self.screen.blit(image,(int(right-image.get_width()),int(y)))
    def _blit_contained(self,surface,rect):
        """Blit without changing the renderer's aspect ratio (no egg-ball)."""
        x,y,w,h=rect; sw,sh=surface.get_size(); scale=min(w/sw,h/sh)
        tw,th=max(1,int(sw*scale)),max(1,int(sh*scale))
        image=self.pg.transform.smoothscale(surface,(tw,th))
        ix,iy=x+(w-tw)//2,y+(h-th)//2
        self.screen.blit(image,(ix,iy))
        return ix,iy,tw,th
    def _panel(self,r,title="",accent=None):
        self.pg.draw.rect(self.screen,self.PANEL,r,border_radius=10); self.pg.draw.rect(self.screen,accent or self.BORDER,r,1,border_radius=10)
        if title:self._text_fit(title,r[0]+11,r[1]+8,self.small,accent or self.TEXT,r[2]-22)
    def draw(self):
        w,h,top,foot,split=self._layout(); self.screen.fill(self.BG); self.pg.draw.rect(self.screen,(8,18,31),(0,0,w,top)); self.pg.draw.line(self.screen,self.BORDER,(0,top-1),(w,top-1))
        phase=("SAVE • PLAYOUT" if self.aftermath>0. and self.last=="SAVE" else ("SENSING" if self.active and self.last_diag.get("sensing") else ("EXECUTING" if self.active else self.last or "READY"))); color={"SENSING":self.CYAN,"EXECUTING":(80,158,255),"SAVE":self.GREEN,"SAVE • PLAYOUT":self.GREEN,"GOAL":self.RED,"MISS":self.MUTED,"READY":self.AMBER}.get(phase,self.TEXT)
        rate=100*self.saves/max(1,self.shots); self._text("FRUIT FLY GOALKEEPER",18,10,self.title,self.TEXT); self._text(f"MATCH  {self.shots:02d} SHOTS  •  {rate:.0f}% SAVE RATE",19,35,self.small,self.MUTED); self._text(f"SAVES  {self.saves:02d}",w//2-180,20,self.title,self.GREEN); self._text(f"● {phase}"+(f"  {self.decisions}/{MAX_DECISIONS}" if self.active else ""),w//2-35,20,self.title,color); self._text(f"SIM  {self.realtime_x:.2f}× REAL TIME" if self.active else "SIM  —",w-305,35,self.small,self.CYAN if self.realtime_x>=.95 else self.AMBER); self._text(f"GOALS  {self.goals:02d}",w-150,20,self.title,self.RED)
        self._game_view(16,top+10,split-28,h-top-foot-20); self._science(split+4,top+10,w-split-20,h-top-foot-20); self._footer(w,h-foot,foot)
        if self.last and self.result_age<1.05:
            s=self.big.render(self.last+"!",True,{"SAVE":self.GREEN,"GOAL":self.RED,"MISS":self.MUTED}.get(self.last,self.TEXT)); self.screen.blit(s,(split//2-s.get_width()//2,top+38))
            if self.last_shot:self._text(f"aim {self.last_shot.get('aim',0):+.2f}  •  height {self.last_shot.get('height',0):.0%}  •  {self.last_shot.get('speed',0):.2f} cm/s",split//2-125,top+88,self.tiny,self.TEXT)
        self.pg.display.flip()
    def _game_view(self,x,y,w,h):
        self._panel((x,y,w,h),"",self.BORDER); cam,label=self.CAMERAS[self.camera_i]; self.renderer.update_scene(self.world.data,camera=self.presentation_camera if cam=="presentation" else cam); im=self.renderer.render(); sf=self.pg.surfarray.make_surface(np.transpose(im,(1,0,2))); image_rect=self._blit_contained(sf,(x+6,y+6,w-12,h-12)); self._text("GAME VIEW  •  "+label,x+16,y+16,self.small,self.CYAN)
        if self.show_sports_preview:
            pw,ph=max(160,int(w*.25)),max(90,int(h*.22)); px,py=x+w-pw-18,y+36
            self.renderer.update_scene(self.world.data,camera=self.sports_preview_camera); preview=self.renderer.render(); ps=self.pg.surfarray.make_surface(np.transpose(preview,(1,0,2)))
            self.pg.draw.rect(self.screen,(4,10,18),(px-3,py-20,pw+6,ph+23),border_radius=6); self._blit_contained(ps,(px,py,pw,ph)); self.pg.draw.rect(self.screen,self.ORANGE,(px-3,py-20,pw+6,ph+23),1,border_radius=6); self._text("BEHIND FLY  [C] HIDE",px+6,py-17,self.tiny,self.ORANGE)
        if not self.active:
            self._draw_shot_arrow(*image_rect)
        if self.research:self._text(f"aim y={self.shot.aim_y:+.2f} z={self.shot.aim_z:.2f} physical speed={self.shot.speed:.2f}",x+15,y+h-29,self.tiny,self.TEXT)
    def _draw_shot_arrow(self,x,y,w,h):
        """One camera-anchored 3D arrow: rotating it never tears it apart."""
        # Exact camera-space inner face of the goal opening.  `x,y,w,h` are
        # the rendered image bounds (not its surrounding letterbox panel), so
        # 0% and 100% height line up with the visible grass and crossbar.
        gx,gy,gw,gh=int(x+w*.349),int(y+h*.300),int(w*.310),int(h*.227)
        start=np.array((x+w*.50,y+h*.79),float)
        target=np.array((gx+gw*(self.display_aim/LATERAL_AIM+1)/2,gy+gh*(1-(.50 if self.shot.stage=="direction" else self.display_height))),float)
        direction=(target-start)/max(float(np.linalg.norm(target-start)),1.)
        # The arrow follows the ray to the exact displayed shot point, but
        # stops well in front of the fly and goal.
        tip=start+(target-start)*.31; head_base=tip-direction*10.; normal=np.array((-direction[1],direction[0]))
        # Colour encodes the same height/power value, but the selector itself
        # is a single solid object rather than four independently moving bars.
        low,high=np.array((221,131,48)),np.array((74,190,120)); mix=(self.shot.power-.35)/.65
        color=tuple(((1-mix)*low+mix*high).astype(int)); shade=tuple(max(0,c-78) for c in color)
        # A small projected aim point states exactly what the arrow represents.
        # It is a target marker, not an interaction crosshair.
        self.pg.draw.circle(self.screen,(20,37,42),tuple(target.astype(int)),9)
        self.pg.draw.circle(self.screen,color,tuple(target.astype(int)),7,1)
        self.pg.draw.circle(self.screen,(238,244,219),tuple(target.astype(int)),2)
        body=[start+normal*6.,head_base+normal*3.5,head_base-normal*3.5,start-normal*6.]
        head=[tip,head_base+normal*7.5,head_base-normal*7.5]; depth=np.array((1.5,2.5))
        # The shadow is deliberately just a thin reference line: it gives the
        # raised arrow depth without visually detaching it from its own base.
        self.pg.draw.line(self.screen,(18,34,38),tuple((start+depth).astype(int)),tuple((tip+depth).astype(int)),4)
        self.pg.draw.polygon(self.screen,shade,[tuple(p.astype(int)) for p in (body[2],body[3],body[3]+depth,body[2]+depth)])
        self.pg.draw.polygon(self.screen,shade,[tuple(p.astype(int)) for p in (head[1],head[2],head[2]+depth,head[1]+depth)])
        self.pg.draw.polygon(self.screen,color,[tuple(p.astype(int)) for p in body])
        self.pg.draw.polygon(self.screen,color,[tuple(p.astype(int)) for p in head])
        # A narrow highlight gives the single-piece arrow a clean game skin,
        # without a separate tail ornament or distracting animation.
        self.pg.draw.line(self.screen,(244,247,221),tuple((start+normal*2.).astype(int)),tuple((head_base+normal*.8).astype(int)),1)
        label="LOCK DIRECTION" if self.shot.stage=="direction" else f"LOCK HEIGHT  •  POWER {self.shot.power:.0%}"
        self._text(label,int(start[0]-54),int(start[1]+18),self.tiny,self.AMBER if self.shot.stage=="direction" else self.ORANGE)
    def _science(self,x,y,w,h):
        gap=8; eh=max(104,int(h*.20)); ih=max(115,int(h*.20)); self._eye((x,y,w,eh),"LEFT EYE",getattr(self.vision,"last_left_frame",None),self.eye_info.get("left_lum_mean",0)); self._eye((x,y+eh+gap,w,eh),"RIGHT EYE",getattr(self.vision,"last_right_frame",None),self.eye_info.get("right_lum_mean",0)); self._intent((x,y+(eh+gap)*2,w,ih)); self._brain((x,y+(eh+gap)*2+ih+gap,w,h-((eh+gap)*2+ih+gap)))
    def _eye(self,r,label,frame,lum):
        self._panel(r,label,self.CYAN); x,y,w,h=r
        if frame is not None:
            sf=self.pg.surfarray.make_surface(np.transpose(frame,(1,0,2))); sf=self.pg.transform.smoothscale(sf,(int(min(w*.62,(h-34)*1.66)),h-34)); self.screen.blit(sf,(x+9,y+25))
        tx=x+int(w*.66); self._text_fit("● ACTIVE",tx,y+28,self.tiny,self.CYAN,w-(tx-x)-8); self._text_fit(f"{int(lum*1000)} retinal",tx,y+48,self.tiny,self.MUTED,w-(tx-x)-8); self._text_fit("spike estimate",tx,y+63,self.tiny,self.MUTED,w-(tx-x)-8)
    def _bar(self,x,y,w,v,c,label,value):
        self._text_fit(label,x,y,self.tiny,self.MUTED,w); self.pg.draw.rect(self.screen,(28,45,64),(x,y+16,w,8),border_radius=4); self.pg.draw.rect(self.screen,c,(x,y+16,int(w*np.clip(v,0,1)),8),border_radius=4)
    def _intent(self,r):
        self._panel(r,"INTENT  /  MOTOR",self.VIOLET); x,y,w,h=r; it=float(self.last_diag.get("intent_lat",0)); cf=float(self.last_diag.get("confidence",0)); ul=float(self.last_command.get("lateral",0)); uv=float(self.last_command.get("vertical",0)); mid=x+w*.27
        self._text("DIRECTIONAL INTENT",x+12,y+30,self.tiny,self.VIOLET); self.pg.draw.line(self.screen,self.BORDER,(mid-w*.18,y+48),(mid+w*.18,y+48),2); self.pg.draw.circle(self.screen,self.VIOLET,(int(mid+it*w*.18),y+48),5); self._text("LEFT",x+12,y+56,self.tiny,self.MUTED); self._text("RIGHT",x+w-48,y+56,self.tiny,self.MUTED); self._bar(x+12,y+76,int(w*.42),cf,self.VIOLET,"CONFIDENCE",f"{cf:.2f}"); self._bar(x+int(w*.54),y+30,int(w*.30),abs(ul),self.ORANGE,"LATERAL",f"{ul:+.2f}"); self._bar(x+int(w*.54),y+76,int(w*.30),uv,self.ORANGE,"VERTICAL",f"{uv:+.2f}")
    def _brain(self,r):
        self._panel(r,"MaleCNS  •  ACTIVITY",self.CYAN); x,y,w,h=r
        active=int((self.map_spikes>0).sum()); total=len(self.map_ids); dn_count=len(self.decoder.readout_ids()); visual_count=len(self.bridge.body_ids)
        self._text_fit(f"FIRING {active}/{total}  •  GLOBAL {int(self.last_diag.get('spikes',0))}",x+12,y+29,self.tiny,self.TEXT,w-24)
        self._text_fit(f"RETINA L {self.eye_info.get('left_lum_mean',0):.3f}  R {self.eye_info.get('right_lum_mean',0):.3f}  •  PATH {visual_count}→{dn_count} DNs",x+12,y+44,self.tiny,self.MUTED,w-24)
        self._retina_strip(x+12,y+58,w-24,10)
        map_top=y+76
        if len(self.map_xy) and h>100:
            lo=self.map_xy.min(0); span=np.maximum(self.map_xy.max(0)-lo,1); pts=(self.map_xy-lo)/span
            for (px,py),sp,k in zip(pts,self.map_spikes,self.map_kind):
                col=self.ORANGE if k=="DN" else self.CYAN; self.pg.draw.circle(self.screen,col if sp else ((92,59,40) if k=="DN" else (35,70,90)),(int(x+12+px*(w-24)),int(y+h-12-py*(h-(map_top-y+12)))),min(6,2+int(sp)) if sp else 2)
        if self.research:
            move=str(getattr(self.world.fly,"state","grounded")).upper(); asym=float(self.last_diag.get("asymmetry",0)); self._text_fit(f"DN L {self.last_diag.get('left_spikes',0):.0f}  R {self.last_diag.get('right_spikes',0):.0f}  asym {asym:+.2f}  •  {move}  •  d {self.decisions}/{MAX_DECISIONS}",x+12,y+h-28,self.tiny,self.TEXT,w-24); self._sparkline(self.command_history,x+12,y+h-16,w-24,8)
    def _retina_strip(self,x,y,w,h):
        """Binned actual receptor luminance, not a reconstructed ball cue."""
        if self.last_luminance is None:return
        values=np.asarray(self.last_luminance,float); bins=24
        for side,mask,color in (("L",self.vision.left_mask,self.CYAN),("R",self.vision.right_mask,self.VIOLET)):
            vals=values[mask]; chunks=np.array_split(vals,bins); base=x+(0 if side=="L" else w//2); bw=max(1,(w//2-8)//bins)
            self._text(side,base,y-2,self.tiny,color)
            for i,chunk in enumerate(chunks):
                level=float(np.mean(chunk)) if len(chunk) else 0.; self.pg.draw.rect(self.screen,(25,43,60),(base+10+i*bw,y,bw-1,h)); self.pg.draw.rect(self.screen,color,(base+10+i*bw,y+int(h*(1-np.clip(level,0,1))),bw-1,max(1,int(h*np.clip(level,0,1)))))
    def _sparkline(self,values,x,y,w,h):
        if len(values)<2:return
        pts=[]
        for i,value in enumerate(values):pts.append((x+int(i*w/(len(values)-1)),y+int(h/2-np.clip(value,-1,1)*(h/2-1))))
        self.pg.draw.lines(self.screen,self.ORANGE,False,pts,1)
    def _footer(self,w,y,h):
        self.pg.draw.rect(self.screen,(8,18,31),(0,y,w,h)); self._text("[RESEARCH MODE]" if self.research else "[GAME MODE]",16,y+15,self.small,self.VIOLET); action="[SPACE] LOCK DIRECTION" if self.shot.stage=="direction" else "[SPACE] LOCK HEIGHT / FIRE"; self._text(f"{action}   [C] BEHIND-FLY VIEW   [1–7] CAMERA   [TAB] RESEARCH   [R] RESET",160,y+16,self.small,self.TEXT); self._text("POWER",w-190,y+9,self.tiny,self.AMBER); self.pg.draw.rect(self.screen,(32,47,63),(w-190,y+25,150,10),border_radius=5); self.pg.draw.rect(self.screen,self.AMBER,(w-190,y+25,int(150*self.shot.power),10),border_radius=5); self._text(f"{self.shot.power:.0%}",w-36,y+22,self.tiny,self.TEXT)
    def _aim(self,dt):
        if self.active:return
        self.shot.sweep=(self.shot.sweep+dt*.42)%2.0
        # Cosine sweep has zero velocity at each reversal.  The old triangle
        # wave changed direction instantly, which made the aiming animation
        # feel like a disconnected object snapping sideways.
        value=(1.-np.cos(np.pi*self.shot.sweep))/2.
        if self.shot.stage=="direction": self.shot.aim_y=float((2*value-1)*LATERAL_AIM)
        else: self.shot.height_frac=float(value)
        settle=1.-np.exp(-dt*14.)
        self.display_aim+=(self.shot.aim_y-self.display_aim)*settle
        self.display_height+=(self.shot.height_frac-self.display_height)*settle
    def run(self):
        running=True
        while running:
            dt=self.clock.tick(50)/1000.; self.result_age+=dt
            for e in self.pg.event.get():
                if e.type==self.pg.QUIT:running=False
                elif e.type==self.pg.VIDEORESIZE:self.screen=self.pg.display.set_mode(e.size,self.pg.RESIZABLE)
                elif e.type==self.pg.KEYDOWN:
                    if e.key in (self.pg.K_ESCAPE,self.pg.K_q):running=False
                    elif e.key==self.pg.K_SPACE and not self.active:
                        if self.shot.stage=="direction": self.shot.stage="height"; self.shot.sweep=0.
                        else:self.fire()
                    elif e.key==self.pg.K_c:self.show_sports_preview=not self.show_sports_preview
                    elif self.pg.K_1<=e.key<=self.pg.K_7:self.camera_i=e.key-self.pg.K_1
                    elif e.key==self.pg.K_TAB:self.research=not self.research
                    elif e.key==self.pg.K_r:self.reset_game()
            self._aim(dt);self.step(dt);self.draw()
        self.renderer.close();self.vision._renderer.close();self.brain.close();self.pg.quit()
def main():
    import argparse
    p=argparse.ArgumentParser(description="Playable Fruit Fly Goalkeeper");p.add_argument("--policy",default="arcade_early_intent_vert_refined_rl.npz");p.add_argument("--sense-steps",type=int,default=8);a=p.parse_args();ArcadeGame(policy=a.policy,sense_steps=a.sense_steps).run()
if __name__=="__main__":main()
