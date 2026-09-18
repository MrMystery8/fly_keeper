"""Synchronized V3 MID-height and vertical-motor causal audit.

Diagnostics may read simulator truth; the V3 bridge receives only bilateral
retinal input exactly as deployed.  This module neither trains nor changes a
bridge/DN/motor parameter.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path: sys.path.insert(0,str(ROOT/"workspace"))
if str(ROOT/"upstream"/"doomfly") not in sys.path: sys.path.insert(0,str(ROOT/"upstream"/"doomfly"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import ArcadeController,Arcade2AxisDecoder,DECISION_MS,DECISION_S,MAX_DECISIONS
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_v3 import load as load_v3
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
OUT=ROOT/"workspace"/"outputs"/"arcade_demo"

def _tgoal(world):
    b=world._observe_ball(); vx=float(b['vel'][0]); return None if vx>=-1e-6 else float((-0.3-b['pos'][0])/vx)

def _distance(world):
    b=world._observe_ball()['pos']; f=world.fly.position
    q=((float(b[0])-float(f[0]))/world.REACH_RADIUS_X)**2+((float(b[1])-float(f[1]))/world.REACH_RADIUS_Y)**2+((float(b[2])-float(f[2]))/world.REACH_RADIUS_Z)**2
    return float(np.sqrt(q))

def _trace_episode(brain, seed, group, hname, hf):
    w=ArcadeGoalkeeperWorld(seed=seed); basis=ArcadeDNBasis(); bridge,_=load_v3(OUT/'arcade_binocular_bridge_v3.npz',basis)
    ctrl=ArcadeController(brain,BinocularVisionBridge(w.fly,brain,condition='both'),Arcade2AxisDecoder(),bridge)
    shot=w.sample_shot(group,height_frac=hf); w.reset(shot); brain.reset(); ctrl.reset(); rows=[]
    for step in range(MAX_DECISIONS):
        if w.result is not None: break
        (teacher_lat,teacher_vert),intercept=teacher_command(w); before=bridge._ema.copy()
        # Current injected at this step is prior causal bridge state.
        ids,drive=basis.currents(float(before[0])*bridge.lat_gain,float(max(0,before[1]))*bridge.vert_gain)
        command,diag=ctrl.act(w); dn=brain.read(basis.readout_ids()); ball_before=w._observe_ball()
        w.fly.set_command(command['forward'],command.get('turn',0),command['gait_on'],lateral=command['lateral'],vertical=command['vertical']); w.step(DECISION_S)
        vel=w.fly.data.qvel[w.fly.root_dofadr:w.fly.root_dofadr+3]
        rows.append(dict(step=step,time_ms=step*20,time_to_goal_s=_tgoal(w),teacher_lat=float(teacher_lat),teacher_vert=float(teacher_vert),teacher_y_cross=float(intercept['y_cross']),teacher_z_cross=float(intercept['z_cross']),
          v3_u_lat=float(diag['bridge_u_lat']),v3_u_vert=float(diag['bridge_u_vert']),injected_lateral_dn_mv=[float(x) for x in drive[:4]],injected_vertical_dn_mv=[float(x) for x in drive[4:]],
          lateral_dn_spikes=dict(left=float(diag['left_spikes']),right=float(diag['right_spikes'])),vertical_dn_spikes=float(diag['vert_spikes']),decoded_lateral=float(command['lateral']),decoded_vertical=float(command['vertical']),
          fly_y=float(w.fly.position[1]),fly_z=float(w.fly.position[2]),fly_vy=float(vel[1]),fly_vz=float(vel[2]),flight_state=str(w.fly.state),ball_y=float(ball_before['pos'][1]),ball_z=float(ball_before['pos'][2]),interception_distance_radii=_distance(w),keeper_contact=bool(w.keeper_contact),touch_but_goal=bool(getattr(w,'_touch_but_goal',False))))
    return dict(seed=seed,group=group,height=hname,height_frac=hf,shot_launch_ms=0,rows=rows,outcome=w.outcome_diagnostics()|dict(result=w.result,keeper_contact=bool(w.keeper_contact)))

def vertical_sweep(backend='metal',steps=40):
    out=[]
    for u in np.round(np.arange(0,1.01,.1),2):
        b=MaleCNSBrain(backend=backend); w=ArcadeGoalkeeperWorld(seed=130000); w.reset(w.sample_shot('center',height_frac=.5)); dec=Arcade2AxisDecoder(); basis=ArcadeDNBasis(); z0=float(w.fly.position[2]); rise=None; peak=z0; states=[]; cmdvals=[]; dnvals=[]
        for i in range(steps):
            basis.inject(b,0.,float(u)); b.step(DECISION_MS); a=b.read(dec.readout_ids()); cmd,diag=dec.decode(a); w.fly.set_command(0,0,cmd['gait_on'],lateral=0,vertical=cmd['vertical']); w.step(DECISION_S); dz=float(w.fly.position[2])-z0; peak=max(peak,float(w.fly.position[2])); states.append(w.fly.state);cmdvals.append(float(cmd['vertical']));dnvals.append(float(diag['vert_spikes']))
            if rise is None and dz>.05: rise=i*20
        b.close(); out.append(dict(u_vert=float(u),dn_drive_mv=float(u*basis.drive_mv),decoded_vertical_mean=float(np.mean(cmdvals)),vertical_dn_spikes_mean=float(np.mean(dnvals)),takeoff=bool(any(s in ('TAKEOFF','FLIGHT') for s in states)),rise_latency_ms=rise,peak_height=float(peak-z0),final_height=float(w.fly.position[2]-z0),states=states))
    return out

def run(backend='metal'):
    b=MaleCNSBrain(backend=backend); episodes=[]; seed=131000
    for hname,hf in (('low',0.),('mid',.5),('high',.85)):
      for group in ('left','center','right'):
        episodes.append(_trace_episode(b,seed,group,hname,hf)); seed+=1
    b.close(); report=dict(schema_version=1,diagnostic_only=True,backend=backend,episodes=episodes,vertical_command_sweep=vertical_sweep(backend))
    (OUT/'arcade_binocular_bottleneck_audit.json').write_text(json.dumps(report,indent=2)); return report

if __name__=='__main__':
 r=run(); print(json.dumps(dict(mid=[dict(group=e['group'],result=e['outcome']['result'],n=len(e['rows'])) for e in r['episodes'] if e['height']=='mid'],sweep=[{k:v for k,v in x.items() if k!='states'} for x in r['vertical_command_sweep']]),indent=2))
