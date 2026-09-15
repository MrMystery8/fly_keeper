import argparse,csv,json,time,uuid,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'workspace'))
from adapters.brain import MaleCNSBrain
from doom.game import retinal_samples
from experiments.flykeeper.environment import FlyKeeper,State
from experiments.flykeeper.renderer import render
from experiments.flykeeper.controller import MaleCNSGoalkeeper,RandomGoalkeeper
from telemetry import TelemetryServer
CONDITIONS=('normal','blind','mirrored','shuffled','static_ball')
def _x(group):return {'left':35,'center':80,'right':125}.get(group)
def _image(kind,img,static):return np.zeros_like(img) if kind=='blind' else img[:,::-1].copy() if kind=='mirrored' else static if kind=='static_ball' else img
def run(controller_name,episodes,seed,telemetry=None,condition='normal',group='all',backend='cpu'):
 env=FlyKeeper(seed);brain=MaleCNSBrain(backend=backend);ids=[int(brain._brain.ids[i]) for i in brain._brain.retina];ctrl=MaleCNSGoalkeeper() if controller_name=='malecns' else RandomGoalkeeper(seed);rows=[];decisions=[];start=time.perf_counter()
 for ep in range(episodes):
  s=env.reset(seed+ep,_x(group));static=render(State(s.ball_x,8,80),env);frames=[]
  while not s.done:frames.append(render(s,env));s=env.step('STAY')
  s=env.reset(seed+ep,_x(group));actions=[];previous='STAY';t0=brain._brain.sim_ms
  while not s.done:
   step=len(actions);raw=frames[-1-step] if condition=='shuffled' else render(s,env);light=retinal_samples(_image(condition,raw,static),brain._brain.uv);brain.stimulate_retinal_luminance(ids,light);brain.step(20.);read=brain.read(MaleCNSGoalkeeper.LEFT+MaleCNSGoalkeeper.RIGHT);a,scores=ctrl.decode(read);s=env.step(a);actions.append(a)
   decisions.append({'episode':ep,'step':step,'brain_time_ms':brain._brain.sim_ms,'ball_visual_x':s.ball_x,'left_score':scores['left'],'right_score':scores['right'],'score_difference':scores['right']-scores['left'],'action':a,'previous_action':previous,'condition':condition,'retinal_mean':float(light.mean())});previous=a
   if telemetry:telemetry.publish({'brain_time_ms':brain._brain.sim_ms,'episode':ep,'game':{'ball_x':s.ball_x,'ball_y':s.ball_y,'keeper_x':s.keeper_x,'action':a,'result':s.result},'readout':scores,'brain':{'active_ids':[str(i) for i in np.flatnonzero(brain._brain.counts)[:512]],'readout_ids':['10162','10059'],'label':'simulator activity'}})
  rows.append({'episode':ep,'group':group,'result':s.result,'actions':' '.join(actions),'final_keeper_x':s.keeper_x,'brain_ms':brain._brain.sim_ms-t0})
 saves=sum(r['result']=='SAVE' for r in rows);return rows,decisions,{'controller':controller_name,'backend':backend,'condition':condition,'group':group,'episodes':episodes,'saves':saves,'goals':episodes-saves,'save_rate':saves/episodes,'wall_seconds':time.perf_counter()-start,'brain_ms':brain._brain.sim_ms}
def main():
 p=argparse.ArgumentParser();p.add_argument('--headless',action='store_true');p.add_argument('--visualizer',action='store_true');p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=7);p.add_argument('--controller',choices=['malecns','random'],default='malecns');p.add_argument('--backend',choices=['cpu','metal'],default='cpu');p.add_argument('--condition',choices=CONDITIONS,default='normal');p.add_argument('--group',choices=['all','left','center','right'],default='all');a=p.parse_args();t=TelemetryServer() if a.visualizer else None
 if t:t.start()
 rows,decisions,summary=run(a.controller,a.episodes,a.seed,t,a.condition,a.group,a.backend);out=ROOT/'workspace/outputs/flykeeper'/uuid.uuid4().hex;out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));config=(ROOT/'workspace/experiments/flykeeper/config.yaml').read_text().replace('backend: cpu',f'backend: {a.backend}');(out/'config.yaml').write_text(config);(out/'run.log').write_text(json.dumps(summary)+'\n')
 for name,data in [('episodes.csv',rows),('decisions.csv',decisions)]:
  with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 print(json.dumps({'output':str(out),**summary},indent=2))
if __name__=='__main__':main()
