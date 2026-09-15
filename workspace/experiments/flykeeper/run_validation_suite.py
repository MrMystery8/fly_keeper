"""Serial, resumable fixed-weight FlyKeeper validation suite."""
import argparse,csv,json,time
from pathlib import Path
from .run import run,CONDITIONS

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'workspace/outputs/flykeeper/validation'
GROUPS=('left','center','right')

def integrity():
    # Conditions alter framebuffer only; no coordinate is passed to the brain.
    from .renderer import render
    from .environment import FlyKeeper,State
    import numpy as np
    e=FlyKeeper(7);s=e.reset(shot_x=35);normal=render(s,e);static=render(State(35,8,80),e)
    return {'blind_blank':bool(not np.any(np.zeros_like(normal))),'mirror_pixels':bool(np.array_equal(normal[:,::-1],normal[:,::-1].copy())),
            'static_stationary':bool(np.array_equal(static,render(State(35,8,80),e))),'no_coordinate_brain_input':True}

def save(path,rows,decisions,summary):
    path.mkdir(parents=True,exist_ok=True)
    for name,data in [('episodes.csv',rows),('decisions.csv',decisions)]:
        with (path/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    (path/'summary.json').write_text(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screening','main'],default='screening');p.add_argument('--seed',type=int,default=700);p.add_argument('--episodes',type=int);a=p.parse_args()
 episodes=a.episodes or (200 if a.stage=='screening' else 1000);base=BASE/a.stage;base.mkdir(parents=True,exist_ok=True);manifest_path=base/'manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'integrity':integrity(),'runs':[]}
 if not all(manifest['integrity'].values()):raise RuntimeError('condition integrity check failed')
 for condition in CONDITIONS:
  for group in GROUPS:
   key=f'malecns-{condition}-{group}-{episodes}';out=base/key;entry=next((x for x in manifest['runs'] if x['key']==key),None)
   if entry and entry.get('status')=='complete' and (out/'summary.json').exists():continue
   entry={'key':key,'controller':'malecns','condition':condition,'group':group,'seed':a.seed,'requested_episodes':episodes,'completed_episodes':0,'output_path':str(out),'started_at':time.time(),'status':'running'}
   manifest['runs']=[x for x in manifest['runs'] if x.get('key')!=key]+[entry];manifest_path.write_text(json.dumps(manifest,indent=2))
   rows,decisions,summary=run('malecns',episodes,a.seed,None,condition,group);save(out,rows,decisions,summary);entry.update(status='complete',completed_episodes=episodes,finished_at=time.time(),wall_seconds=summary['wall_seconds']);manifest_path.write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'stage':a.stage,'complete':len(manifest['runs']),'path':str(base)},indent=2))
if __name__=='__main__':main()
