"""Matched both/left-only/right-only offline fusion diagnostic (analysis only)."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'workspace') not in sys.path:sys.path.insert(0,str(ROOT/'workspace'))
from experiments.arcade_demo.binocular_temporal_foundation import _design,_fit_task,stratified_episode_split,WINDOWS,BUDGETS
OUT=ROOT/'workspace'/'outputs'/'arcade_demo'
def one(name):
 d=np.load(OUT/name,allow_pickle=False);raw=d['features'].astype(float);pool=len(d['pool_body_ids']);g=d['groups'].astype(str);h=d['height_classes'].astype(str);s=d['seeds'].astype(int);y=d['u_lat'].astype(float);mw=int(np.asarray(d['n_windows']).ravel()[0]);parts,_=stratified_episode_split(s,g,h);m={k:np.isin(s,list(v)) for k,v in parts.items()};cand=[]
 for w in WINDOWS:
  if w>mw:continue
  X=_design(raw,w,pool)
  for k in BUDGETS:
   f=_fit_task(X,y,m['train'],m['val'],m['test'],pool,'lateral',k);cand.append((f['validation']['corr'],w,k,f))
 best=max(cand,key=lambda x:x[0]);return dict(dataset=name,window_bins=best[1],budget=best[2],validation=best[3]['validation'],test=best[3]['test'])
if __name__=='__main__':
 out={'both':one('binocular_balanced_dataset.npz'),
      'left_only':one('binocular_balanced_left_only.npz'),
      'right_only':one('binocular_balanced_right_only.npz')}
 (OUT/'binocular_balanced_fusion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
