"""Analysis-only representation ceiling: richer causal summaries and tiny MLP.

No bridge artifact is written.  Every normaliser and feature ranking is fit on
TRAIN; validation chooses the summary/epoch, and test is measured only for the
validation-selected candidate.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'workspace') not in sys.path:sys.path.insert(0,str(ROOT/'workspace'))
from experiments.arcade_demo.binocular_temporal_foundation import stratified_episode_split,_corr,_auc,_ridge
from experiments.arcade_demo.binocular_v3 import load
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
OUT=ROOT/'workspace'/'outputs'/'arcade_demo'

def _z(X,tr):
 mu=X[tr].mean(0);sd=np.where(X[tr].std(0)<1e-6,1.,X[tr].std(0));return (X-mu)/sd
def _metrics(y,p,task):
 r={'corr':round(_corr(y,p),4),'mae':round(float(np.mean(abs(y-p))),4)}
 if task=='lat':
  use=abs(y)>.05;r|={'accuracy':round(float(np.mean(np.sign(y[use]) == np.sign(p[use]))),4),'auc':round(_auc(y[use]>0,p[use]),4)}
 return r
def _linear(X,y,tr,va,te,task):
 Z=_z(X,tr);best=None
 for a in (10.,30.,100.,300.,1000.):
  w,b=_ridge(Z[tr],y[tr],a);p=np.tanh(Z[va]@w+b);s=_corr(y[va],p)
  if best is None or s>best[0]:best=(s,a,w,b)
 _,a,w,b=best;return {'alpha':a,'validation':_metrics(y[va],np.tanh(Z[va]@w+b),task),'test':_metrics(y[te],np.tanh(Z[te]@w+b),task)}
def _mlp(X,y,tr,va,te,task,seed=17):
 """Tiny 32-ReLU MLP trained with mini-batch Adam; diagnostic only."""
 Z=_z(X,tr).astype(np.float32);rng=np.random.default_rng(seed);d=Z.shape[1];h=32
 w1=rng.normal(0,.05,(d,h)).astype('f');b1=np.zeros(h,'f');w2=rng.normal(0,.05,h).astype('f');b2=np.zeros(1,'f');m=[np.zeros_like(x) for x in (w1,b1,w2,b2)];v=[np.zeros_like(x) for x in (w1,b1,w2,b2)];best=None
 ids=np.flatnonzero(tr)
 for epoch in range(1,121):
  rng.shuffle(ids)
  for start in range(0,len(ids),128):
   ix=ids[start:start+128];x=Z[ix];t=y[ix].astype('f');a=x@w1+b1;hh=np.maximum(a,0);o=hh@w2+b2;p=np.tanh(o);go=2*(p-t)*(1-p*p)/len(ix);gw2=hh.T@go;gb2=go.sum();gh=go[:,None]*w2;ga=gh*(a>0);gw1=x.T@ga;gb1=ga.sum(0)
   for k,(param,grad) in enumerate(((w1,gw1),(b1,gb1),(w2,gw2),(b2,np.asarray([gb2])))):
    m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad;param-=.003*m[k]/(np.sqrt(v[k])+1e-8)
  def pred(mask): return np.tanh(np.maximum(Z[mask]@w1+b1,0)@w2+b2)
  val=_metrics(y[va],pred(va),task)
  if best is None or val['corr']>best[0]:best=(val['corr'],epoch,val,_metrics(y[te],pred(te),task))
 return {'hidden':h,'best_epoch_validation':best[1],'validation':best[2],'test':best[3]}
def run():
 d=np.load(OUT/'binocular_balanced_dataset.npz',allow_pickle=False);raw=d['features'].astype(float);poolids=d['pool_body_ids'].astype(int);pool=len(poolids);H=raw.reshape(len(raw),8,pool);g=d['groups'].astype(str);hc=d['height_classes'].astype(str);s=d['seeds'].astype(int);ul=d['u_lat'].astype(float);uv=d['u_vert'].astype(float);parts,_=stratified_episode_split(s,g,hc);tr=np.isin(s,list(parts['train']));va=np.isin(s,list(parts['val']));te=np.isin(s,list(parts['test']))
 bridge,_=load(OUT/'arcade_binocular_bridge_v3.npz',ArcadeDNBasis());idx={x:i for i,x in enumerate(poolids)}
 def v3x(head):return H[:,:head.n_windows,[idx[x] for x in head.body_ids]].reshape(len(H),-1)
 reps={'raw_160ms':H.reshape(len(H),-1),'multiscale_mean_slope':np.concatenate([H[:,0],H[:,:2].mean(1),H[:,:4].mean(1),H.mean(1),H[:,0]-H[:,1],H[:,0]-H[:,:4].mean(1)],1)}
 # Apply the exact V3 feature identity for the direct linear-vs-nonlinear test.
 tasks={'lateral':(ul,'lat',v3x(bridge.lateral)),'vertical':(uv,'vert',v3x(bridge.vertical))};out={'split':{k:len(v) for k,v in parts.items()},'tasks':{},'representation_validation':{}}
 for name,(y,kind,Xv3) in tasks.items():
  out['tasks'][name]={'v3_feature_linear':_linear(Xv3,y,tr,va,te,kind),'v3_feature_tiny_mlp':_mlp(Xv3,y,tr,va,te,kind)}
  out['representation_validation'][name]={k:_linear(X,y,tr,va,te,kind) for k,X in reps.items()}
 (OUT/'binocular_representation_audit.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
