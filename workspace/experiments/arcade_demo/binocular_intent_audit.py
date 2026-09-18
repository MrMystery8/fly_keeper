"""Analysis-only comparison of thrust labels with interception-intent labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'workspace') not in sys.path: sys.path.insert(0,str(ROOT/'workspace'))
from experiments.arcade_demo.binocular_temporal_foundation import ALPHAS,BUDGETS,WINDOWS,_corr,_design,_ridge,_select,stratified_episode_split
from embodiment.mujoco_world import BALL_RADIUS, GOAL_HEIGHT
OUT=ROOT/'workspace'/'outputs'/'arcade_demo'

def _fit(raw,target,train,val,test,pool,max_w):
    candidates=[]
    for window in WINDOWS:
        if window>max_w:continue
        X=_design(raw,window,pool); act=X.reshape(len(X),window,pool).sum(1)
        for budget in BUDGETS:
            chosen,_=_select(act,target,train,budget); xs=X.reshape(len(X),window,pool)[:,:,chosen].reshape(len(X),-1)
            mu=xs[train].mean(0); sd=np.where(xs[train].std(0)<1e-6,1.,xs[train].std(0)); z=(xs-mu)/sd
            for alpha in ALPHAS:
                w,b=_ridge(z[train],target[train],alpha); p=np.tanh(z@w+b)
                candidates.append((_corr(target[val],p[val]),window,budget,alpha,p))
    _,window,budget,alpha,p=max(candidates,key=lambda r:r[0])
    return p,dict(window_bins=window,budget=budget,alpha=alpha,validation_corr=round(_corr(target[val],p[val]),4),test_corr=round(_corr(target[test],p[test]),4))

def _classification(labels,p,names,train_labels,train_target):
    """Classify using TRAIN target centroids, respecting physical sign/order."""
    centers=np.array([train_target[train_labels==n].mean() for n in names])
    pred=np.abs(p[:,None]-centers).argmin(1); truth=np.array([names.index(x) for x in labels])
    conf=[[int(((truth==i)&(pred==j)).sum()) for j in range(3)] for i in range(3)]
    return dict(accuracy=round(float((truth==pred).mean()),4),confusion_matrix=dict(rows_true=list(names),columns_predicted=list(names),values=conf),class_mean_prediction={n:round(float(p[truth==i].mean()),4) for i,n in enumerate(names)})

def main():
    d=np.load(OUT/'binocular_balanced_dataset.npz',allow_pickle=False)
    raw=d['features'].astype(float); pool=len(d['pool_body_ids']); max_w=int(d['n_windows'][0]); seeds=d['seeds'].astype(int); groups=d['groups'].astype(str); heights=d['height_classes'].astype(str); steps=d['steps'].astype(int)
    parts,_=stratified_episode_split(seeds,groups,heights); masks={k:np.isin(seeds,list(v)) for k,v in parts.items()}
    # Normalized destination labels have physical meaning but fit the same
    # bounded linear decoder as motor targets.  Labels are training-only.
    # A destination must remain stable for an episode.  Per-timestep teacher
    # extrapolation becomes nonsensical after an uncommanded ball hits a post;
    # use the launch-time goal-plane destination as the privileged TRAIN label
    # and repeat it only as a label, never as input.
    yraw=d['teacher_y_cross'].astype(float).copy()
    for seed in np.unique(seeds):
        idx=np.flatnonzero(seeds==seed); yraw[idx]=yraw[idx[0]]
    # The explicit height fraction and Arcade shot construction are already in
    # this balanced artifact. Reconstruct the immutable target directly rather
    # than trusting the legacy dynamic extrapolation column.
    zraw=BALL_RADIUS+d['heights'].astype(float)*(GOAL_HEIGHT-BALL_RADIUS-.1)
    yscale=max(float(np.abs(yraw[masks['train']]).max()),1e-6); ystar=np.clip(yraw/yscale,-1,1)
    zmin,zmax=zraw[masks['train']].min(),zraw[masks['train']].max(); zstar=np.clip(2*(zraw-zmin)/max(zmax-zmin,1e-6)-1,-1,1)
    ul=d['u_lat'].astype(float); uv=d['u_vert'].astype(float)
    outputs={}
    for name,target,classnames in [('motor_lateral',ul,('left','center','right')),('intent_y_star',ystar,('left','center','right')),('motor_vertical',uv,('low','mid','high')),('intent_z_star',zstar,('low','mid','high'))]:
        p,meta=_fit(raw,target,masks['train'],masks['val'],masks['test'],pool,max_w)
        semantic=groups if 'lateral' in name or 'y_star' in name else heights
        row=dict(**meta,test_classification=_classification(semantic[masks['test']],p[masks['test']],classnames,semantic[masks['train']],target[masks['train']]))
        # Fixed causal times, not re-selected: model selected globally on VAL.
        row['fixed_causal_times_ms']={}
        for ms in (80,160,240):
            m=masks['test']&(steps==ms//20)
            if m.any(): row['fixed_causal_times_ms'][str(ms)]=dict(n=int(m.sum()),corr=round(_corr(target[m],p[m]),4),classification=_classification(semantic[m],p[m],classnames,semantic[masks['train']],target[masks['train']]))
        outputs[name]=row
    out=dict(analysis_only=True,dataset='binocular_balanced_dataset.npz',split_episode_counts={k:len(v) for k,v in parts.items()},normalization=dict(y_star=f'launch-time teacher_y_cross / train abs-max {yscale:.5f}',z_star=f'launch-time teacher_z_cross; train-range [{zmin:.5f}, {zmax:.5f}] mapped to [-1,1]'),models=outputs,limitations=['Teacher interception coordinates are stable launch-time labels only; neural activity is the sole decoder input.','z* scaling is fitted on TRAIN only.','Fixed-time reports use only causal bins available by that step.'])
    (OUT/'binocular_interception_intent_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
