"""Train V3 only after the balanced temporal diagnostic supports it."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path: sys.path.insert(0,str(ROOT/"workspace"))
from experiments.arcade_demo.binocular_temporal_foundation import _design,_fit_task,stratified_episode_split,WINDOWS,BUDGETS
from experiments.arcade_demo.binocular_v3 import V3Head,save
OUT=ROOT/"workspace"/"outputs"/"arcade_demo"

def train(dataset="binocular_balanced_dataset.npz", out="arcade_binocular_bridge_v3.npz", split_seed=20260917):
    d=np.load(OUT/dataset,allow_pickle=False); raw=d["features"].astype(float); ids=d["pool_body_ids"].astype(np.int64); pool=len(ids)
    groups=d["groups"].astype(str); heights=d["height_classes"].astype(str); seeds=d["seeds"].astype(int); ul=d["u_lat"].astype(float); uv=d["u_vert"].astype(float); maxw=int(np.asarray(d["n_windows"]).ravel()[0])
    parts,_=stratified_episode_split(seeds,groups,heights,split_seed); masks={k:np.isin(seeds,list(v)) for k,v in parts.items()}
    candidates={"lateral":[],"vertical":[]}
    # Validation is the sole architecture selector.  Test predictions are not
    # consulted until after one lateral/one vertical candidate has been chosen.
    for win in WINDOWS:
        if win>maxw: continue
        X=_design(raw,win,pool)
        for budget in BUDGETS:
            for name,y in (("lateral",ul),("vertical",uv)):
                fit=_fit_task(X,y,masks["train"],masks["val"],masks["val"],pool,name,budget)
                candidates[name].append((fit["validation"]["corr"],win,budget,fit))
    chosen={name:max(items,key=lambda x:x[0]) for name,items in candidates.items()}
    heads={}; tests={}
    for name,y in (("lateral",ul),("vertical",uv)):
        _,win,budget,_=chosen[name]; X=_design(raw,win,pool)
        fit=_fit_task(X,y,masks["train"],masks["val"],masks["test"],pool,name,budget)
        head=V3Head(ids[fit["selected_indices"]],win,fit["W"],fit["b"],fit["mu"],fit["sd"]); heads[name]=head; tests[name]=fit["test"]
        chosen[name]=dict(window_bins=win,window_ms=win*20,budget=budget,alpha=fit["alpha"],validation=fit["validation"],test=fit["test"],body_ids=head.body_ids)
    meta=dict(version="arcade-binocular-bridge-v3",dataset=dataset,backend="metal",vision="both",split_seed=split_seed,split={k:sorted(map(int,v)) for k,v in parts.items()},heads=chosen,lat_gain=3.5,vert_gain=4.5,smoothing=.2,
      architecture="separate causal linear lateral and vertical heads; real DN injection retained",selection="feature/window/alpha selected with TRAIN/VAL only; test read once after selection")
    save(OUT/out,heads["lateral"],heads["vertical"],meta); (OUT/out.replace(".npz","_train.json")).write_text(json.dumps(meta,indent=2)); print(json.dumps(dict(lateral=chosen["lateral"],vertical=chosen["vertical"]),indent=2)); return meta

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="binocular_balanced_dataset.npz");p.add_argument("--out",default="arcade_binocular_bridge_v3.npz");a=p.parse_args();train(a.dataset,a.out)
