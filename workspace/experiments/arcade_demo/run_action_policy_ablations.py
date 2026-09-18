"""Run and persist the four causal eye ablations for the action policy."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path:sys.path.insert(0,str(ROOT/"workspace"))
from experiments.arcade_demo.evaluate_arcade_action_policy import cells,evaluate,OUT

def run(policy="arcade_binocular_action_policy_90.npz",per_cell=10,base_seed=290000,scale=.2):
    shot_cells=cells(per_cell,base_seed);result={"protocol":{"retina_map":"fullframe","matched_reset_corrected":True,"policy":policy,"policy_mode":"residual","residual_scale":scale,"n_shots_per_condition":len(shot_cells)},"conditions":{}}
    for condition in ("both","left_blind","right_blind","both_blind"):
        result["conditions"][condition]=evaluate(shot_cells,policy,scale,condition,"residual")
        print(condition,result["conditions"][condition]["overall"],flush=True)
    (OUT/"arcade_binocular_action_policy_eye_ablations.json").write_text(json.dumps(result,indent=2))
    return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--policy",default="arcade_binocular_action_policy_90.npz");p.add_argument("--per-cell",type=int,default=10);p.add_argument("--base-seed",type=int,default=290000);p.add_argument("--scale",type=float,default=.2);a=p.parse_args();run(a.policy,a.per_cell,a.base_seed,a.scale)
