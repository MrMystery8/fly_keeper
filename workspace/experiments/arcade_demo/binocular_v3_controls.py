"""Held-out causal eye controls for the frozen V3 bridge (no retraining)."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"workspace") not in sys.path: sys.path.insert(0,str(ROOT/"workspace"))
from experiments.arcade_demo.binocular_evaluate import _cells,eval_neural,OUT

if __name__=="__main__":
    cells=_cells(1,94000) # explicitly fresh, distinct from final 90000 matrix
    out={"n_shots_per_condition":len(cells),"base_seed":94000,"model":"arcade_binocular_bridge_v3.npz","conditions":{}}
    for cond in ("both","left_blind","right_blind","both_blind"):
        out["conditions"][cond]=eval_neural(cells,"v3",eye_condition=cond)
        print(cond,out["conditions"][cond]["overall"],flush=True)
    (OUT/"arcade_binocular_v3_eye_controls.json").write_text(json.dumps(out,indent=2))
