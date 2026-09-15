"""Closed-loop bridge-gain calibration on VALIDATION seeds (not the test set).

The frozen linear model outputs u = tanh(...) in (-1,1); the bridge gain scales
|u| before the DN motor basis. Gain is the one runtime scalar we still choose,
and per Section 24 it must be frozen BEFORE the final test evaluation and chosen
WITHOUT touching the test seeds. We sweep gain on a dedicated validation shot
set (seeds disjoint from both the training dataset and the 90000-base test set)
and pick the gain maximizing overall save rate, then that value is frozen into
the bridge for evaluate.py.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.learned_bridge.evaluate import (make_shot_set, eval_neural,
                                                 summarize, build_bridge)

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
GROUPS = ("left", "center", "right")


def val_seeds(n_per_group, base_seed=70000):
    sg = []; s = base_seed
    for g in GROUPS:
        for _ in range(n_per_group):
            sg.append((s, g)); s += 1
    return sg


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--gains", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    p.add_argument("--smoothing", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    p.add_argument("--model", type=str, default="bridge_model.npz")
    a = p.parse_args()

    shots = make_shot_set(val_seeds(a.per_group))
    print(f"[gain-cal] {len(shots)} validation shots ({a.per_group}/group), "
          f"seeds 70000..  gains={a.gains} smoothing={a.smoothing}")
    res = {}
    t0 = time.perf_counter()
    for gain in a.gains:
        for sm in a.smoothing:
            bridge, meta = build_bridge(OUT / a.model, gain=gain, cmd_smoothing=sm)
            rows = eval_neural(shots, bridge=bridge, condition="normal", enabled=True)
            s = summarize(rows)
            key = f"gain={gain},sm={sm}"
            res[key] = dict(save_rate=s["save_rate"],
                            by_group={g: s["by_group"][g]["save_rate"] for g in GROUPS})
            print(f"  {key}: {s['save_rate']}  "
                  f"L={s['by_group']['left']['save_rate']} "
                  f"C={s['by_group']['center']['save_rate']} "
                  f"R={s['by_group']['right']['save_rate']}")
    # prefer the config maximizing overall save rate, tie-break by min(L,R)
    # balance so a symmetric controller wins over a one-sided one.
    def score(kv):
        v = kv[1]; bg = v["by_group"]
        bal = min(bg["left"] or 0, bg["right"] or 0)
        return (v["save_rate"], bal)
    best = max(res.items(), key=score)
    out = dict(per_group=a.per_group, results=res, best_config=best[0],
               best_save_rate=best[1]["save_rate"],
               wall_seconds=round(time.perf_counter() - t0, 1))
    (OUT / "calibrate_gain.json").write_text(json.dumps(out, indent=2))
    print(f"\nBEST: {best[0]}  save_rate={best[1]['save_rate']}")
    print("saved calibrate_gain.json")


if __name__ == "__main__":
    main()
