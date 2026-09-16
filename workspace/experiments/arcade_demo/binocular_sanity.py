"""Binocular sanity + ablation tests (analysis only; nothing retrained).

For deterministic LEFT / CENTER / RIGHT shots, run the fly through the arcade
world (bridge OFF) under each vision condition and measure:

  * per-eye retinal luminance (left vs right receptor populations);
  * optic-lobe activity per hemisphere (somaSide L vs R optic-lobe neurons);
  * whether each eye contributes DISTINCT information (both vs mono_left vs
    left_only vs right_only) and how blinding an eye changes activity.

This verifies the two cameras feed genuinely different neural activity before we
spend compute training a binocular bridge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

import pandas as pd

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def optic_lobe_hemisphere_ids(brain):
    """Return (left_ol_body_ids, right_ol_body_ids) for optic-lobe intrinsic
    neurons, split by somaSide (OL neurons DO carry somaSide, unlike retina)."""
    ann = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/"
                          "malecns_v1/annotations.feather")
    is_ol = ann["superclass"].astype(str).str.contains("optic", case=False,
                                                        na=False)
    ol = ann[is_ol]
    all_ids = set(int(x) for x in brain._brain.ids)
    left = [int(b) for b, s in zip(ol["bodyId"].astype(int),
                                   ol["somaSide"].astype(str))
            if s == "L" and int(b) in all_ids]
    right = [int(b) for b, s in zip(ol["bodyId"].astype(int),
                                    ol["somaSide"].astype(str))
             if s == "R" and int(b) in all_ids]
    # subsample for speed of read (few hundred each is plenty for a mean)
    rng = np.random.default_rng(0)
    if len(left) > 500:
        left = list(rng.choice(left, 500, replace=False))
    if len(right) > 500:
        right = list(rng.choice(right, 500, replace=False))
    return left, right


def run_condition(brain, condition, group, seed, n_steps=30, ol_left=None,
                  ol_right=None):
    """Run one shot under one vision condition; return end-approach activity."""
    w = ArcadeGoalkeeperWorld(seed=seed)
    vis = BinocularVisionBridge(w.fly, brain, condition=condition)
    w.reset(w.sample_shot(group, height_frac=0.0))
    brain.reset()
    last = None
    read_ids = (ol_left or []) + (ol_right or [])
    for _ in range(n_steps):
        lum, info = vis.perceive()
        brain.step(DECISION_MS)
        last = info
        w.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
        w.step(DECISION_S)
    rd = brain.read(read_ids) if read_ids else {}
    ol_l = float(np.mean([rd[i]["spikes"] for i in (ol_left or [])])) if ol_left else 0.0
    ol_r = float(np.mean([rd[i]["spikes"] for i in (ol_right or [])])) if ol_right else 0.0
    return dict(left_lum=round(last["left_lum_mean"], 4),
                right_lum=round(last["right_lum_mean"], 4),
                ol_left_spikes=round(ol_l, 4), ol_right_spikes=round(ol_r, 4))


def main(backend="metal", seed=91000):
    brain = MaleCNSBrain(backend=backend)
    ol_left, ol_right = optic_lobe_hemisphere_ids(brain)
    conditions = ["both", "mono_left", "left_only", "right_only",
                  "left_blind", "right_blind", "both_blind"]
    report = {"n_ol_left": len(ol_left), "n_ol_right": len(ol_right),
              "per_group": {}}
    for group in ("left", "center", "right"):
        report["per_group"][group] = {}
        for cond in conditions:
            r = run_condition(brain, cond, group, seed, ol_left=ol_left,
                              ol_right=ol_right)
            report["per_group"][group][cond] = r
            print(f"  {group:6s} {cond:11s} Llum {r['left_lum']:.3f} "
                  f"Rlum {r['right_lum']:.3f} OL_L {r['ol_left_spikes']:.2f} "
                  f"OL_R {r['ol_right_spikes']:.2f}", flush=True)
        seed += 1
    brain.close()

    # distinctness summary: does `both` differ from `mono_left`? (right eye adds info)
    def _delta(g):
        b = report["per_group"][g]["both"]
        m = report["per_group"][g]["mono_left"]
        return dict(
            right_lum_both_minus_mono=round(b["right_lum"] - m["right_lum"], 4),
            ol_right_both_minus_mono=round(b["ol_right_spikes"]
                                           - m["ol_right_spikes"], 4))
    report["binocular_adds_info"] = {g: _delta(g) for g in
                                     ("left", "center", "right")}
    (OUT / "binocular_sanity.json").write_text(json.dumps(report, indent=2))
    print("\nbinocular vs mono_left (right-eye contribution):")
    print(json.dumps(report["binocular_adds_info"], indent=2))
    print("saved binocular_sanity.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="metal")
    a = p.parse_args()
    main(backend=a.backend)
