"""Honest one-eye (v2) vs binocular offline comparison + blind-eye ablations.

Offline (held-out, identical episode split):
  * v2 one-eye        : frozen 207 neurons, arcade_dataset_v2.npz
  * binocular selected : 207 data-selected bilateral neurons, binocular_dataset.npz
  * binocular frozen207: same 207 as v2 but two-eye vision (isolates vision only)

Blind-eye ablation (on the binocular pool dataset): re-collect a small held-out
shot set under left_only / right_only / both and score the SELECTED binocular
model's lateral decodability, to see whether each eye contributes.

Analysis only; nothing retrained here (models already frozen by the trainers).
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

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _load_train_report(name):
    return json.load(open(OUT / name))


def offline_table():
    v2 = _load_train_report("arcade_bridge_train_v2.json")["meta"]["test"]
    bino = _load_train_report("binocular_bridge_train.json")["meta"]["test"]
    return {
        "v2_one_eye_frozen207": {
            "lateral_corr": v2["lateral"]["corr"],
            "lateral_dir_acc": v2["lateral"]["direction_accuracy"],
            "vertical_corr": v2["vertical"]["corr"]},
        "binocular_selected207": {
            "lateral_corr": bino["lateral"]["corr"],
            "lateral_dir_acc": bino["lateral"]["direction_accuracy"],
            "vertical_corr": bino["vertical"]["corr"],
            "selected_sides": _load_train_report(
                "binocular_bridge_train.json")["meta"]["selected_side_counts"]},
    }


def blind_eye_ablation(n_per_group=4, base_seed=93000, backend="metal"):
    """Score the frozen binocular SELECTED model's lateral decode under each eye
    condition on a small held-out shot set (fresh seeds, not in any split)."""
    from adapters.brain import MaleCNSBrain
    from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
    from experiments.arcade_demo.arcade_shots import teacher_command
    from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S
    from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
    from experiments.arcade_demo.arcade_bridge import LinearBridge2D, MetalSafeFeatureExtractor

    meta = _load_train_report("binocular_bridge_train.json")["meta"]
    sel_ids = [int(x) for x in meta["selected_body_ids"]]
    model = LinearBridge2D.load(OUT / "binocular_bridge_model.npz")
    nW = int(meta["n_windows"])

    brain = MaleCNSBrain(backend=backend)
    out = {}
    for cond in ("both", "left_only", "right_only"):
        preds, labels, groups = [], [], []
        seed = base_seed
        for group in ("left", "center", "right"):
            for _ in range(n_per_group):
                w = ArcadeGoalkeeperWorld(seed=seed)
                vis = BinocularVisionBridge(w.fly, brain, condition=cond)
                ex = MetalSafeFeatureExtractor(sel_ids, n_windows=nW)
                w.reset(w.sample_shot(group, height_frac=0.0)); brain.reset(); ex.reset()
                while w.shot_live:
                    vis.perceive(); brain.step(DECISION_MS); ex.observe(brain)
                    x = ex.features(order=model.feature_order)
                    p = model.predict(x)
                    (ul, _), _ = teacher_command(w)
                    preds.append(float(p[0])); labels.append(float(ul)); groups.append(group)
                    w.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0); w.step(DECISION_S)
                seed += 1
        preds = np.array(preds); labels = np.array(labels); groups = np.array(groups)
        lr = groups != "center"
        corr = (float(np.corrcoef(labels[lr], preds[lr])[0, 1])
                if lr.sum() > 2 and preds[lr].std() > 1e-9 else 0.0)
        dir_acc = float(np.mean(np.sign(preds[lr]) == np.sign(labels[lr]))) if lr.any() else 0.0
        out[cond] = dict(lateral_corr=round(corr, 4),
                         lateral_dir_acc=round(dir_acc, 4), n_lr=int(lr.sum()))
        print(f"  {cond:11s} lateral_corr {out[cond]['lateral_corr']:.3f} "
              f"dir_acc {out[cond]['lateral_dir_acc']:.3f}", flush=True)
    brain.close()
    return out


def main():
    report = {"offline": offline_table()}
    print(json.dumps(report["offline"], indent=2))
    print("\nblind-eye ablation (binocular selected model):")
    report["blind_eye_ablation"] = blind_eye_ablation()
    (OUT / "binocular_compare.json").write_text(json.dumps(report, indent=2))
    print("saved binocular_compare.json")


if __name__ == "__main__":
    main()
