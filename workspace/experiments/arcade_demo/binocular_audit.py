"""Binocular Arcade experiment - STEP 1: audit eye / receptor mapping.

Before implementing two-eye vision we establish, explicitly and reproducibly:

  1. Which R1-R6 receptor body IDs belong to the LEFT eye vs the RIGHT eye.
     The upstream retinal projection (doom/prepare.py) assigns each receptor to
     an eye by the connectome `rootSide` field (NOT somaSide, which is ~all None
     for retina), then lays out overlapping image viewports:
         left  eye: uv_x = 0.60 * z            -> uv_x in [0.00, 0.60]
         right eye: uv_x = 0.40 + 0.60*(1 - z)  -> uv_x in [0.40, 1.00]
     So the current single-image sampler reads BOTH eye populations from one
     rendered frame; the right-eye receptors are sampling left-camera pixels.

  2. That the eye_left and eye_right cameras produce genuinely distinct images.

  3. Per-eye retinal response to a deterministic shot (both cameras), confirming
     each eye carries usable, distinct luminance.

Writes an explicit manifest (receptor index, body id, rootSide, uv) to
workspace/outputs/arcade_demo/binocular_eye_manifest.npz + a JSON summary.
Analysis only; nothing is retrained and Science Mode is untouched.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

import pandas as pd
import mujoco

from adapters.brain import MaleCNSBrain
from doom.game import retinal_samples
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_S

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def build_eye_manifest():
    """Map each retina receptor to an eye via connectome rootSide + record uv."""
    b = MaleCNSBrain(backend="cpu")
    uv = np.asarray(b._brain.uv)                      # (n_ret, 2)
    retina = np.asarray(b._brain.retina)              # graph indices
    ids = np.asarray(b._brain.ids)
    retina_ids = ids[retina].astype(np.int64)
    b.close()

    ann = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/"
                          "malecns_v1/annotations.feather")
    id2root = dict(zip(ann["bodyId"].astype(int), ann["rootSide"].astype(str)))
    roots = np.array([id2root.get(int(i), "?") for i in retina_ids])

    left_mask = roots == "L"
    right_mask = roots == "R"
    manifest = dict(
        n_receptors=int(len(retina_ids)),
        rootSide_counts={k: int(v) for k, v in Counter(roots).items()},
        left=dict(n=int(left_mask.sum()),
                  uv_x_min=float(uv[left_mask, 0].min()),
                  uv_x_max=float(uv[left_mask, 0].max()),
                  uv_x_mean=float(uv[left_mask, 0].mean())),
        right=dict(n=int(right_mask.sum()),
                   uv_x_min=float(uv[right_mask, 0].min()),
                   uv_x_max=float(uv[right_mask, 0].max()),
                   uv_x_mean=float(uv[right_mask, 0].mean())),
        overlap_uv_x_0p4_0p6=dict(
            left=int(((roots == "L") & (uv[:, 0] >= 0.4) & (uv[:, 0] <= 0.6)).sum()),
            right=int(((roots == "R") & (uv[:, 0] >= 0.4) & (uv[:, 0] <= 0.6)).sum())),
        laterality_field="rootSide (somaSide is ~all None for retina)",
        viewport_projection=("left uv_x=0.60*z; right uv_x=0.40+0.60*(1-z) "
                             "(overlapping viewports, doom/prepare.py)"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "binocular_eye_manifest.npz",
             retina_graph_index=retina.astype(np.int64),
             retina_body_id=retina_ids,
             rootSide=roots.astype("U4"),
             uv=uv.astype(np.float32),
             left_mask=left_mask, right_mask=right_mask)
    return manifest, uv, roots, retina_ids


def audit_cameras(n_steps=24, seed=90000):
    """Render both eye cameras mid-approach for L and R shots; compare images and
    per-eye retinal luminance (each eye sampling its OWN camera)."""
    man = np.load(OUT / "binocular_eye_manifest.npz", allow_pickle=False)
    uv = man["uv"]; left_mask = man["left_mask"]; right_mask = man["right_mask"]

    w = ArcadeGoalkeeperWorld(seed=seed)
    r = mujoco.Renderer(w.model, height=96, width=160)
    out = {}
    for group in ("left", "center", "right"):
        w.reset(w.sample_shot(group, height_frac=0.0))
        for _ in range(n_steps):
            w.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
            w.step(DECISION_S)
        r.update_scene(w.data, camera="eye_left"); img_l = r.render()
        r.update_scene(w.data, camera="eye_right"); img_r = r.render()
        img_absdiff = float(np.abs(img_l.astype(float) - img_r.astype(float)).mean())
        # one-eye (current): ALL receptors sample the LEFT image
        lum_oneeye = retinal_samples(img_l, uv)
        # binocular: left receptors <- left image, right receptors <- right image
        lum_bino = retinal_samples(img_l, uv).copy()
        lum_r = retinal_samples(img_r, uv)
        lum_bino[right_mask] = lum_r[right_mask]
        # how much the right-eye receptors change when fed the right camera
        right_delta = float(np.abs(lum_bino[right_mask]
                                   - lum_oneeye[right_mask]).mean())
        out[group] = dict(
            left_vs_right_img_absdiff_mean=round(img_absdiff, 2),
            left_eye_lum_mean=round(float(lum_oneeye[left_mask].mean()), 4),
            right_eye_lum_mean_oneeye=round(float(lum_oneeye[right_mask].mean()), 4),
            right_eye_lum_mean_binocular=round(float(lum_bino[right_mask].mean()), 4),
            right_receptor_luminance_change=round(right_delta, 4),
        )
    return out


def main():
    manifest, uv, roots, rids = build_eye_manifest()
    cams = audit_cameras()
    report = dict(eye_manifest=manifest, camera_audit=cams)
    (OUT / "binocular_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nsaved binocular_eye_manifest.npz + binocular_audit.json")


if __name__ == "__main__":
    main()
