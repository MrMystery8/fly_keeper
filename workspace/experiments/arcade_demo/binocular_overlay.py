"""PHASE 3 overlays: camera image + receptor sample locations + ball centroid.

Renders eye_left and eye_right for a few representative world ball positions and
draws:
  * the receptor UV sample locations for THIS eye's population (as points),
  * the ball's image centroid (from ball-removal difference),
  * the receptor-population uv_x coverage window (vertical guide lines).

This makes the receptor-coverage vs ball-landing mismatch visually obvious.
Saves PNGs to workspace/outputs/arcade_demo/calib_overlays/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.binocular_vision import load_eye_manifest
from experiments.arcade_demo.binocular_calibration import (
    place_ball, hide_ball, render, ball_image_stats, W, H)

OUT = ROOT / "workspace" / "outputs" / "arcade_demo" / "calib_overlays"

CASES = [
    ("FARLEFT_MID", 1.2, 1.2, 0.7),
    ("LEFT_MID", 1.2, 0.6, 0.7),
    ("CENTER_MID", 1.2, 0.0, 0.7),
    ("RIGHT_MID", 1.2, -0.6, 0.7),
    ("FARRIGHT_MID", 1.2, -1.2, 0.7),
]


def draw(img, uv_px, ball_cx, ball_cy, win_lo, win_hi, path):
    im = img.copy()
    # receptor sample points: mark green
    for (px, py) in uv_px:
        xi, yi = int(round(px)), int(round(py))
        if 0 <= xi < W and 0 <= yi < H:
            im[max(0, yi - 0):min(H, yi + 1), max(0, xi):min(W, xi + 1)] = [0, 255, 0]
    # coverage window guide lines (cyan)
    for wx in (win_lo, win_hi):
        wxi = int(round(wx))
        if 0 <= wxi < W:
            im[:, wxi] = [0, 200, 255]
    # ball centroid: red cross
    if ball_cx is not None:
        cx, cy = int(round(ball_cx)), int(round(ball_cy))
        im[max(0, cy - 3):min(H, cy + 4), max(0, cx):min(W, cx + 1)] = [255, 0, 0]
        im[max(0, cy):min(H, cy + 1), max(0, cx - 3):min(W, cx + 4)] = [255, 0, 0]
    try:
        from PIL import Image
        Image.fromarray(im).save(path)
        return True
    except Exception:
        np.save(str(path).replace(".png", ".npy"), im)
        return False


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    man = load_eye_manifest()
    uv = man["uv"]; lm = man["left_mask"].astype(bool); rm = man["right_mask"].astype(bool)
    uv_px_l = np.stack([uv[lm, 0] * (W - 1), uv[lm, 1] * (H - 1)], 1)
    uv_px_r = np.stack([uv[rm, 0] * (W - 1), uv[rm, 1] * (H - 1)], 1)
    win_l = (uv[lm, 0].min() * (W - 1), uv[lm, 0].max() * (W - 1))
    win_r = (uv[rm, 0].min() * (W - 1), uv[rm, 0].max() * (W - 1))

    world = ArcadeGoalkeeperWorld(seed=770000)
    world.fly.set_pose(xy=(0.0, 0.0), yaw=0.0)
    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
    mujoco.mj_forward(world.model, world.data)
    r = mujoco.Renderer(world.model, height=H, width=W)

    saved = []
    for name, x, y, z in CASES:
        hide_ball(world)
        bg_l = render(r, world, "eye_left"); bg_r = render(r, world, "eye_right")
        place_ball(world, x, y, z)
        img_l = render(r, world, "eye_left"); img_r = render(r, world, "eye_right")
        sl = ball_image_stats(img_l, bg_l); sr = ball_image_stats(img_r, bg_r)
        pL = OUT / f"{name}_LEFTeye.png"; pR = OUT / f"{name}_RIGHTeye.png"
        okL = draw(img_l, uv_px_l, sl["cx"], sl["cy"], win_l[0], win_l[1], pL)
        okR = draw(img_r, uv_px_r, sr["cx"], sr["cy"], win_r[0], win_r[1], pR)
        saved += [str(pL if okL else str(pL).replace('.png', '.npy')),
                  str(pR if okR else str(pR).replace('.png', '.npy'))]
        print(f"{name}: LEFTeye ball cx={sl['cx']} (Lrecept window px "
              f"{win_l[0]:.0f}-{win_l[1]:.0f}) | RIGHTeye ball cx={sr['cx']} "
              f"(Rrecept window px {win_r[0]:.0f}-{win_r[1]:.0f})")
    print("\nsaved overlays:")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    run()
