"""PHASE 1-3: deterministic binocular visibility / retinotopy calibration.

This is an ANALYSIS experiment. Nothing is trained; Science Mode is untouched.

It answers, experimentally and reproducibly:

  PHASE 1  Can each eye camera actually SEE the ball across the goalkeeper
           visual envelope (LEFT/CENTER/RIGHT x LOW/MID/HIGH, plus intermediate
           positions)? For each world ball position we render eye_left and
           eye_right and measure, per eye:
               ball visible?      (ball-removal render difference)
               pixel centroid     (image x, y in pixels)
               pixel area         (# changed pixels, and summed magnitude)
           and the receptor-space landing (which UV the ball lands on, and how
           many LEFT vs RIGHT rootSide receptors actually change).

  PHASE 2  Report the loaded camera geometry (body attachment, local pos, local
           quat, fovy) AND the derived WORLD pose (position + optical axis) of
           each eye when the fly stands on the goal line. Determine whether the
           two cameras are plausible mirror counterparts and where each eye's
           optical axis actually points relative to the incoming-ball axis.

  PHASE 3  Trace how a synthetic ball at world-left / center / right moves across
           each retina (image pixel -> receptor UV) and confirm the
           left/right UV orientation (the two eyes may legitimately encode
           horizontal position in opposite image-coordinate directions because
           the right viewport uses uv_x = 0.40 + 0.60*(1-z)).

The ball is PLACED (kinematically) at controlled world positions rather than
fired, so visibility is measured on the exact geometry we care about, decoupled
from shot dynamics. The fly is held at its standing goal-line pose (yaw=0,
facing +x toward incoming shots -- matching ArcadeGoalkeeperWorld.reset).

Outputs:
  workspace/outputs/arcade_demo/binocular_calibration.json   (full tables)
  workspace/outputs/arcade_demo/calib_overlays/*.png         (optional overlays)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from doom.game import retinal_samples
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.binocular_vision import load_eye_manifest
from embodiment.mujoco_world import (GOAL_LINE_X, GOAL_HALF_WIDTH, GOAL_HEIGHT,
                                     SHOT_X, BALL_RADIUS)

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
W, H = 160, 96

# --- visual envelope grid (world coordinates) -------------------------------
# y: lateral. The goal mouth spans +/-1.6 cm; shots aim at +/-~0.55*scale. We
# probe a wide lateral band so we can see where each eye's coverage ends.
Y_LABELS = ["FARLEFT", "LEFT", "CENTER", "RIGHT", "FARRIGHT"]
Y_VALUES = [1.2, 0.6, 0.0, -0.6, -1.2]          # +y is world-left
# z: height. LOW = rolling ball, HIGH = near crossbar.
Z_LABELS = ["LOW", "MID", "HIGH"]
Z_VALUES = [BALL_RADIUS, 0.7, 1.25]
# x: ball depth in front of the goal. We probe the approach band the keeper
# actually decides in: near the spawn, mid-flight, and near the line.
X_LABELS = ["FAR", "MID", "NEAR"]
X_VALUES = [2.4, 1.2, 0.4]


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def camera_world_pose(model, data, cam_name):
    """Return (world_pos(3), world_optical_axis(3)) for a camera.

    MuJoCo camera looks down its local -z axis; local +x is image-right, local
    +y is image-up. We read the live xmat/xpos so the head joint pose is
    included.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    pos = data.cam_xpos[cid].copy()
    mat = data.cam_xmat[cid].reshape(3, 3)
    # columns of xmat are the camera's local x,y,z axes in world frame
    axis_look = -mat[:, 2]        # optical axis (viewing direction)
    axis_right = mat[:, 0]
    axis_up = mat[:, 1]
    return pos, axis_look, axis_right, axis_up


def place_ball(world, x, y, z):
    q = world.data.qpos
    q[world._ball_qadr + 0] = x
    q[world._ball_qadr + 1] = y
    q[world._ball_qadr + 2] = z
    q[world._ball_qadr + 3:world._ball_qadr + 7] = [1, 0, 0, 0]
    world.data.qvel[world._ball_dofadr:world._ball_dofadr + 6] = 0.0
    mujoco.mj_forward(world.model, world.data)


def hide_ball(world):
    place_ball(world, 100.0, 100.0, 100.0)


def render(renderer, world, cam):
    renderer.update_scene(world.data, camera=cam)
    return renderer.render().copy()


def ball_image_stats(img, bg, thresh=8.0):
    """Ball visibility from a render-difference vs the ball-removed background.

    Returns dict(visible, area_px, mag_sum, cx, cy) where (cx,cy) is the
    intensity-weighted centroid in PIXELS (x image-right, y image-down).
    """
    d = np.abs(img.astype(np.float32) - bg.astype(np.float32)).mean(axis=2)
    mask = d > thresh
    area = int(mask.sum())
    mag = float(d.sum())
    if area == 0:
        return dict(visible=False, area_px=0, mag_sum=round(mag, 2),
                    cx=None, cy=None)
    ys, xs = np.indices(d.shape)
    w = d * mask
    tot = w.sum()
    cx = float((xs * w).sum() / tot)
    cy = float((ys * w).sum() / tot)
    return dict(visible=True, area_px=area, mag_sum=round(mag, 2),
                cx=round(cx, 2), cy=round(cy, 2))


def receptor_landing(img, bg, uv, left_mask, right_mask):
    """How the ball lands on the retina (per eye receptor population).

    Samples luminance at every receptor UV for the ball frame and the
    ball-removed frame; the positive delta is the retinal 'ball' signal. Returns
    per-population delta stats + the delta-weighted UV centroid.
    """
    lum = retinal_samples(img, uv)
    base = retinal_samples(bg, uv)
    delta = np.maximum(lum - base, 0.0)

    def pop(mask):
        d = delta[mask]
        tot = float(d.sum())
        if tot < 1e-6:
            return dict(delta_mean=round(float(d.mean()), 5),
                        delta_sum=round(tot, 5), n_active=int((d > 1e-3).sum()),
                        uv_centroid=None)
        uvc = np.average(uv[mask], axis=0, weights=d + 1e-9)
        return dict(delta_mean=round(float(d.mean()), 5),
                    delta_sum=round(tot, 5), n_active=int((d > 1e-3).sum()),
                    uv_centroid=[round(float(uvc[0]), 4), round(float(uvc[1]), 4)])

    return dict(left=pop(left_mask), right=pop(right_mask))


def run(seed=770000, save_overlays=True):
    man = load_eye_manifest()
    uv = man["uv"]
    left_mask = man["left_mask"].astype(bool)
    right_mask = man["right_mask"].astype(bool)

    world = ArcadeGoalkeeperWorld(seed=seed)
    # Put the fly in the exact standing goal-line pose used at episode start.
    world.fly.set_pose(xy=(0.0, 0.0), yaw=0.0)
    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
    mujoco.mj_forward(world.model, world.data)
    renderer = mujoco.Renderer(world.model, height=H, width=W)

    # ---------- PHASE 2: camera geometry (local defn + live world pose) -------
    cam_geom = {}
    for name in ("eye_left", "eye_right"):
        cid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        bid = int(world.model.cam_bodyid[cid])
        body = mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        pos, look, right, up = camera_world_pose(world.model, world.data, name)
        cam_geom[name] = dict(
            attached_body=body,
            local_pos=world.model.cam_pos[cid].round(5).tolist(),
            local_quat=world.model.cam_quat[cid].round(5).tolist(),
            fovy_deg=float(world.model.cam_fovy[cid]),
            world_pos=pos.round(5).tolist(),
            world_optical_axis=look.round(4).tolist(),
            world_image_right_axis=right.round(4).tolist(),
            world_image_up_axis=up.round(4).tolist(),
        )
    # angle of each optical axis to the +x (incoming-ball) axis and to each other
    la = np.asarray(cam_geom["eye_left"]["world_optical_axis"])
    ra = np.asarray(cam_geom["eye_right"]["world_optical_axis"])
    xhat = np.array([1.0, 0.0, 0.0])

    def ang(a, b):
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))

    cam_geom["_derived"] = {
        "left_axis_to_x_deg": round(ang(la, xhat), 2),
        "right_axis_to_x_deg": round(ang(ra, xhat), 2),
        "interocular_axis_angle_deg": round(ang(la, ra), 2),
        "note": ("Cameras sit on the head body; +x is the incoming-ball axis. "
                 "fly compound eyes point laterally, so large axis-to-+x angles "
                 "are expected/biological, not a bug."),
    }

    # ---------- PHASE 1 + 3: visibility grid ---------------------------------
    if save_overlays:
        (OUT / "calib_overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    left_visible = 0
    right_visible = 0
    both_visible = 0
    total = 0
    for xi, xlab in zip(X_VALUES, X_LABELS):
        for yi, ylab in zip(Y_VALUES, Y_LABELS):
            for zi, zlab in zip(Z_VALUES, Z_LABELS):
                # ball-removed background per camera (ball far away)
                hide_ball(world)
                bg_l = render(renderer, world, "eye_left")
                bg_r = render(renderer, world, "eye_right")
                # ball at target
                place_ball(world, xi, yi, zi)
                img_l = render(renderer, world, "eye_left")
                img_r = render(renderer, world, "eye_right")

                sl = ball_image_stats(img_l, bg_l)
                sr = ball_image_stats(img_r, bg_r)
                # receptor landing uses each eye's OWN camera image per population
                lum_l = retinal_samples(img_l, uv)
                base_l = retinal_samples(bg_l, uv)
                lum_r = retinal_samples(img_r, uv)
                base_r = retinal_samples(bg_r, uv)
                dl = np.maximum(lum_l - base_l, 0.0)
                dr = np.maximum(lum_r - base_r, 0.0)

                def pop_stats(delta, mask):
                    d = delta[mask]
                    tot = float(d.sum())
                    uvc = (np.average(uv[mask], axis=0, weights=d + 1e-9).tolist()
                           if tot > 1e-6 else None)
                    return dict(delta_sum=round(tot, 5),
                                n_active=int((d > 1e-3).sum()),
                                uv_centroid=([round(c, 4) for c in uvc]
                                             if uvc else None))

                entry = dict(
                    world=dict(x=xi, y=yi, z=zi, xlab=xlab, ylab=ylab, zlab=zlab),
                    left_cam=sl, right_cam=sr,
                    left_receptors=pop_stats(dl, left_mask),   # left eye -> left cam
                    right_receptors=pop_stats(dr, right_mask), # right eye -> right cam
                )
                rows.append(entry)
                total += 1
                left_visible += int(sl["visible"])
                right_visible += int(sr["visible"])
                both_visible += int(sl["visible"] and sr["visible"])

    summary = dict(
        n_positions=total,
        left_cam_visible=left_visible,
        right_cam_visible=right_visible,
        both_cam_visible=both_visible,
        left_only=left_visible - both_visible,
        right_only=right_visible - both_visible,
        neither=total - left_visible - right_visible + both_visible,
    )

    out = dict(
        purpose="binocular visibility + camera geometry + retinotopy calibration",
        arena=dict(GOAL_LINE_X=GOAL_LINE_X, GOAL_HALF_WIDTH=GOAL_HALF_WIDTH,
                   GOAL_HEIGHT=GOAL_HEIGHT, SHOT_X=SHOT_X, BALL_RADIUS=BALL_RADIUS),
        fly_pose="standing goal line xy=(0,0) yaw=0 facing +x (incoming ball)",
        grid=dict(x=dict(labels=X_LABELS, values=X_VALUES),
                  y=dict(labels=Y_LABELS, values=Y_VALUES, note="+y = world-left"),
                  z=dict(labels=Z_LABELS, values=Z_VALUES)),
        uv_projection=("left uv_x=0.60*z; right uv_x=0.40+0.60*(1-z) "
                       "-> right viewport horizontally MIRRORED vs left "
                       "(doom/prepare.py, overlapping experimental viewports)"),
        camera_geometry=cam_geom,
        visibility_summary=summary,
        positions=rows,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "binocular_calibration.json").write_text(json.dumps(out, indent=2))
    return out


def _print_tables(out):
    print("=" * 78)
    print("PHASE 2  CAMERA GEOMETRY (loaded model + live world pose)")
    print("=" * 78)
    cg = out["camera_geometry"]
    for name in ("eye_left", "eye_right"):
        c = cg[name]
        print(f"\n{name}: body={c['attached_body']} fovy={c['fovy_deg']}")
        print(f"  local pos  {c['local_pos']}  local quat {c['local_quat']}")
        print(f"  world pos  {c['world_pos']}")
        print(f"  optical axis (world) {c['world_optical_axis']}")
        print(f"  image-right  {c['world_image_right_axis']}")
    d = cg["_derived"]
    print(f"\n  left  optical axis vs +x : {d['left_axis_to_x_deg']} deg")
    print(f"  right optical axis vs +x : {d['right_axis_to_x_deg']} deg")
    print(f"  interocular axis angle   : {d['interocular_axis_angle_deg']} deg")

    print("\n" + "=" * 78)
    print("PHASE 1  BALL VISIBILITY GRID  (per world position)")
    print("=" * 78)
    print(f"{'x':>4} {'y-lab':>8} {'z-lab':>5} | "
          f"{'L vis':>5} {'L cx':>6} {'L area':>6} | "
          f"{'R vis':>5} {'R cx':>6} {'R area':>6} | "
          f"{'Lrec Σ':>7} {'Rrec Σ':>7}")
    print("-" * 78)
    for r in out["positions"]:
        w = r["world"]
        lc, rc = r["left_cam"], r["right_cam"]
        lr, rr = r["left_receptors"], r["right_receptors"]
        print(f"{w['x']:>4} {w['ylab']:>8} {w['zlab']:>5} | "
              f"{str(lc['visible']):>5} {str(lc['cx']):>6} {lc['area_px']:>6} | "
              f"{str(rc['visible']):>5} {str(rc['cx']):>6} {rc['area_px']:>6} | "
              f"{lr['delta_sum']:>7} {rr['delta_sum']:>7}")
    s = out["visibility_summary"]
    print("-" * 78)
    print(f"positions={s['n_positions']}  left_cam_visible={s['left_cam_visible']}"
          f"  right_cam_visible={s['right_cam_visible']}"
          f"  both={s['both_cam_visible']}  neither={s['neither']}")
    print(f"left_only={s['left_only']}  right_only={s['right_only']}")


if __name__ == "__main__":
    out = run()
    _print_tables(out)
    print(f"\nsaved {OUT/'binocular_calibration.json'}")
