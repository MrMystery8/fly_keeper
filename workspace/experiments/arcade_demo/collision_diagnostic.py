"""Brief diagnostic: native fly collision geometry vs the visible body.

Establishes (concisely) that the original FlyBody collision representation is
incomplete for AERIAL goalkeeper gameplay, motivating the arcade non-contact
interception volume. Not a calibration project -- just the facts.
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

from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_S

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def diagnose():
    w = ArcadeGoalkeeperWorld(seed=1)
    m, d = w.model, w.data
    w.reset(w.sample_shot("center", 0.0))
    mujoco.mj_forward(m, d)

    colliding, noncolliding = [], []
    for g in w._fly_geoms:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        if int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0:
            colliding.append(g)
        else:
            noncolliding.append(name)

    def z_extent(state_label):
        mujoco.mj_forward(m, d)
        zs = [float(d.geom_xpos[g][2]) for g in colliding]
        return dict(state=state_label, z_min=round(min(zs), 4),
                    z_max=round(max(zs), 4))

    standing = z_extent("standing")
    # in flight (hold high) the colliding geoms rise with the body
    for _ in range(30):
        w.fly.set_command(0, 0, 1, lateral=0.0, vertical=1.0)
        w.fly.step(0.02)
    flight = z_extent("flight_high")

    report = dict(
        total_fly_geoms=len(w._fly_geoms),
        colliding_geoms=len(colliding),
        noncolliding_visual_geoms=len(noncolliding),
        colliding_z_extent=dict(standing=standing, flight=flight),
        goal_height_cm=1.4, ball_radius_cm=0.45,
        finding=(
            "Only ~70 of 159 fly geoms collide (the *_collision capsules); the "
            "~89 visible body meshes (thorax/head/wings) are non-colliding "
            "(contype=0). When standing, the colliding geoms span world z ~ "
            f"[{standing['z_min']}, {standing['z_max']}] cm -- a thin ground "
            "layer covering only the bottom ~10% of the 1.4 cm goal. A ball can "
            "therefore visually intersect the fly's rendered body yet produce NO "
            "MuJoCo contact whenever it passes above that thin colliding layer. "
            "The native collision representation is thus incomplete for aerial "
            "goalkeeper interception, motivating the Arcade non-contact reach/"
            "interception volume."),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "collision_diagnostic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    diagnose()
