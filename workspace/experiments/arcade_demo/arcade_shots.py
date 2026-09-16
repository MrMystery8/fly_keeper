"""Arcade shot generation + human aim (lateral + vertical) and the 2-axis teacher.

Human input:
    aim_x in [-1, +1]  : +1 = full left (+y), -1 = full right (-y)
    aim_y in [ 0, +1]  : 0 = low rolling ball, 1 = lofted near the crossbar
    power in [ 0, +1]  : shot speed

The arcade world (`ArcadeGoalkeeperWorld`) fires low->lofted shots; the vertical
aim chooses the shot's target height at the goal line. The `group` label is
derived from aim_x sign for scoring only and is never fed to the controller.

The 2-axis TEACHER (for training the arcade bridge) uses privileged ball state
to produce the ideal (lateral, vertical) command: strafe toward the ball's
predicted goal-line crossing y, and lift proportional to how high the ball will
arrive. Teacher = labels only; removed from the controller at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from embodiment.mujoco_world import (ShotSpec, GOAL_HALF_WIDTH, GOAL_HEIGHT,
                                     GOAL_LINE_X, BALL_RADIUS)

AIM_MAX = 0.9              # lateral aim extent (cm); arcade allows wider than 0.67
SPEED_MIN, SPEED_MAX = 4.5, 6.0
CENTER_BAND = 0.18


def group_from_aim(y_target: float) -> str:
    if y_target > CENTER_BAND:
        return "left"
    if y_target < -CENTER_BAND:
        return "right"
    return "center"


def shot_from_human(aim_x: float, aim_y: float, power: float) -> ShotSpec:
    """Map human lateral aim, vertical aim, and power to an arcade ShotSpec."""
    aim_x = float(np.clip(aim_x, -1.0, 1.0))
    aim_y = float(np.clip(aim_y, 0.0, 1.0))
    power = float(np.clip(power, 0.0, 1.0))
    y_target = float(np.clip(aim_x * AIM_MAX, -(GOAL_HALF_WIDTH - 0.3),
                             GOAL_HALF_WIDTH - 0.3))
    speed = SPEED_MIN + power * (SPEED_MAX - SPEED_MIN)
    target_h = BALL_RADIUS + aim_y * (GOAL_HEIGHT - BALL_RADIUS - 0.1)
    return ShotSpec(group_from_aim(y_target), y_target, float(speed), float(target_h))


def aim_from_shot(shot: ShotSpec):
    """Inverse: recover (aim_x, aim_y, power) for display."""
    aim_x = float(np.clip(shot.y_target / AIM_MAX, -1.0, 1.0))
    denom = max(GOAL_HEIGHT - BALL_RADIUS - 0.1, 1e-6)
    aim_y = float(np.clip((shot.height - BALL_RADIUS) / denom, 0.0, 1.0))
    power = float(np.clip((shot.speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN),
                          0.0, 1.0))
    return aim_x, aim_y, power


def teacher_command(world):
    """Privileged 2-axis teacher command (labels only): (u_lateral, u_vertical).

    Predicts where and how high the ball crosses the goal line and returns the
    ideal command in the arcade convention:
      u_lateral in [-1,1]: <0 strafe left (+y), >0 strafe right (-y)
      u_vertical in [0,1]: how much to fly up to meet a high ball
    """
    ball = world._observe_ball()
    bx, by, bz = ball["pos"]
    vx, vy, vz = ball["vel"]
    # Predict the goal-line crossing under the SAME physics the arcade world
    # actually simulates. For a lofted shot the world cancels gravity on the
    # ball (straight-line rise at constant vz, see ArcadeGoalkeeperWorld.step),
    # so the correct model is z_cross = bz + vz*t -- NOT a ballistic arc. Using
    # -981 cm/s^2 here (as the frozen teacher did) drives z_cross hugely
    # negative for a lofted ball, collapsing u_vert to ~0; that only appeared to
    # work in v1 because the stale post-resolution frames (ball actually falling
    # after the loft ended) supplied the vertical signal. This matches the
    # oracle's loft model exactly.
    lofting = bool(getattr(world, "_ball_no_gravity", False))
    g = 0.0 if lofting else float(world.model.opt.gravity[2])
    if vx < -1e-3:
        t = (GOAL_LINE_X - bx) / vx
        y_cross = by + vy * t
        z_cross = bz + vz * t + 0.5 * g * t * t
    else:
        y_cross, z_cross = by, bz
    fly = world.fly.position
    fly_y = float(fly[1])
    # lateral: same convention as the frozen teacher (u = -lateral, u<0 = left)
    lateral = float(np.clip(2.5 * (y_cross - fly_y), -1, 1))
    u_lat = -lateral
    # vertical: lift proportional to how high the ball will be, above a small
    # threshold (no need to fly for low balls). Normalized by crossbar height.
    stand_z = float(getattr(world.fly, "_stand_z", None) or fly[2])
    rise_needed = max(0.0, z_cross - (stand_z + 0.25))
    u_vert = float(np.clip(rise_needed / max(GOAL_HEIGHT - 0.25, 1e-6), 0.0, 1.0))
    return (u_lat, u_vert), dict(y_cross=float(y_cross), z_cross=float(z_cross))


def science_demo_shots(n_per_cell=2, base_seed=90000):
    """Deterministic arcade shot set spanning low/high x left/center/right."""
    from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
    groups = ("left", "center", "right")
    heights = (0.0, 0.85)          # low and lofted
    shots = []
    s = base_seed
    for g in groups:
        for h in heights:
            for _ in range(n_per_cell):
                w = ArcadeGoalkeeperWorld(seed=s)
                shots.append((s, g, h, w.sample_shot(g, height_frac=h)))
                s += 1
    return shots
