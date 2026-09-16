"""Human-aim/power -> ShotSpec mapping and deterministic science-demo shot sets.

The environment fires a penalty described by a ``ShotSpec(group, y_target,
speed, height)`` (see embodiment/mujoco_world.py). The scientific evaluation
generated shots via ``GoalkeeperWorld.sample_shot(group)``, which picks:

    aim y_target = {left:+0.55, center:0.0, right:-0.55} + U(-0.12, +0.12)
    speed        = U(4.5, 6.0) cm/s
    height       = BALL_RADIUS  (low rolling ball)

To keep the interactive shots inside the SAME physical regime the frozen system
was validated on (Section 14), human aim and power are CLAMPED to those ranges.
A continuous horizontal aim in [-1, +1] maps to y_target in the validated band,
and a power in [0, 1] maps to speed in [4.5, 6.0] cm/s. The ``group`` label is
derived from the aim sign purely for scoring/reporting; it is NOT given to any
controller.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from embodiment.mujoco_world import ShotSpec, GOAL_HALF_WIDTH, BALL_RADIUS

# Validated regime (matches GoalkeeperWorld.sample_shot ranges).
AIM_MAX = 0.67            # 0.55 nominal + 0.12 jitter; validated lateral extent
SPEED_MIN, SPEED_MAX = 4.5, 6.0
CENTER_BAND = 0.18        # |y_target| below this -> 'center' group label


def group_from_aim(y_target: float) -> str:
    if y_target > CENTER_BAND:
        return "left"     # +y is the fly's left
    if y_target < -CENTER_BAND:
        return "right"
    return "center"


def shot_from_human(aim: float, power: float) -> ShotSpec:
    """Map continuous human aim/power to a physically-clamped ShotSpec.

    aim   in [-1, +1]: +1 = full left (+y), -1 = full right (-y).
    power in [0, 1]  : 0 -> slowest validated speed, 1 -> fastest.

    Both are clamped to the validated regime so the shot stays in-distribution.
    """
    aim = float(np.clip(aim, -1.0, 1.0))
    power = float(np.clip(power, 0.0, 1.0))
    y_target = aim * AIM_MAX
    # keep within the posts with the same 0.3 cm margin reset() clips to.
    y_target = float(np.clip(y_target, -(GOAL_HALF_WIDTH - 0.3),
                             GOAL_HALF_WIDTH - 0.3))
    speed = SPEED_MIN + power * (SPEED_MAX - SPEED_MIN)
    return ShotSpec(group=group_from_aim(y_target), y_target=y_target,
                    speed=float(speed), height=BALL_RADIUS)


def aim_power_from_shot(shot: ShotSpec) -> tuple[float, float]:
    """Inverse mapping (for displaying a science-demo shot's aim/power)."""
    aim = float(np.clip(shot.y_target / AIM_MAX, -1.0, 1.0))
    power = float(np.clip((shot.speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN),
                          0.0, 1.0))
    return aim, power


def science_demo_shots(n_per_group: int = 4, base_seed: int = 90000):
    """Deterministic, reproducible shot list for repeatable mode comparisons.

    Reuses the SAME construction the frozen evaluation used
    (``GoalkeeperWorld(seed).sample_shot(group)`` per held-out seed), so a
    Science-Demo run lines up directly with the experiment's held-out shots and
    does not overwrite the benchmark numbers.

    Returns a list of ``(seed, group, ShotSpec)``.
    """
    from embodiment.mujoco_world import GoalkeeperWorld
    groups = ("left", "center", "right")
    shots = []
    s = base_seed
    for g in groups:
        for _ in range(n_per_group):
            w = GoalkeeperWorld(seed=s)
            shots.append((s, g, w.sample_shot(g)))
            s += 1
    return shots
