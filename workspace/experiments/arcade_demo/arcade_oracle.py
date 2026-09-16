"""Arcade oracle (heuristic) controller + physical-ceiling evaluation.

The oracle reads TRUE ball state (privileged) to compute the ideal interception
and drives the SAME arcade flight motor layer as the neural controller via
continuous (u_lat, u_vert). It measures the PHYSICAL ceiling of the arcade
locomotion: can the fly physically reach the intended arcade shot distribution
(±~0.9 cm lateral, low->crossbar high) within the ball flight time?

The oracle never touches the brain; it exists to validate/tune the motor layer
BEFORE the neural bridge is evaluated. If the oracle cannot reach shots, the fix
is locomotion/arena timing, not the bridge.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GOAL_LINE_X, GOAL_HALF_WIDTH
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_S, MAX_DECISIONS

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


class ArcadeOracleController:
    """Privileged interception target -> continuous (u_lat, u_vert) -> arcade motor.

    Estimates the ball's goal-line crossing (t_cross, y_cross, z_cross) with a
    SMOOTHED, latched target so it does not chase tiny per-step changes, and
    commands the flight controller toward that goalkeeper target. Once the
    keeper has intercepted the ball for this shot, it STOPS using the pre-save
    target (so it never chases its own deflection) and simply holds position.
    """

    BALL_BLOCK = 0.55          # standing fly covers up to ~here (cm)

    def __init__(self, lat_gain=3.0):
        self.lat_gain = float(lat_gain)
        self._y_target = 0.0
        self._z_target = 0.0
        self._have_target = False

    def reset(self):
        self._y_target = 0.0
        self._z_target = 0.0
        self._have_target = False

    def act(self, world):
        # Once intercepted, hold: do not chase the deflected ball.
        if getattr(world, "keeper_contact", False):
            return ({"forward": 0.0, "lateral": 0.0, "turn": 0.0,
                     "gait_on": 1.0, "vertical": 0.0, "move": "STAY"},
                    {"phase": "post_intercept"})

        ball = world._observe_ball()
        bx, by, bz = ball["pos"]
        vx, vy, vz = ball["vel"]
        if vx < -1e-3:
            t = (GOAL_LINE_X - bx) / vx
            y_cross = by + vy * t
            z_cross = bz + vz * t          # arcade loft holds vz ~ const
        else:
            y_cross, z_cross = by, bz
        # smoothed/latched target (EMA) so we track a stable goalkeeper spot
        if not self._have_target:
            self._y_target, self._z_target = y_cross, z_cross
            self._have_target = True
        else:
            a = 0.3
            self._y_target = (1 - a) * self._y_target + a * y_cross
            self._z_target = (1 - a) * self._z_target + a * z_cross

        fly = world.fly.position
        fly_y = float(fly[1])
        sz = getattr(world.fly, "_stand_z", None)
        stand_z = float(sz) if sz is not None else 0.0

        # LATERAL: track the smoothed crossing, gentle deadzone near center.
        y_err = self._y_target - fly_y
        u_lat = float(np.clip(self.lat_gain * y_err, -1, 1))

        # VERTICAL: only rise for a genuinely high target; map to height frac.
        MAX_HEIGHT = float(getattr(world.fly, "MAX_HEIGHT", 1.45))
        rise = self._z_target - stand_z
        u_vert = (0.0 if rise <= self.BALL_BLOCK
                  else float(np.clip(rise / MAX_HEIGHT, 0.05, 1.0)))

        command = {"forward": 0.0, "lateral": u_lat, "turn": 0.0,
                   "gait_on": 1.0, "vertical": u_vert,
                   "move": ("LEFT" if u_lat > 0.05 else "RIGHT"
                            if u_lat < -0.05 else "STAY")}
        diag = {"y_cross": float(self._y_target), "z_cross": float(self._z_target),
                "u_lat": u_lat, "u_vert": u_vert}
        return command, diag


def run_oracle_episode(world, ctrl, shot):
    world.reset(shot)
    ctrl.reset()
    n = 0
    while world.result is None and n < MAX_DECISIONS:
        cmd, _ = ctrl.act(world)
        world.fly.set_command(cmd["forward"], cmd.get("turn", 0.0),
                              cmd["gait_on"], lateral=cmd.get("lateral", 0.0),
                              vertical=cmd.get("vertical", 0.0))
        world.step(DECISION_S)
        n += 1
    return world.result or "GOAL"


def evaluate_ceiling(per_cell=6, base_seed=90000, lat_gain=3.0,
                     heights=(0.0, 0.5, 0.85)):
    ctrl = ArcadeOracleController(lat_gain=lat_gain)
    groups = ("left", "center", "right")
    res = defaultdict(lambda: [0, 0])
    seed = base_seed
    for g in groups:
        for hf in heights:
            for _ in range(per_cell):
                w = ArcadeGoalkeeperWorld(seed=seed)
                shot = w.sample_shot(g, height_frac=hf)
                r = run_oracle_episode(w, ctrl, shot)
                key = f"{g}/{'low' if hf < 0.34 else 'mid' if hf < 0.67 else 'high'}"
                res[key][0] += int(r == "SAVE")
                res[key][1] += 1
                seed += 1
    tot_s = sum(v[0] for v in res.values())
    tot_n = sum(v[1] for v in res.values())
    report = dict(overall=round(tot_s / tot_n, 3), saves=tot_s, n=tot_n,
                  lat_gain=lat_gain,
                  by_cell={k: f"{v[0]}/{v[1]}" for k, v in sorted(res.items())})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--lat-gain", type=float, default=3.0)
    a = p.parse_args()
    rep = evaluate_ceiling(per_cell=a.per_cell, base_seed=a.base_seed,
                           lat_gain=a.lat_gain)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle_ceiling.json").write_text(json.dumps(rep, indent=2))
    print(f"ARCADE ORACLE PHYSICAL CEILING: {rep['overall']:.1%} "
          f"({rep['saves']}/{rep['n']})")
    for k, v in rep["by_cell"].items():
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
