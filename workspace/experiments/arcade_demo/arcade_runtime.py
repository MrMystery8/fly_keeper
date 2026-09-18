"""Closed-loop arcade runtime: vision -> MaleCNS -> 2-axis bridge -> arcade body.

Mirrors the frozen BridgeController loop but drives TWO body axes (lateral strafe
+ vertical hop/flight) from real descending neurons, and uses the arcade body.

Per 20 ms decision (t):
    1. vision.perceive()                       # queue retina (pixels only)
    2. bridge.inject(brain)  (u from t-1)      # bounded current on real DNs
    3. brain.step(20 ms)                        # fixed MaleCNS dynamics (Metal)
    4. bridge.observe(brain)                    # record OL features
    5. read DNs -> arcade decoder -> (lateral, vertical) body command
    6. u_t = bridge.command()                   # (u_lat, u_vert) for next inject
    7. body gets set_command(lateral=, vertical=)

The controller sees ONLY rendered pixels; ball state never reaches it.

The arcade decoder reads the SAME real DNs the basis drives:
  * lateral: DNp20/DNpe017 L vs R opponent asymmetry -> strafe (as frozen)
  * vertical: DNp01 pair activation -> lift command (hop/flight)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.arcade_demo.vertical_dn import (ArcadeDNBasis, VERTICAL_DN,
                                                 LATERAL_LEFT_DN,
                                                 LATERAL_RIGHT_DN)

DECISION_MS = 20.0
DECISION_S = DECISION_MS / 1000.0
MAX_DECISIONS = 90              # a bit longer than frozen (lofted shots hang up)


class Arcade2AxisDecoder:
    """Reads real DNs -> (lateral, vertical) body command for the arcade body.

    Lateral: normalized L/R opponent asymmetry (same idea as the frozen decoder;
    asym>0 == right DNs dominate == strafe right == lateral<0).
    Vertical: normalized activation of the vertical DN pair -> lift in [0,1].
    Both are engineered mappings from real DN spikes; no ball state.
    """

    def __init__(self, left_ids=LATERAL_LEFT_DN, right_ids=LATERAL_RIGHT_DN,
                 vertical_ids=VERTICAL_DN, turn_gain=2.5, lift_gain=1.5,
                 smoothing=0.5, vert_ref=4.0):
        self.left_ids = tuple(left_ids)
        self.right_ids = tuple(right_ids)
        self.vertical_ids = tuple(vertical_ids)
        self.turn_gain = float(turn_gain)
        self.lift_gain = float(lift_gain)
        self.smoothing = float(smoothing)
        self.vert_ref = float(vert_ref)     # spike count that maps to full lift
        self._lat = 0.0
        self._vert = 0.0

    def readout_ids(self):
        return list(dict.fromkeys(self.left_ids + self.right_ids
                                  + self.vertical_ids))

    def decode(self, activity):
        left = float(sum(activity[i]["spikes"] for i in self.left_ids))
        right = float(sum(activity[i]["spikes"] for i in self.right_ids))
        vert = float(sum(activity[i]["spikes"] for i in self.vertical_ids))
        total = left + right
        asym = 0.0 if total <= 0 else (right - left) / total
        lat_raw = float(np.clip(-self.turn_gain * asym, -1, 1))
        vert_raw = float(np.clip(self.lift_gain * vert / self.vert_ref, 0, 1))
        self._lat = (1 - self.smoothing) * self._lat + self.smoothing * lat_raw
        self._vert = (1 - self.smoothing) * self._vert + self.smoothing * vert_raw
        move = ("LEFT" if left > right else "RIGHT" if right > left else "STAY")
        gait_on = 1.0 if (abs(self._lat) > 1e-3 or self._vert > 1e-3) else 0.0
        command = {"forward": 0.0, "lateral": self._lat, "turn": 0.0,
                   "gait_on": gait_on, "vertical": self._vert, "move": move}
        diag = {"left_spikes": left, "right_spikes": right, "vert_spikes": vert,
                "asymmetry": asym, "lateral": self._lat, "vertical": self._vert}
        return command, diag

    def reset(self):
        self._lat = 0.0
        self._vert = 0.0


class ArcadeController:
    """vision -> fixed MaleCNS -> (2-axis learned bridge current) -> real DNs
    -> arcade decoder -> arcade body. bridge=None => natural (no bridge)."""

    def __init__(self, brain, vision, decoder, bridge=None):
        self.brain = brain
        self.vision = vision
        self.decoder = decoder
        self.bridge = bridge

    def reset(self):
        self.decoder.reset()
        if self.bridge is not None:
            self.bridge.reset()

    def act(self, world):
        luminance, _ = self.vision.perceive()
        inj_vals = []
        if self.bridge is not None:
            _, inj_vals = self.bridge.inject(self.brain)
        step_info = self.brain.step(DECISION_MS)
        if self.bridge is not None:
            self.bridge.observe(self.brain)
        activity = self.brain.read(self.decoder.readout_ids())
        command, diag = self.decoder.decode(activity)
        # Most frozen bridges need only neural history.  The new Arcade action
        # policy additionally uses *keeper proprioception* (never ball state)
        # for commitment/recovery and mid-air correction.  Keep the original
        # bridge interface intact for every historical controller.
        if self.bridge is not None:
            if getattr(self.bridge, "uses_proprioception", False):
                u_lat, u_vert = self.bridge.command(world)
            else:
                u_lat, u_vert = self.bridge.command()
        else:
            u_lat, u_vert = (0.0, 0.0)
        diag["bridge_u_lat"] = float(u_lat)
        diag["bridge_u_vert"] = float(u_vert)
        diag["bridge_on"] = bool(self.bridge is not None and self.bridge.enabled)
        diag["retinal_mean"] = float(luminance.mean())
        diag["spikes"] = step_info["spikes"]
        diag["inj_total_mv"] = float(sum(inj_vals))
        return command, diag


def run_episode(world, controller, shot, collect_trace=False):
    world.reset(shot)
    # A controller episode must start from the same neural state as collection.
    # Without this, residual MaleCNS membrane/spike state from a prior shot
    # contaminates the next supposedly independent matched-seed evaluation.
    # Oracle controllers have no MaleCNS instance; neural controllers do.
    if hasattr(controller, "brain"):
        controller.brain.reset()
    controller.reset()
    trace = []
    n = 0
    while world.result is None and n < MAX_DECISIONS:
        command, diag = controller.act(world)
        world.fly.set_command(command["forward"], command.get("turn", 0.0),
                              command["gait_on"],
                              lateral=command.get("lateral", 0.0),
                              vertical=command.get("vertical", 0.0))
        world.step(DECISION_S)
        if collect_trace:
            ball = world._observe_ball()
            trace.append(dict(
                step=n, group=shot.group,
                fly_y=round(float(world.fly.position[1]), 4),
                fly_z=round(float(world.fly.position[2]), 4),
                ball_x=round(float(ball["pos"][0]), 4),
                ball_y=round(float(ball["pos"][1]), 4),
                ball_z=round(float(ball["pos"][2]), 4),
                lateral=round(float(command.get("lateral", 0.0)), 4),
                vertical=round(float(command.get("vertical", 0.0)), 4),
                u_lat=round(float(diag.get("bridge_u_lat", 0.0)), 4),
                u_vert=round(float(diag.get("bridge_u_vert", 0.0)), 4),
                movement_state=str(getattr(world.fly, "state", "UNKNOWN")),
            ))
        n += 1
    diag_out = (world.outcome_diagnostics()
                if hasattr(world, "outcome_diagnostics") else {})
    out = dict(group=shot.group, result=world.result or "GOAL", decisions=n,
               final_fly_y=round(float(world.fly.position[1]), 4),
               final_fly_z=round(float(world.fly.position[2]), 4),
               keeper_contact=bool(getattr(world, "keeper_contact", False)),
               deflected=bool(getattr(world, "deflected", False)))
    out.update({k: diag_out.get(k) for k in
                ("touch_but_goal", "untouched_goal") if k in diag_out})
    return out, trace
