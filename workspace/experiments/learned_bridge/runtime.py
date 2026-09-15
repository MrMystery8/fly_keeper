"""Closed-loop runtime for the learned-bridge goalkeeper (bridge ON/OFF).

The controller runs the EXACT Natural-MaleCNS closed loop -- render vision, step
the fixed brain 20 ms, read the descending neurons, decode a lateral command via
the existing DescendingMotorDecoder -- and, when the bridge is ENABLED, ADDS a
bounded external current to the real DNs via the fixed motor basis before the
next brain step. Bridge OFF injects nothing, so it reproduces Natural MaleCNS
bit-for-bit given the same seeds/state.

Ordering per 20 ms decision (t):
    1. vision.perceive()                      # queue retinal luminance for step
    2. (bridge ON) basis.inject(u_{t-1})      # queued additive DN current
    3. brain.step(20 ms)                       # fixed LIF dynamics advance
    4. bridge.observe(brain)                   # record this window's OL features
    5. read DNs -> decoder.decode -> command   # existing motor readout
    6. u_t = bridge.command()                  # scalar for the NEXT step's inject
    7. world.step(command)                     # body + ball physics

The bridge current from decision t-1 is applied at the brain step of decision t;
this one-window lag matches the additive-current-consumed-by-next-step semantics
of the adapter and keeps the injection causal.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN

DECISION_MS = 20.0
DECISION_S = DECISION_MS / 1000.0
MAX_DECISIONS = 75


class BridgeController:
    """Vision -> fixed MaleCNS -> (learned bridge current) -> real DNs -> decoder.

    bridge: a LearnedBridge (or None for pure Natural MaleCNS). `bridge.enabled`
    toggles ON/OFF. The decoder reads the SAME DNs used as the motor basis so the
    injected current and the readout are on the same real neurons.
    """

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
        # 1. queue vision
        luminance, _ = self.vision.perceive()
        # 2. queue the previous decision's bridge current (bridge ON only)
        inj_ids, inj_vals = ([], [])
        if self.bridge is not None:
            inj_ids, inj_vals = self.bridge.inject(self.brain)  # uses last_u
        # 3. advance fixed dynamics
        step_info = self.brain.step(DECISION_MS)
        # 4. record optic-lobe features from this window
        if self.bridge is not None:
            self.bridge.observe(self.brain)
        # 5. read DNs and decode command
        activity = self.brain.read(self.decoder.readout_ids())
        command, diag = self.decoder.decode(activity)
        # 6. compute the scalar command for the NEXT step's injection
        u = self.bridge.command() if self.bridge is not None else 0.0
        diag["bridge_u"] = float(u)
        diag["bridge_on"] = bool(self.bridge is not None and self.bridge.enabled)
        diag["retinal_mean"] = float(luminance.mean())
        diag["spikes"] = step_info["spikes"]
        diag["inj_total_mv"] = float(sum(inj_vals))
        return command, diag


def make_world_brain_vision(seed, condition="normal", camera="eye_left",
                            decoder=None):
    """Construct the standard closed-loop stack (fresh brain each call)."""
    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=seed)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera=camera, condition=condition)
    if decoder is None:
        # Use the SAME DNs as the motor basis for the readout so injection and
        # readout live on the same real neurons (opponent L/R pairs).
        decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN),
                                         right_ids=list(RIGHT_DN),
                                         turn_gain=2.5, forward_bias=0.0,
                                         smoothing=0.5)
    return world, brain, vision, decoder


def run_episode(world, controller, shot, *, condition="normal",
                collect_trace=False):
    """Run one penalty episode; return outcome dict (+ optional per-step trace)."""
    world.reset(shot)
    controller.reset()
    if condition == "static_ball" and controller.vision is not None:
        controller.vision.set_static_reference(controller.vision.render())
    trace = []
    n = 0
    while world.result is None and n < MAX_DECISIONS:
        command, diag = controller.act(world)
        world.fly.set_command(command["forward"], command.get("turn", 0.0),
                              command["gait_on"], lateral=command.get("lateral", 0.0))
        world.step(DECISION_S)
        if collect_trace:
            ball = world._observe_ball()
            trace.append(dict(
                step=n, group=shot.group,
                fly_y=round(float(world.fly.position[1]), 4),
                ball_x=round(float(ball["pos"][0]), 4),
                ball_y=round(float(ball["pos"][1]), 4),
                lateral=round(float(command.get("lateral", 0.0)), 4),
                move=command.get("move"),
                bridge_u=round(float(diag.get("bridge_u", 0.0)), 4),
                left_spikes=diag.get("left_spikes"),
                right_spikes=diag.get("right_spikes"),
                inj_total_mv=round(float(diag.get("inj_total_mv", 0.0)), 2),
            ))
        n += 1
    result = world.result or "GOAL"
    return dict(group=shot.group, result=result, decisions=n,
                final_fly_y=round(float(world.fly.position[1]), 4),
                contact=world.contact), trace
