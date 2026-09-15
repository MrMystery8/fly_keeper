"""Controllers for the embodied goalkeeper.

Three controllers, all emitting the same high-level command dict
{forward, turn, gait_on, move} consumed by the CPG locomotion layer:

- MaleCNSController : vision -> MaleCNS -> descending-neuron decoder. The brain
  receives ONLY the rendered image (via VisionBridge). No game state reaches it.

- RandomController  : random locomotion commands. Establishes chance level.

- HeuristicController : uses the TRUE ball state (position/velocity) to steer
  toward the ball's predicted goal-line crossing. This is an explicit "cheat"
  baseline that measures what the body + environment are MECHANICALLY capable
  of. It never touches the brain; MaleCNS must not use its logic.
"""
from __future__ import annotations
import numpy as np


class MaleCNSController:
    """Closed-loop neural controller. Only pixels enter the brain."""

    def __init__(self, brain, vision_bridge, decoder, neural_window_ms=20.0):
        self.brain = brain
        self.vision = vision_bridge
        self.decoder = decoder
        self.window = neural_window_ms

    def reset(self):
        self.decoder.reset()

    def act(self, world):
        # 1. Render body-relative vision and drive the retina.
        luminance, image = self.vision.perceive()
        # 2. Advance the real MaleCNS dynamics for one decision window.
        step_info = self.brain.step(self.window)
        # 3. Read descending-neuron activity and decode a command.
        activity = self.brain.read(self.decoder.readout_ids())
        command, diag = self.decoder.decode(activity)
        diag["retinal_mean"] = float(luminance.mean())
        diag["spikes"] = step_info["spikes"]
        return command, diag


class PassiveController:
    """Never moves. The true stand-still baseline.

    If MaleCNS matches this (0% left, 100% center, 0% right), then the neural
    controller is effectively equivalent to standing still.
    """

    def reset(self):
        pass

    def act(self, world):
        command = {"forward": 0.0, "lateral": 0.0, "turn": 0.0,
                   "gait_on": 0.0, "move": "STAY"}
        return command, {"move": "STAY"}


class RandomController:
    """Random high-level locomotion commands."""

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, world):
        move = str(self.rng.choice(["LEFT", "STAY", "RIGHT"]))
        # LEFT strafes toward body +y (lateral>0), RIGHT toward -y.
        lateral = {"LEFT": 1.0, "STAY": 0.0, "RIGHT": -1.0}[move]
        gait_on = 1.0 if move != "STAY" else 0.0
        command = {"forward": 0.0, "lateral": lateral, "turn": 0.0,
                   "gait_on": gait_on, "move": move}
        return command, {"move": move, "lateral": lateral}


class HeuristicController:
    """Cheat baseline: steer toward the ball using TRUE ball state.

    Establishes the mechanical ceiling of the body+environment. Does NOT use the
    brain. The command still flows through the same CPG, so it is a fair test of
    what the locomotion layer can achieve.
    """

    def __init__(self, goal_line_x, turn_gain=2.5):
        self.goal_line_x = goal_line_x
        self.turn_gain = turn_gain

    def reset(self):
        pass

    def act(self, world):
        ball = world._observe_ball()
        bx, by, bz = ball["pos"]
        vx, vy, vz = ball["vel"]
        # Predict lateral crossing y at the goal-line plane.
        if vx < -1e-3:
            t = (self.goal_line_x - bx) / vx
            y_cross = by + vy * t
        else:
            y_cross = by
        fly_y = world.fly.position[1]
        # The fly faces +x; its +y axis is its LEFT. Strafe toward y_cross.
        err = y_cross - fly_y
        # lateral > 0 moves toward +y, so lateral tracks +err directly.
        lateral = float(np.clip(self.turn_gain * err, -1, 1))
        move = "LEFT" if err > 0.03 else "RIGHT" if err < -0.03 else "STAY"
        gait_on = 1.0 if abs(err) > 0.02 else 0.0
        command = {"forward": 0.0, "lateral": lateral, "turn": 0.0,
                   "gait_on": gait_on, "move": move}
        return command, {"move": move, "lateral": lateral, "y_cross": float(y_cross),
                         "fly_y": float(fly_y)}


def build_controller(name, *, brain=None, vision_bridge=None, decoder=None,
                     goal_line_x=-0.3, seed=7):
    if name == "malecns":
        return MaleCNSController(brain, vision_bridge, decoder)
    if name == "random":
        return RandomController(seed)
    if name == "heuristic":
        return HeuristicController(goal_line_x)
    if name == "passive":
        return PassiveController()
    raise ValueError(f"unknown controller {name!r}")
