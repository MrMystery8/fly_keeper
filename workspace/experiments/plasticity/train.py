"""Targeted plasticity training harness + bootstrap diagnostics.

Runs embodied goalkeeper episodes with the CPU MaleCNS brain, accumulates the
subthreshold eligibility trace on the masked pathway during each episode, and
applies a reward-modulated weight update at episode end (SAVE +1 / GOAL -1).

Guards enforced here:
- LEFT/RIGHT-ONLY early curriculum (no center shots) so the fly cannot be
  rewarded for standing still and blocking centre shots.
- Reward is task outcome ONLY; no ball state / direction / target is injected.
- Optional small MOTOR-level exploration during training (not direction-informed).
- CPU reference backend only.
- The connectome file on disk is never modified.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from experiments.plasticity.engine import PlasticPathway

# Target descending opponent populations (from the mask).
LEFT_DN = [10162, 10527, 10259]     # DNp20_L, DNpe017_L, DNp11_L
RIGHT_DN = [10059, 555871, 10106]   # DNp20_R, DNpe017_R, DNp11_R
DECISION_MS = 20.0
DECISION_S = DECISION_MS / 1000.0
MAX_DECISIONS = 75


class PathwayDecoder:
    """Interpretable readout of the target DN opponent populations -> lateral
    strafe. Baseline-subtracted so a constant firing bias does not drive motion;
    only the vision-driven left/right difference produces a command. No ball
    state enters. This is the controller both during training and evaluation."""

    def __init__(self, left_ids=LEFT_DN, right_ids=RIGHT_DN, gain=0.25,
                 deadband=0.5, smoothing=0.5, baseline_tau=0.98):
        self.left_ids = left_ids; self.right_ids = right_ids
        self.gain = gain; self.deadband = deadband; self.smoothing = smoothing
        self.baseline_tau = baseline_tau
        self._lat = 0.0; self._baseline = 0.0

    def readout_ids(self):
        return list(dict.fromkeys(self.left_ids + self.right_ids))

    def reset(self):
        self._lat = 0.0; self._baseline = 0.0

    def decode(self, activity):
        left = float(sum(activity[i]["spikes"] for i in self.left_ids))
        right = float(sum(activity[i]["spikes"] for i in self.right_ids))
        diff = right - left
        self._baseline = self.baseline_tau * self._baseline + (1 - self.baseline_tau) * diff
        centered = diff - self._baseline
        # right DNs win -> strafe right (lateral < 0); left DNs win -> +y
        raw = 0.0
        if abs(centered) > self.deadband:
            raw = float(np.clip(-self.gain * centered, -1, 1))
        self._lat = (1 - self.smoothing) * self._lat + self.smoothing * raw
        move = "LEFT" if self._lat > 0.05 else "RIGHT" if self._lat < -0.05 else "STAY"
        gait = 1.0 if abs(self._lat) > 1e-3 else 0.0
        return ({"forward": 0.0, "lateral": self._lat, "turn": 0.0,
                 "gait_on": gait, "move": move},
                {"left": left, "right": right, "centered": centered})


def run_episode(world, brain, vision, decoder, pathway, shot, *, learn=True,
                explore=0.0, rng=None, plasticity_on=True, explore_tau=0.85,
                credit="scalar"):
    """One episode. Returns (result, diagnostics). If learn, accumulate
    eligibility each window and apply reward at the end.

    Exploration is TEMPORALLY CORRELATED (Ornstein-Uhlenbeck-like): a slowly
    drifting lateral bias per episode, so the fly commits to a direction long
    enough to actually move, instead of white noise that cancels out. The
    exploration direction is random - it never uses ball position/direction."""
    world.reset(shot)
    decoder.reset()
    if pathway is not None:
        pathway.clear_eligibility()
    n = 0
    moves = []
    # Per-episode COMMITTED exploration: with probability `explore`, pick a
    # random left/right strafe direction and hold it for the whole episode
    # (magnitude drawn once). This guarantees the fly actually travels far
    # enough to sometimes make a save, giving a usable positive reward signal.
    # The direction is random - it never uses ball position or shot side.
    explore_dir = 0.0
    if explore > 0 and rng is not None and rng.random() < explore:
        explore_dir = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0))
    while world.result is None and n < MAX_DECISIONS:
        vision.perceive()
        brain.step(DECISION_MS)
        if pathway is not None and plasticity_on:
            pathway.accumulate()
        act = brain.read(decoder.readout_ids())
        cmd, diag = decoder.decode(act)
        lateral = cmd["lateral"]
        if explore_dir != 0.0:
            # blend the committed exploratory strafe with the decoder output
            lateral = float(np.clip(0.7 * explore_dir + 0.3 * lateral, -1, 1))
        world.fly.set_command(0.0, 0.0, 1.0 if abs(lateral) > 1e-3 else 0.0, lateral=lateral)
        world.step(DECISION_S)
        moves.append(cmd["move"])
        n += 1
    result = world.result or "GOAL"
    reward = +1.0 if result == "SAVE" else -1.0
    if learn and pathway is not None:
        if credit == "directional" and explore_dir != 0.0:
            # action_side: +1 if the fly committed leftward (+y), -1 if rightward
            pathway.apply_reward_directional(reward, np.sign(explore_dir))
        else:
            pathway.apply_reward(reward)
    from collections import Counter
    return result, dict(reward=reward, decisions=n, moves=dict(Counter(moves)),
                        final_fly_y=round(float(world.fly.position[1]), 3))


def make_lr_shot(world, side, rng, aim_mag=None, speed_range=None):
    """A balanced left/right shot. `aim_mag` overrides the lateral aim magnitude
    (curriculum: start easier/closer to centre, then widen). `speed_range`
    overrides shot speed. No centre shots are ever produced here."""
    from embodiment.mujoco_world import ShotSpec, BALL_RADIUS
    if aim_mag is None:
        return world.sample_shot(side)
    sign = +1.0 if side == "left" else -1.0
    aim = sign * aim_mag + float(rng.uniform(-0.1, 0.1))
    lo, hi = speed_range or (4.5, 6.0)
    speed = float(rng.uniform(lo, hi))
    return ShotSpec(side, aim, speed, BALL_RADIUS)


def bootstrap_check(episodes=8, seed=1, elig_mode="subthreshold"):
    """Measure whether the rule produces meaningful, bounded updates before any
    long training. Prints eligible synapses, updates, |dw|, VP/DN activity."""
    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=seed)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left")
    decoder = PathwayDecoder()
    pathway = PlasticPathway(brain, lr=0.02, elig_mode=elig_mode)
    rng = np.random.default_rng(seed)

    w_before = pathway.current_weights()
    dn_all = LEFT_DN + RIGHT_DN
    dn_spikes_before = []
    results = []
    per_ep = []
    for ep in range(episodes):
        side = "left" if ep % 2 == 0 else "right"
        shot = make_lr_shot(world, side, rng)
        # measure eligibility BEFORE reward for this episode
        res, diag = run_episode(world, brain, vision, decoder, pathway, shot,
                                learn=True, explore=0.1, rng=rng)
        results.append(res)
        per_ep.append(dict(ep=ep, side=side, result=res,
                           eligible=pathway.eligible_count(),
                           moves=diag["moves"]))
    w_after = pathway.current_weights()
    report = pathway.weight_report()
    changed = int(np.count_nonzero(np.abs(w_after - w_before) > 1e-9))
    out = dict(elig_mode=elig_mode, episodes=episodes,
               results=results,
               weights_changed=changed,
               total_abs_dw=report["total_abs_dw"],
               n_updates=report["n_updates"],
               weight_report=report,
               per_episode=per_ep)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--elig", choices=["subthreshold", "spike"], default="subthreshold")
    p.add_argument("--episodes", type=int, default=8)
    a = p.parse_args()
    if a.bootstrap:
        out = bootstrap_check(episodes=a.episodes, elig_mode=a.elig)
        print(json.dumps(out, indent=2))
