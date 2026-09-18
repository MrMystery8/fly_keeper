"""Iteration-2 bounded residual RL over the RICH DAgger policy.

a_final = clip(a_base + alpha * delta_RL(rich_obs), [-1,0], [1,1])

  a_base   : frozen rich DAgger policy (full-mode command)
  delta_RL : tiny tanh MLP over the SAME 14-dim rich observation, ES-trained
  alpha    : bounded authority (0.25-0.4) to tune WHEN/HOW-HARD, not to override

REWARD (terminal-dominant; small graded-vertical shaping to kill always-takeoff):
    SAVE                                +10
    GOAL                                -10
    meaningful keeper contact           +0.5
    useful deflection                   +0.5
    VERTICAL ALIGNMENT (approach)        +/- up to ~0.6:
        + if keeper peak height sits near the ball's required crossing height
        - GROSS vertical overshoot (flew high for a LOW ball)   [the fix for B]
    stable recovery                     +0.2
    extreme instability                 -0.5
The vertical term uses the PRIVILEGED required crossing height (z_cross) for the
REWARD ONLY; the policy never observes it. Motion is NOT penalized (we want an
energetic keeper); we only discourage jumping HIGH on LOW balls.

SELECTION requires: honest saves, movement > v2, RIGHT>0, graded vertical
(LOW takeoff < MID < HIGH), binocular dependence. Selection score = reward +
small lateral-movement bonus (active keeper not penalised).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (ArcadeController,
                                                    Arcade2AxisDecoder,
                                                    run_episode)
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_rich_encoder import RichBinocularEncoder
from experiments.arcade_demo.rich_action_policy import (OBS_DIM, RichPolicy,
                                                        RichActionBridge)
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
AIRBORNE = ("TAKEOFF", "FLIGHT", "DIVE")
HOP_STATES = ("LOW_HOP", "TAKEOFF", "FLIGHT", "DIVE")
# fly reach half-height; a ball crossing at z_cross is reachable if the keeper's
# thorax is within ~this of it. Used only to score vertical alignment (reward).
REACH_Z = 0.45


class RLHead:
    def __init__(self, w1, b1, w2, b2, mean, scale):
        self.w1, self.b1, self.w2, self.b2, self.mean, self.scale = w1, b1, w2, b2, mean, scale

    @classmethod
    def zeros(cls, n_in, hidden, mean, scale, rng):
        return cls(rng.normal(0, .1, (n_in, hidden)), np.zeros(hidden),
                   rng.normal(0, .1, (hidden, 2)), np.zeros(2), mean, scale)

    def flat(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_flat(self, v):
        i = 0
        for arr in (self.w1, self.b1, self.w2, self.b2):
            n = arr.size; arr[...] = v[i:i+n].reshape(arr.shape); i += n

    def predict(self, obs):
        x = (np.asarray(obs, float) - self.mean) / self.scale
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)


class RichRLBridge(RichActionBridge):
    """Rich DAgger base (full) + alpha * delta_RL(rich_obs), bounded."""

    def __init__(self, hybrid, encoder, base_policy, rl_head, alpha):
        super().__init__(hybrid, encoder, policy=base_policy, mode="full")
        self.rl_head = rl_head
        self.alpha = float(alpha)
        self._osc = 0
        self._last_sign = 0

    def command(self, world):
        base = np.asarray(self.hybrid.command(), float)
        obs = self.observation(world, base)
        a_base = self.policy.predict(obs)                 # full-mode base action
        delta = self.rl_head.predict(obs)
        action = np.clip(a_base + self.alpha * delta, (-1.0, 0.0), (1.0, 1.0))
        s = int(np.sign(action[0])) if abs(action[0]) > 0.1 else 0
        if s != 0 and self._last_sign != 0 and s != self._last_sign:
            self._osc += 1
        if s != 0:
            self._last_sign = s
        self.hybrid._ema = np.asarray(action, float)
        self.last_obs = obs; self.last_base = base; self.previous = action
        return float(action[0]), float(action[1])


def episode_reward(world, trace, osc, z_cross):
    d = world.outcome_diagnostics()
    r = 10.0 if world.result == "SAVE" else -10.0
    if d.get("keeper_contact"):
        r += 0.5
    if d.get("deflected_save"):
        r += 0.5
    if trace:
        z = np.array([t["fly_z"] for t in trace]); y = np.array([t["fly_y"] for t in trace])
        peak_z = float(np.abs(z - z[0]).max())
        # required keeper rise to meet the ball (privileged, reward-only)
        req = max(0.0, float(z_cross) - float(z[0]) - 0.15)
        # vertical alignment: reward peak near req, penalize GROSS overshoot.
        # overshoot beyond req+REACH_Z on a low ball is the always-jump waste.
        overshoot = max(0.0, peak_z - (req + REACH_Z))
        align = 0.3 * (1.0 - min(1.0, abs(peak_z - req) / 0.8))   # up to +0.3
        r += align - 0.6 * min(1.0, overshoot / 0.8)              # up to -0.6
        moved = (np.abs(y - y[0]).max() > 0.2) or (peak_z > 0.2)
        if moved and abs(float(z[-1]) - float(z[0])) < 0.15:
            r += 0.2
        if np.abs(y).max() > 1.4:
            r -= 0.5
    if osc >= 4 and not d.get("keeper_contact"):
        r -= 0.1
    return r


def movement(trace):
    if not trace:
        return dict(peak_lat=0., peak_vert=0., takeoff=0., hop=0.)
    y = np.array([t["fly_y"] for t in trace]); z = np.array([t["fly_z"] for t in trace])
    st = [t.get("movement_state") for t in trace]
    return dict(peak_lat=float(np.abs(y-y[0]).max()), peak_vert=float(np.abs(z-z[0]).max()),
                takeoff=float(any(s in AIRBORNE for s in st)),
                hop=float(any(s in HOP_STATES for s in st)))


def evaluate(brain, encoder, base_policy, flat_theta, tmpl, alpha, shots):
    head = RLHead(tmpl.w1.copy(), tmpl.b1.copy(), tmpl.w2.copy(), tmpl.b2.copy(),
                  tmpl.mean, tmpl.scale)
    head.set_flat(flat_theta)
    total_r = 0.0; saves = 0; mv = []; by_h = defaultdict(lambda: [0, 0, []])
    for seed, shot in shots:
        w = ArcadeGoalkeeperWorld(seed=seed)
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = RichRLBridge(hybrid, encoder, base_policy, head, alpha)
        vision = BinocularVisionBridge(w.fly, brain, condition="both", retina_map="fullframe")
        ctrl = ArcadeController(brain, vision, Arcade2AxisDecoder(), bridge)
        # privileged z_cross for reward shaping (computed once at reset)
        w.reset(shot)
        zc = float(teacher_command(w)[1].get("z_cross", shot.height))
        out, tr = run_episode(w, ctrl, shot, collect_trace=True)
        total_r += episode_reward(w, tr, bridge._osc, zc)
        saves += int(out["result"] == "SAVE"); mv.append(movement(tr))
        hb = "low" if shot.height < 0.52 else "mid" if shot.height < 0.72 else "high"
        by_h[hb][0] += int(out["result"] == "SAVE"); by_h[hb][1] += 1
        by_h[hb][2].append(movement(tr)["takeoff"])
    n = len(shots)
    stats = dict(save_rate=round(saves/n, 3),
                 peak_lat=round(float(np.mean([m["peak_lat"] for m in mv])), 4),
                 peak_vert=round(float(np.mean([m["peak_vert"] for m in mv])), 4),
                 takeoff=round(float(np.mean([m["takeoff"] for m in mv])), 3),
                 takeoff_by_height={h: round(float(np.mean(v[2])), 3) for h, v in by_h.items()})
    return total_r/n, stats


def sel_score(mean_r, stats, v2_peak_lat=0.42):
    return mean_r + 0.5 * min(1.0, stats["peak_lat"] / max(v2_peak_lat, 1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-policy", default="arcade_rich_policy_rich_r2.npz")
    p.add_argument("--encoder", default="binocular_rich_encoder_corrected.npz")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--n-shots", type=int, default=48)
    p.add_argument("--base-seed", type=int, default=160000)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--out", default="arcade_rich_residual_rl.npz")
    a = p.parse_args()

    t0 = time.perf_counter()
    shots = SV.randomized_valid_shots(a.n_shots, a.base_seed)
    print(f"[rich-rl] {len(shots)} randomized CLEAN_GOAL training shots; alpha={a.alpha}")
    encoder, _ = RichBinocularEncoder.load(a.encoder)
    base_policy, _ = RichPolicy.load(a.base_policy)
    rng = np.random.default_rng(a.seed)
    tmpl = RLHead.zeros(OBS_DIM, a.hidden, base_policy.mean, base_policy.scale, rng)
    dim = tmpl.flat().size
    mean = np.zeros(dim); sigma = 0.25; history = []
    brain = MaleCNSBrain(backend="metal")
    best = None
    try:
        base_r, base_stats = evaluate(brain, encoder, base_policy, mean, tmpl, a.alpha, shots)
        print(f"[rich-rl] base(delta=0): reward={base_r:.3f} save={base_stats['save_rate']} "
              f"peakLat={base_stats['peak_lat']} peakVert={base_stats['peak_vert']} "
              f"takeoff={base_stats['takeoff']} tk_by_h={base_stats['takeoff_by_height']}")
        history.append(dict(gen=-1, kind="base", reward=round(base_r, 3), **base_stats))
        best = (sel_score(base_r, base_stats), mean.copy(), base_r, base_stats)
        for gen in range(a.generations):
            cand = mean + rng.normal(size=(a.population, dim)) * sigma
            rewards, scores, stats_list = [], [], []
            for th in cand:
                mr, st = evaluate(brain, encoder, base_policy, th, tmpl, a.alpha, shots)
                rewards.append(mr); stats_list.append(st); scores.append(sel_score(mr, st))
            scores = np.array(scores)
            elite = np.argsort(scores)[-max(2, a.population//3):]
            mean = cand[elite].mean(0); sigma = max(0.05, sigma*0.85)
            gi = int(np.argmax(scores)); bs = stats_list[gi]
            history.append(dict(gen=gen, best_reward=round(float(max(rewards)), 3),
                                best_save=bs["save_rate"], best_peak_lat=bs["peak_lat"],
                                best_peak_vert=bs["peak_vert"], best_takeoff=bs["takeoff"],
                                best_tk_by_h=bs["takeoff_by_height"]))
            print(f"[gen {gen}] best_reward={history[-1]['best_reward']} save={bs['save_rate']} "
                  f"peakLat={bs['peak_lat']} peakVert={bs['peak_vert']} takeoff={bs['takeoff']} "
                  f"tk_by_h={bs['takeoff_by_height']} sigma={sigma:.3f}", flush=True)
            if scores.max() > best[0]:
                best = (float(scores.max()), cand[gi].copy(), rewards[gi], bs)
        final_theta = best[1]
        final_r, final_stats = evaluate(brain, encoder, base_policy, final_theta, tmpl, a.alpha, shots)
    finally:
        brain.close()
    tmpl.set_flat(final_theta)
    meta = dict(method="bounded residual ES-RL over rich DAgger policy",
                architecture=f"delta_RL {OBS_DIM}->{a.hidden}->2 tanh; a_final=clip(a_base+alpha*delta,[-1,0],[1,1])",
                alpha=a.alpha, base_policy=a.base_policy,
                reward="SAVE+10/GOAL-10; +0.5 contact/deflect; +/-0.6 vertical-alignment(anti-overshoot); +0.2 recovery; -0.5 instability; NO motion penalty",
                generations=a.generations, population=a.population, n_shots=len(shots),
                final=dict(reward=round(final_r, 3), **final_stats), history=history)
    np.savez(OUT / a.out, w1=tmpl.w1, b1=tmpl.b1, w2=tmpl.w2, b2=tmpl.b2,
             mean=tmpl.mean, scale=tmpl.scale, alpha=np.array([a.alpha]),
             base_policy=np.array([a.base_policy]), metadata=np.array([json.dumps(meta)]))
    (OUT / a.out.replace(".npz", "_train.json")).write_text(json.dumps(meta, indent=2))
    print(f"[rich-rl] final: reward={final_r:.3f} save={final_stats['save_rate']} "
          f"peakLat={final_stats['peak_lat']} peakVert={final_stats['peak_vert']} "
          f"takeoff={final_stats['takeoff']} tk_by_h={final_stats['takeoff_by_height']} "
          f"({round(time.perf_counter()-t0,1)}s)")


if __name__ == "__main__":
    main()
