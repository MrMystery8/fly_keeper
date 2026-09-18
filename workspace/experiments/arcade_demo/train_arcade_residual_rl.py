"""Phase 9+10: bounded residual RL over the DAgger policy (ES, dependency-free).

Architecture (matches the mandate a_final = a_base + alpha * delta_RL):

    two retinas -> frozen MaleCNS -> L/R stream heads
      -> DAgger residual policy  ......................  a_base (frozen)
      -> RL residual MLP (same policy-visible obs)  ...  delta_RL  (trained)
    a_final = clip(a_base + alpha * delta_RL,  [-1,0], [1,1])
      -> real descending neurons -> DN decoder -> force-driven ArcadeFlyBody

alpha in [0.25, 0.4] gives the RL head BOUNDED authority: it can shift WHEN and
HOW HARD the keeper commits (jump height, lateral commitment, mid-air correction,
recovery) but cannot override the vision-grounded base. delta_RL is a tiny tanh
MLP; ES optimises its weights (no torch/PPO in this env). a=[u_lat,u_vert].

Reward (Phase 10): terminal task reward DOMINANT, tiny shaping, ~no movement
penalty (we explicitly want a MORE animated keeper than v2, so staying still is
never rewarded):
    SAVE                          +10
    GOAL                          -10
    meaningful keeper contact     +0.5   (touched the ball this shot)
    useful deflection             +0.5   (deflected clear)
    stable recovery               +0.2   (settled upright near stand after moving)
    extreme instability           -0.5   (tumbled / flew far out of the box)
    pointless oscillation         -0.1   (many lateral sign flips with no contact)
POLICY OBSERVATIONS: the exact policy-visible vector (L/R/vert MaleCNS stream,
previous action, proprioception, airborne). NO ball truth. Privileged state is
used ONLY to compute the reward (allowed), never fed to the policy.

Model selection is on saves AND movement (Phase 10): a candidate that saves a
touch more but stands still is NOT automatically preferred.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.arcade_action_policy import (OBS_NAMES, TinyPolicy,
                                                    BinocularActionPolicyBridge)
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))
AIRBORNE = ("TAKEOFF", "FLIGHT", "DIVE")
HOP_STATES = ("LOW_HOP", "TAKEOFF", "FLIGHT", "DIVE")


def valid_cells(per_cell, base_seed):
    cells, seed = [], base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            got = 0
            while got < per_cell:
                cls, _, _ = SV.classify_shot(seed, g, hf)
                if cls == SV.CLEAN_GOAL:
                    cells.append((seed, g, hname, hf)); got += 1
                seed += 1
    return cells


class RLResidualHead:
    """Tiny tanh MLP delta_RL, ES-trainable, sharing the policy-visible obs."""

    def __init__(self, w1, b1, w2, b2, mean, scale):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.mean, self.scale = mean, scale

    @classmethod
    def zeros(cls, n_in, hidden, mean, scale, rng):
        return cls(rng.normal(0, 0.10, (n_in, hidden)), np.zeros(hidden),
                   rng.normal(0, 0.10, (hidden, 2)), np.zeros(2), mean, scale)

    def flat(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_flat(self, v):
        i = 0
        for arr in (self.w1, self.b1, self.w2, self.b2):
            n = arr.size
            arr[...] = v[i:i + n].reshape(arr.shape); i += n

    def predict(self, obs):
        x = (np.asarray(obs, float) - self.mean) / self.scale
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)


class RLResidualBridge(BinocularActionPolicyBridge):
    """a_base (frozen DAgger policy) + alpha * delta_RL(obs), bounded."""

    def __init__(self, visual, base_policy, rl_head, alpha, base_scale=0.35,
                 base_mode="residual"):
        super().__init__(visual, base_policy, residual_scale=base_scale,
                         mode=base_mode)
        self.rl_head = rl_head
        self.alpha = float(alpha)
        self._osc = 0
        self._last_sign = 0

    def command(self, world):
        base_lat, base_vert = super().command(world)   # a_base already applied
        a_base = np.array([base_lat, base_vert], float)
        delta = self.rl_head.predict(self.last_observation)
        action = np.clip(a_base + self.alpha * delta, [-1.0, 0.0], [1.0, 1.0])
        # oscillation bookkeeping (reward shaping only)
        s = int(np.sign(action[0])) if abs(action[0]) > 0.1 else 0
        if s != 0 and self._last_sign != 0 and s != self._last_sign:
            self._osc += 1
        if s != 0:
            self._last_sign = s
        self.visual_bridge._ema = action
        self.previous = action
        return float(action[0]), float(action[1])


def _episode_reward(world, trace, osc):
    """Terminal-dominant reward with tiny shaping (privileged, reward-only)."""
    d = world.outcome_diagnostics()
    r = 10.0 if world.result == "SAVE" else -10.0
    if d.get("keeper_contact"):
        r += 0.5
    if d.get("deflected_save"):
        r += 0.5
    # stable recovery: ended upright and near stand height after having moved
    if trace:
        z = np.array([t["fly_z"] for t in trace])
        y = np.array([t["fly_y"] for t in trace])
        moved = (np.abs(y - y[0]).max() > 0.2) or (np.abs(z - z[0]).max() > 0.2)
        settled = abs(float(z[-1]) - float(z[0])) < 0.15
        if moved and settled:
            r += 0.2
        # extreme instability: flew far outside the goal box laterally
        if np.abs(y).max() > 1.4:
            r -= 0.5
    # pointless oscillation: many lateral sign flips with no contact
    if osc >= 4 and not d.get("keeper_contact"):
        r -= 0.1
    return r


def _movement(trace):
    if not trace:
        return dict(peak_lat=0.0, peak_vert=0.0, takeoff=0.0, hop=0.0)
    y = np.array([t["fly_y"] for t in trace]); z = np.array([t["fly_z"] for t in trace])
    st = [t.get("movement_state") for t in trace]
    return dict(peak_lat=float(np.abs(y - y[0]).max()),
                peak_vert=float(np.abs(z - z[0]).max()),
                takeoff=float(any(s in AIRBORNE for s in st)),
                hop=float(any(s in HOP_STATES for s in st)))


def evaluate_candidate(brain, base_policy, flat_theta, template_head, alpha,
                       cells, base_scale=0.35, base_mode="residual"):
    """Roll out one RL-residual candidate across cells; return (mean_reward, stats)."""
    head = RLResidualHead(template_head.w1.copy(), template_head.b1.copy(),
                          template_head.w2.copy(), template_head.b2.copy(),
                          template_head.mean, template_head.scale)
    head.set_flat(flat_theta)
    total_r = 0.0
    saves = 0
    mv = []
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        visual, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = RLResidualBridge(visual, base_policy, head, alpha, base_scale,
                                  base_mode=base_mode)
        vision = BinocularVisionBridge(w.fly, brain, condition="both",
                                       retina_map="fullframe")
        ctrl = ArcadeController(brain, vision, Arcade2AxisDecoder(), bridge)
        out, tr = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf),
                              collect_trace=True)
        total_r += _episode_reward(w, tr, bridge._osc)
        saves += int(out["result"] == "SAVE")
        mv.append(_movement(tr))
    n = len(cells)
    stats = dict(save_rate=round(saves / n, 3),
                 peak_lat=round(float(np.mean([m["peak_lat"] for m in mv])), 4),
                 peak_vert=round(float(np.mean([m["peak_vert"] for m in mv])), 4),
                 takeoff=round(float(np.mean([m["takeoff"] for m in mv])), 3),
                 hop=round(float(np.mean([m["hop"] for m in mv])), 3))
    return total_r / n, stats


def selection_score(mean_r, stats, v2_peak_lat=0.44):
    """Phase 10 selection: saves dominate, but movement is part of the score.

    A tiny bonus rewards lateral activity at/above v2, so a static high-save
    candidate is NOT automatically preferred over an active one. The bonus is
    small (<=~0.5 reward units) so it never overrides a real save difference.
    """
    move_bonus = 0.5 * min(1.0, stats["peak_lat"] / max(v2_peak_lat, 1e-6))
    return mean_r + move_bonus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-policy", default="arcade_dagger_full_policy_r2.npz")
    p.add_argument("--base-mode", choices=("residual", "full"), default="full")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--base-scale", type=float, default=0.35)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--per-cell", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=150000)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--out", default="arcade_residual_rl.npz")
    a = p.parse_args()

    if not 0.2 <= a.alpha <= 0.45:
        print(f"[warn] alpha={a.alpha} outside recommended 0.25-0.4 band")

    t0 = time.perf_counter()
    cells = valid_cells(a.per_cell, a.base_seed)
    print(f"[rl] {len(cells)} CLEAN_GOAL training cells; alpha={a.alpha}")
    base_policy, _ = TinyPolicy.load(OUT / a.base_policy)
    rng = np.random.default_rng(a.seed)
    n_in = len(OBS_NAMES)
    # normalisation reused from the base policy (same obs distribution)
    template = RLResidualHead.zeros(n_in, a.hidden, base_policy.mean,
                                    base_policy.scale, rng)
    dim = template.flat().size
    mean = np.zeros(dim)                 # start at delta_RL = ~0 (pure base)
    sigma = 0.25
    history = []
    brain = MaleCNSBrain(backend="metal")
    best_overall = None
    try:
        # baseline (delta=0) reference
        base_r, base_stats = evaluate_candidate(brain, base_policy, mean,
                                                template, a.alpha, cells,
                                                a.base_scale, a.base_mode)
        print(f"[rl] base (delta=0): reward={base_r:.3f} "
              f"save={base_stats['save_rate']} peakLat={base_stats['peak_lat']} "
              f"peakVert={base_stats['peak_vert']}")
        history.append(dict(generation=-1, kind="base", reward=round(base_r, 3),
                            **base_stats))
        best_overall = (selection_score(base_r, base_stats), mean.copy(),
                        base_r, base_stats)

        for gen in range(a.generations):
            cand = mean + rng.normal(size=(a.population, dim)) * sigma
            rewards, scores, stats_list = [], [], []
            for theta in cand:
                mr, st = evaluate_candidate(brain, base_policy, theta, template,
                                            a.alpha, cells, a.base_scale,
                                            a.base_mode)
                rewards.append(mr); stats_list.append(st)
                scores.append(selection_score(mr, st))
            scores = np.array(scores)
            elite = np.argsort(scores)[-max(2, a.population // 3):]
            mean = cand[elite].mean(0)
            sigma = max(0.05, sigma * 0.85)
            gi = int(np.argmax(scores))
            best_stats = stats_list[gi]
            rec = dict(generation=gen, best_reward=round(float(max(rewards)), 3),
                       best_score=round(float(scores.max()), 3),
                       mean_reward=round(float(np.mean(rewards)), 3),
                       best_save=best_stats["save_rate"],
                       best_peak_lat=best_stats["peak_lat"],
                       best_peak_vert=best_stats["peak_vert"],
                       best_takeoff=best_stats["takeoff"])
            history.append(rec)
            print(f"[gen {gen}] best_reward={rec['best_reward']} "
                  f"best_save={rec['best_save']} peakLat={rec['best_peak_lat']} "
                  f"peakVert={rec['best_peak_vert']} takeoff={rec['best_takeoff']} "
                  f"sigma={sigma:.3f}", flush=True)
            # track best candidate by selection score across all gens
            if scores.max() > best_overall[0]:
                best_overall = (float(scores.max()), cand[gi].copy(),
                                rewards[gi], best_stats)
        # re-evaluate the chosen mean at the end as the deployable theta
        final_theta = best_overall[1]
        final_r, final_stats = evaluate_candidate(brain, base_policy, final_theta,
                                                  template, a.alpha, cells,
                                                  a.base_scale, a.base_mode)
    finally:
        brain.close()

    template.set_flat(final_theta)
    meta = dict(method="bounded residual ES-RL over DAgger policy",
                architecture=f"delta_RL {n_in}->{a.hidden}->2 tanh; "
                             f"a_final=clip(a_base+alpha*delta,[-1,0],[1,1])",
                alpha=a.alpha, base_scale=a.base_scale, base_policy=a.base_policy,
                reward="SAVE+10/GOAL-10 dominant; +0.5 contact,+0.5 deflect,"
                       "+0.2 recovery; -0.5 instability,-0.1 oscillation; "
                       "NO movement penalty",
                selection="save_rate-dominant reward + small lateral-movement "
                          "bonus (active keeper not penalised)",
                generations=a.generations, population=a.population,
                per_cell=a.per_cell, n_cells=len(cells),
                final=dict(reward=round(final_r, 3), **final_stats),
                history=history)
    np.savez(OUT / a.out, w1=template.w1, b1=template.b1, w2=template.w2,
             b2=template.b2, mean=template.mean, scale=template.scale,
             alpha=np.array([a.alpha]), base_scale=np.array([a.base_scale]),
             base_mode=np.array([a.base_mode]),
             base_policy=np.array([a.base_policy]),
             metadata=np.array([json.dumps(meta)]))
    (OUT / a.out.replace(".npz", "_train.json")).write_text(json.dumps(meta, indent=2))
    meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"\n[rl] final: reward={final_r:.3f} save={final_stats['save_rate']} "
          f"peakLat={final_stats['peak_lat']} peakVert={final_stats['peak_vert']} "
          f"takeoff={final_stats['takeoff']}  ({meta['wall_seconds']}s)")
    print(f"[rl] saved {a.out}")


if __name__ == "__main__":
    main()
