"""Vertical refinement of the Early-Intent RL goalkeeper.

Starts from the verified 53.7% champion (arcade_early_intent_ei_rl.npz) and uses
differential residual authority (small lateral authority, larger vertical authority)
and structured vertical alignment/overshoot shaping to learn graded jump height:
    LOW  -> stay low / low hop
    MID  -> moderate diagonal jump
    HIGH -> strong rise / dive
while strictly protecting the 61.1% RIGHT / 61.1% CENTER lateral breakthrough.
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
from experiments.arcade_demo.arcade_runtime import (Arcade2AxisDecoder,
                                                    DECISION_MS, DECISION_S,
                                                    MAX_DECISIONS)
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import (EarlyIntentBridge,
                                                         ExecPolicy,
                                                         EXEC_OBS_NAMES)
from experiments.arcade_demo.train_early_intent import RLExecPolicy, rollout_ei, movement
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def episode_reward(world, trace, z_cross):
    """Simulator-truth reward for training only:
    - Dominant terminal reward (+10 SAVE vs -10 GOAL)
    - Contact/deflection bonuses
    - Vertical alignment bonus (matching required height)
    - Strong bounded penalty for excessive overshoot on LOW balls
    - Bonus for staying grounded on LOW balls
    - Lateral movement encouragement to protect energetic dives
    """
    d = world.outcome_diagnostics()
    r = 10.0 if world.result == "SAVE" else -10.0
    if d.get("keeper_contact"):
        r += 0.5
    if d.get("deflected_save"):
        r += 0.5
    if d.get("touch_but_goal"):
        r += 0.2

    if trace:
        z = np.array([t["fly_z"] for t in trace])
        y = np.array([t["fly_y"] for t in trace])
        u_v = np.array([t["u_vert"] for t in trace])
        
        stand_z = float(z[0])
        peak_z = float(np.abs(z - stand_z).max())
        peak_lat = float(np.abs(y - y[0]).max())
        
        # Required rise above stand:
        # LOW rolling ball (z_cross ~0.45) needs 0 rise (grounded reach covers z up to 0.59)
        # MID ball (~0.625) needs rise ~0.14+
        # HIGH ball (~0.80) needs rise ~0.31+
        req = max(0.0, float(z_cross) - stand_z - 0.35)
        
        # 1. Vertical alignment bonus
        align = 1.0 - min(1.0, abs(peak_z - req) / 0.50)
        r += 0.8 * align
        
        # 2. Overshoot penalty
        thresh = 0.25 if req == 0.0 else (req + 0.35)
        overshoot = max(0.0, peak_z - thresh)
        if overshoot > 0:
            scale = 2.5 if req == 0.0 else 1.2
            r -= scale * min(1.0, overshoot / 0.40)
            
        # 3. Grounded low-ball bonus
        if req == 0.0 and peak_z <= 0.25:
            r += 0.5
            
        # 4. Lateral movement bonus
        if peak_lat > 0.20:
            r += 0.4 * min(1.0, peak_lat / 0.50)
            
        # 5. Anti-oscillation penalty
        if len(u_v) > 1:
            osc = float(np.abs(np.diff(u_v)).mean())
            r -= 0.15 * min(1.0, osc / 0.25)
            
        # 6. Out of bounds penalty
        if np.abs(y).max() > 1.4:
            r -= 0.5

    return r


def build_authority_vector(base_model, alpha_lat=0.08, alpha_vert=0.35):
    """Differential authority vector:
    Shared hidden layer & lateral parameters get small alpha_lat.
    Vertical output parameters get larger alpha_vert.
    """
    dim = base_model.w1.size + base_model.b1.size + base_model.w2.size + base_model.b2.size
    authority = np.full(dim, alpha_lat, dtype=np.float32)
    # w2: (16, 2). Col 0 is lateral (even), Col 1 is vertical (odd)
    w2_start = base_model.w1.size + base_model.b1.size
    authority[w2_start + 1 : w2_start + base_model.w2.size : 2] = alpha_vert
    # b2: (2,). [0] is lateral, [1] is vertical
    b2_start = w2_start + base_model.w2.size
    authority[b2_start + 1] = alpha_vert
    return authority


def rl_evaluate_refined(brain, est, base_model, flat_theta, authority, shots, sense_steps):
    model = RLExecPolicy(base_model.w1.copy(), base_model.b1.copy(), base_model.w2.copy(),
                         base_model.b2.copy(), base_model.mean, base_model.scale)
    base_flat = np.concatenate([base_model.w1.ravel(), base_model.b1,
                                base_model.w2.ravel(), base_model.b2])
    model.set_flat(base_flat + authority * flat_theta)
    
    total_r = 0.0
    saves = 0
    mv = []
    by_h = defaultdict(lambda: [0, 0, [], []])
    by_g = defaultdict(lambda: [0, 0])
    
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        world.reset(shot)
        zc = float(teacher_command(world)[1].get("z_cross", shot.height))
        result, tr = rollout_ei(brain, world, est, model, sense_steps, shot)
        total_r += episode_reward(world, tr, zc)
        is_save = int(result == "SAVE")
        saves += is_save
        m = movement(tr)
        mv.append(m)
        
        hb = "low" if shot.height < 0.52 else "mid" if shot.height < 0.72 else "high"
        by_h[hb][0] += is_save
        by_h[hb][1] += 1
        by_h[hb][2].append(m["takeoff"])
        by_h[hb][3].append(m["peak_vert"])
        
        by_g[shot.group][0] += is_save
        by_g[shot.group][1] += 1
        
    n = len(shots)
    stats = dict(
        save=round(saves / n, 3),
        peak_lat=round(float(np.mean([m["peak_lat"] for m in mv])), 3),
        peak_vert=round(float(np.mean([m["peak_vert"] for m in mv])), 3),
        takeoff=round(float(np.mean([m["takeoff"] for m in mv])), 3),
        tk_by_h={h: round(float(np.mean(v[2])), 2) for h, v in sorted(by_h.items())},
        pz_by_h={h: round(float(np.mean(v[3])), 2) for h, v in sorted(by_h.items())},
        save_by_h={h: f"{v[0]}/{v[1]}" for h, v in sorted(by_h.items())},
        save_by_g={g: f"{v[0]}/{v[1]}" for g, v in sorted(by_g.items())},
    )
    return total_r / n, stats


def compute_score(mr: float, st: dict) -> float:
    pzh = st["pz_by_h"]
    pz_low = pzh.get("low", 0.5)
    pz_mid = pzh.get("mid", 0.5)
    pz_high = pzh.get("high", 0.5)
    
    score = mr
    # Lateral commitment bonus
    score += 0.5 * min(1.0, st["peak_lat"] / 0.45)
    
    # Differentiated vertical commitment: LOW should be lower than MID and HIGH
    if pz_low < pz_mid:
        score += 0.5 * min(1.0, (pz_mid - pz_low) / 0.15)
    else:
        score -= 0.8 * min(1.0, (pz_low - pz_mid) / 0.15)
        
    if pz_low < pz_high:
        score += 0.5 * min(1.0, (pz_high - pz_low) / 0.20)
    else:
        score -= 0.8 * min(1.0, (pz_low - pz_high) / 0.20)
        
    return score


def main():
    p = argparse.ArgumentParser(description="Early-Intent RL Vertical Refinement")
    p.add_argument("--base-policy", default="arcade_early_intent_ei_rl.npz")
    p.add_argument("--alpha-lat", type=float, default=0.08)
    p.add_argument("--alpha-vert", type=float, default=0.35)
    p.add_argument("--rl-generations", type=int, default=3)
    p.add_argument("--rl-population", type=int, default=6)
    p.add_argument("--rl-per-cell", type=int, default=3)
    p.add_argument("--rl-seed", type=int, default=170000)
    p.add_argument("--sense-steps", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--tag", default="vert_refine1")
    p.add_argument("--out-policy", default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    est, est_meta = EarlyIntentEstimator.load()
    base_model, base_meta = ExecPolicy.load(args.base_policy)
    
    authority = build_authority_vector(base_model, args.alpha_lat, args.alpha_vert)
    dim = authority.size
    
    rl_shots = SV.stratified_valid_shots(args.rl_per_cell, args.rl_seed)
    print(f"[refine] loaded base {args.base_policy} (dim={dim})", flush=True)
    print(f"[refine] RL on {len(rl_shots)} shots ({args.rl_per_cell}/cell x9), alpha_lat={args.alpha_lat}, alpha_vert={args.alpha_vert}, sigma={args.sigma}", flush=True)
    
    brain = MaleCNSBrain(backend="metal")
    summary = dict(
        base_policy=args.base_policy,
        alpha_lat=args.alpha_lat,
        alpha_vert=args.alpha_vert,
        sigma=args.sigma,
        sense_steps=args.sense_steps,
        rl_per_cell=args.rl_per_cell,
        rl_seed=args.rl_seed,
        stages=[]
    )
    
    try:
        rng = np.random.default_rng(42)
        mean = np.zeros(dim, dtype=np.float32)
        sigma = args.sigma
        hist = []
        
        # Evaluate base model
        base_r, base_stats = rl_evaluate_refined(brain, est, base_model, mean, authority,
                                                 rl_shots, args.sense_steps)
        base_score = compute_score(base_r, base_stats)
        print(f"[base] score={base_score:.3f} reward={base_r:.3f} save={base_stats['save']} peakLat={base_stats['peak_lat']} "
              f"pz_by_h={base_stats['pz_by_h']} save_by_h={base_stats['save_by_h']} save_by_g={base_stats['save_by_g']}", flush=True)
        hist.append(dict(gen=-1, score=round(base_score, 3), reward=round(base_r, 3), **base_stats))
        
        best = (base_score, mean.copy(), base_stats)
        
        for gen in range(args.rl_generations):
            # Antithetic (mirror) perturbation sampling
            n_pairs = args.rl_population // 2
            cand = []
            for _ in range(n_pairs):
                delta = rng.normal(size=dim).astype(np.float32) * sigma
                cand.append(mean + delta)
                cand.append(mean - delta)
            if len(cand) < args.rl_population:
                cand.append(mean + rng.normal(size=dim).astype(np.float32) * sigma)
            cand = np.array(cand, dtype=np.float32)
            
            rewards, scores, stats_l = [], [], []
            for th in cand:
                mr, st = rl_evaluate_refined(brain, est, base_model, th, authority,
                                             rl_shots, args.sense_steps)
                rewards.append(mr)
                stats_l.append(st)
                scores.append(compute_score(mr, st))
                
            scores = np.array(scores)
            n_elite = max(2, args.rl_population // 3)
            elite = np.argsort(scores)[-n_elite:]
            mean = cand[elite].mean(0)
            sigma = max(0.12, sigma * 0.85)
            
            gi = int(np.argmax(scores))
            bs = stats_l[gi]
            hist.append(dict(gen=gen, score=round(float(scores[gi]), 3), reward=round(float(rewards[gi]), 3), **bs))
            print(f"[gen {gen}] score={scores[gi]:.3f} reward={rewards[gi]:.3f} save={bs['save']} peakLat={bs['peak_lat']} "
                  f"pz_by_h={bs['pz_by_h']} save_by_h={bs['save_by_h']} save_by_g={bs['save_by_g']}", flush=True)
            
            if scores.max() > best[0]:
                best = (float(scores.max()), cand[gi].copy(), bs)
                
        # Also evaluate the final smoothed ES mean
        mean_r, mean_stats = rl_evaluate_refined(brain, est, base_model, mean, authority,
                                                 rl_shots, args.sense_steps)
        mean_score = compute_score(mean_r, mean_stats)
        print(f"[refine] ES mean: score={mean_score:.3f} reward={mean_r:.3f} save={mean_stats['save']} "
              f"pz_by_h={mean_stats['pz_by_h']} save_by_h={mean_stats['save_by_h']}", flush=True)
        if mean_score > best[0]:
            print(f"[refine] selecting ES mean ({mean_score:.3f} > {best[0]:.3f})")
            final_flat = mean.copy()
        else:
            print(f"[refine] selecting best candidate ({best[0]:.3f} >= {mean_score:.3f})")
            final_flat = best[1]

        base_flat = np.concatenate([base_model.w1.ravel(), base_model.b1,
                                    base_model.w2.ravel(), base_model.b2])
        final = ExecPolicy(*(x.copy() for x in (base_model.w1, base_model.b1, base_model.w2, base_model.b2)),
                           base_model.mean, base_model.scale)
        rl = RLExecPolicy(final.w1, final.b1, final.w2, final.b2, final.mean, final.scale)
        rl.set_flat(base_flat + authority * final_flat)
        
        out_policy = args.out_policy or f"arcade_early_intent_{args.tag}_rl.npz"
        rl.save(out_policy, dict(
            stage="vertical_refine",
            sense_steps=args.sense_steps,
            alpha_lat=args.alpha_lat,
            alpha_vert=args.alpha_vert,
            base_policy=args.base_policy,
            history=hist,
            obs_names=list(EXEC_OBS_NAMES)
        ))
        summary["history"] = hist
        summary["final_policy"] = out_policy
        print(f"[refine] saved policy to {out_policy}")
        
    finally:
        brain.close()
        
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    summary_file = OUT / f"arcade_early_intent_{args.tag}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"[refine] saved summary to {summary_file} ({summary['wall_seconds']}s)")


if __name__ == "__main__":
    main()
