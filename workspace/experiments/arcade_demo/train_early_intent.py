"""Train the early-intent execution controller: supervised seed -> DAgger -> RL.

Stage 1 (supervised): roll out the hand-wired seed controller on stratified
CLEAN_GOAL shots; at each post-sensing step record the 13-dim exec observation
and the privileged teacher (u_lat, u_vert) label; train the ExecPolicy.
Stage 2 (DAgger): roll out the trained ExecPolicy, relabel visited states with
the teacher, aggregate, retrain.
Stage 3 (RL): bounded ES fine-tune of the ExecPolicy weights on SAVE+10/GOAL-10
with a small graded-vertical shaping (anti-overshoot on low balls) and no motion
penalty; select on saves + movement.

Teacher/privileged state is used only for LABELS and REWARD, never in the obs.
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
                                                    run_episode, DECISION_MS,
                                                    DECISION_S, MAX_DECISIONS)
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import (EarlyIntentBridge,
                                                         ExecPolicy, EXEC_OBS_DIM,
                                                         EXEC_OBS_NAMES)
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
AIRBORNE = ("TAKEOFF", "FLIGHT", "DIVE")


def collect(brain, est, shots, policy, sense_steps):
    """Roll out (seed or ExecPolicy) closed-loop; label teacher at post-sensing steps."""
    X, Y, S = [], [], []
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = EarlyIntentBridge(hybrid, est, policy=policy, sense_steps=sense_steps)
        vision = BinocularVisionBridge(world.fly, brain, condition="both", retina_map="fullframe")
        decoder = Arcade2AxisDecoder()
        world.reset(shot); brain.reset(); bridge.reset()
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            vision.perceive(); bridge.inject(brain); brain.step(DECISION_MS); bridge.observe(brain)
            decoder.decode(brain.read(decoder.readout_ids()))
            obs = bridge.observation(world)
            teacher = np.asarray(teacher_command(world)[0], float)
            # SIGN FIX: teacher_command uses u_lat<0 = "strafe left/+y", but the
            # ArcadeFlyBody uses lateral>0 = +y (measured: u_lat=+0.6 -> fly_y=+0.59).
            # The exec bridge feeds the policy action DIRECTLY to set_command(lateral=),
            # so we train the policy in BODY convention by negating the teacher's
            # lateral. Vertical is unaffected.
            teacher_body = np.array([-teacher[0], teacher[1]], float)
            if n >= sense_steps:              # only label the execution phase
                X.append(obs.astype(np.float32)); Y.append(teacher_body.astype(np.float32)); S.append(seed)
            g = bridge.gait_on()
            u_lat, u_vert = bridge.command(world)
            world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S); n += 1
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(S, np.int64)


def train_exec(X, Y, S, out, hidden=16, epochs=800, lr=0.003, split_seed=20260919,
               vert_low_w=2.0):
    ids = np.unique(S); rng = np.random.default_rng(split_seed); rng.shuffle(ids)
    ntr = max(1, int(.7 * len(ids))); nval = max(1, int(.15 * len(ids)))
    tr = np.isin(S, ids[:ntr]); va = np.isin(S, ids[ntr:ntr+nval]); teM = np.isin(S, ids[ntr+nval:])
    mean = X[tr].mean(0); scale = X[tr].std(0); scale[scale < 1e-6] = 1
    Z = (X - mean) / scale
    rng2 = np.random.default_rng(split_seed)
    w1 = rng2.normal(0, .2, (EXEC_OBS_DIM, hidden)); b1 = np.zeros(hidden)
    w2 = rng2.normal(0, .2, (hidden, 2)); b2 = np.zeros(2)
    P = [w1, b1, w2, b2]; m = [np.zeros_like(q) for q in P]; v = [np.zeros_like(q) for q in P]
    sw = np.where(Y[:, 1] < 0.15, vert_low_w, 1.0)     # emphasize low-vertical samples
    best = None
    for step in range(1, epochs + 1):
        h = np.tanh(Z[tr] @ w1 + b1); p = np.tanh(h @ w2 + b2)
        wgt = sw[tr][:, None]; d = 2 * (p - Y[tr]) * wgt / max(1, wgt.sum())
        dz2 = d * (1 - p*p); gw2 = h.T @ dz2; gb2 = dz2.sum(0)
        dh = (dz2 @ w2.T) * (1 - h*h); gw1 = Z[tr].T @ dh; gb1 = dh.sum(0)
        for i, (q, g) in enumerate(zip(P, (gw1, gb1, gw2, gb2))):
            m[i] = .9*m[i]+.1*g; v[i] = .999*v[i]+.001*g*g
            q -= lr*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        hv = np.tanh(Z[va] @ w1 + b1); pv = np.tanh(hv @ w2 + b2)
        loss = float(((pv - Y[va])**2).mean())
        if best is None or loss < best[0]:
            best = (loss, *(q.copy() for q in P), step)
    _, w1, b1, w2, b2, chosen = best
    model = ExecPolicy(w1, b1, w2, b2, mean, scale)

    def sc(msk):
        pr = model.predict(X[msk])
        return dict(mae=round(float(np.abs(pr-Y[msk]).mean()), 4),
                    corr=[round(float(np.corrcoef(Y[msk][:, i], pr[:, i])[0, 1]), 3)
                          if Y[msk][:, i].std() > 1e-8 else None for i in range(2)])
    meta = dict(obs_names=list(EXEC_OBS_NAMES), architecture=f"{EXEC_OBS_DIM}->{hidden}->2 tanh",
                n_steps=int(len(X)), epochs_selected=int(chosen),
                metrics=dict(train=sc(tr), val=sc(va), test=sc(teM)))
    model.save(out, meta)
    return model, meta


# ---- RL stage ----
def episode_reward(world, trace, z_cross):
    d = world.outcome_diagnostics()
    r = 10.0 if world.result == "SAVE" else -10.0
    if d.get("keeper_contact"): r += 0.5
    if d.get("deflected_save"): r += 0.5
    if trace:
        z = np.array([t["fly_z"] for t in trace]); y = np.array([t["fly_y"] for t in trace])
        peak_z = float(np.abs(z - z[0]).max())
        req = max(0.0, float(z_cross) - float(z[0]) - 0.15)
        overshoot = max(0.0, peak_z - (req + 0.45))
        r += 0.3 * (1.0 - min(1.0, abs(peak_z - req) / 0.8)) - 0.6 * min(1.0, overshoot / 0.8)
        if ((np.abs(y - y[0]).max() > 0.2) or (peak_z > 0.2)) and abs(float(z[-1]) - float(z[0])) < 0.15:
            r += 0.2
        if np.abs(y).max() > 1.4: r -= 0.5
    return r


def movement(trace):
    if not trace: return dict(peak_lat=0., peak_vert=0., takeoff=0.)
    y = np.array([t["fly_y"] for t in trace]); z = np.array([t["fly_z"] for t in trace])
    st = [t.get("movement_state") for t in trace]
    return dict(peak_lat=float(np.abs(y-y[0]).max()), peak_vert=float(np.abs(z-z[0]).max()),
                takeoff=float(any(s in AIRBORNE for s in st)))


class RLExecPolicy(ExecPolicy):
    """ExecPolicy whose flat weights ES perturbs."""
    def flat(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])
    def set_flat(self, v):
        i = 0
        for arr in (self.w1, self.b1, self.w2, self.b2):
            nn = arr.size; arr[...] = v[i:i+nn].reshape(arr.shape); i += nn


def rollout_ei(brain, world, est, model, sense_steps, shot):
    """Dedicated early-intent rollout: gait OFF + no DN drive during sensing
    (matches the passive stationary distribution the estimator was fit on)."""
    hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
    bridge = EarlyIntentBridge(hybrid, est, policy=model, sense_steps=sense_steps)
    vision = BinocularVisionBridge(world.fly, brain, condition="both", retina_map="fullframe")
    decoder = Arcade2AxisDecoder()
    world.reset(shot); brain.reset(); bridge.reset()
    trace = []; n = 0
    while world.result is None and n < MAX_DECISIONS:
        vision.perceive(); bridge.inject(brain); brain.step(DECISION_MS); bridge.observe(brain)
        decoder.decode(brain.read(decoder.readout_ids()))
        g = bridge.gait_on()
        u_lat, u_vert = bridge.command(world)
        world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert)
        world.step(DECISION_S)
        trace.append(dict(fly_y=round(float(world.fly.position[1]), 4),
                          fly_z=round(float(world.fly.position[2]), 4),
                          u_lat=round(u_lat, 4), u_vert=round(u_vert, 4),
                          movement_state=str(getattr(world.fly, "state", "?"))))
        n += 1
    return world.result or "GOAL", trace


def rl_evaluate(brain, est, base_model, flat_theta, shots, sense_steps, alpha):
    model = RLExecPolicy(base_model.w1.copy(), base_model.b1.copy(), base_model.w2.copy(),
                         base_model.b2.copy(), base_model.mean, base_model.scale)
    base_flat = np.concatenate([base_model.w1.ravel(), base_model.b1,
                                base_model.w2.ravel(), base_model.b2])
    model.set_flat(base_flat + alpha * flat_theta)
    total_r = 0.0; saves = 0; mv = []; by_h = defaultdict(lambda: [0, 0, [], []])
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        world.reset(shot); zc = float(teacher_command(world)[1].get("z_cross", shot.height))
        result, tr = rollout_ei(brain, world, est, model, sense_steps, shot)
        total_r += episode_reward(world, tr, zc); saves += int(result == "SAVE")
        m = movement(tr); mv.append(m)
        hb = "low" if shot.height < 0.52 else "mid" if shot.height < 0.72 else "high"
        by_h[hb][0] += int(result == "SAVE"); by_h[hb][1] += 1
        by_h[hb][2].append(m["takeoff"]); by_h[hb][3].append(m["peak_vert"])
    n = len(shots)
    stats = dict(save=round(saves/n, 3), peak_lat=round(float(np.mean([m["peak_lat"] for m in mv])), 3),
                 peak_vert=round(float(np.mean([m["peak_vert"] for m in mv])), 3),
                 takeoff=round(float(np.mean([m["takeoff"] for m in mv])), 3),
                 tk_by_h={h: round(float(np.mean(v[2])), 2) for h, v in by_h.items()},
                 pz_by_h={h: round(float(np.mean(v[3])), 2) for h, v in by_h.items()})
    return total_r/n, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sense-steps", type=int, default=8)
    p.add_argument("--per-cell", type=int, default=8)     # supervised/DAgger shots (x9)
    p.add_argument("--dagger-rounds", type=int, default=1)
    p.add_argument("--rl-generations", type=int, default=6)
    p.add_argument("--rl-population", type=int, default=8)
    p.add_argument("--rl-per-cell", type=int, default=3)
    p.add_argument("--rl-alpha", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--sup-seed", type=int, default=300000)
    p.add_argument("--rl-seed", type=int, default=160000)
    p.add_argument("--tag", default="ei")
    p.add_argument("--rl-base", default=None, help="skip sup+DAgger, RL-finetune this policy")
    a = p.parse_args()

    t0 = time.perf_counter()
    est, est_meta = EarlyIntentEstimator.load()
    print(f"[ei] estimator heldout episode acc={est_meta.get('heldout_episode_acc')}")
    sup_shots = SV.stratified_valid_shots(a.per_cell, a.sup_seed)
    print(f"[ei] supervised shots={len(sup_shots)}")
    brain = MaleCNSBrain(backend="metal")
    summary = dict(sense_steps=a.sense_steps, stages=[])
    try:
        if a.rl_base:
            # Resume: skip supervised+DAgger, RL-finetune an existing policy.
            pol_file = a.rl_base
            print(f"[resume] RL base = {pol_file}")
        else:
            # Stage 1: supervised seed
            print("[stage1] collecting seed demos...")
            X, Y, S = collect(brain, est, sup_shots, policy=None, sense_steps=a.sense_steps)
            pol_file = f"arcade_early_intent_{a.tag}_sup.npz"
            model, meta = train_exec(X, Y, S, pol_file, hidden=a.hidden)
            summary["stages"].append(dict(stage="supervised", n=int(len(X)), metrics=meta["metrics"]))
            print(f"[stage1] {len(X)} steps metrics={meta['metrics']}")
            # Stage 2: DAgger
            for r in range(1, a.dagger_rounds + 1):
                print(f"[stage2] DAgger round {r}...")
                model, _ = ExecPolicy.load(pol_file)
                Xr, Yr, Sr = collect(brain, est, sup_shots, policy=model, sense_steps=a.sense_steps)
                X = np.concatenate([X, Xr]); Y = np.concatenate([Y, Yr]); S = np.concatenate([S, Sr])
                pol_file = f"arcade_early_intent_{a.tag}_dagger{r}.npz"
                model, meta = train_exec(X, Y, S, pol_file, hidden=a.hidden)
                summary["stages"].append(dict(stage=f"dagger{r}", n=int(len(X)), metrics=meta["metrics"]))
                print(f"[stage2] +{len(Xr)} (total {len(X)}) metrics={meta['metrics']}")
        # Stage 3: RL
        base_model, _ = ExecPolicy.load(pol_file)
        rl_shots = SV.stratified_valid_shots(a.rl_per_cell, a.rl_seed)
        print(f"[stage3] RL on {len(rl_shots)} shots, alpha={a.rl_alpha}")
        dim = np.concatenate([base_model.w1.ravel(), base_model.b1,
                              base_model.w2.ravel(), base_model.b2]).size
        rng = np.random.default_rng(7); mean = np.zeros(dim); sigma = 1.0
        best = None; hist = []
        base_r, base_stats = rl_evaluate(brain, est, base_model, mean, rl_shots, a.sense_steps, a.rl_alpha)
        print(f"[stage3] base(delta=0): reward={base_r:.3f} save={base_stats['save']} "
              f"peakLat={base_stats['peak_lat']} tk_by_h={base_stats['tk_by_h']} pz_by_h={base_stats['pz_by_h']}")
        hist.append(dict(gen=-1, reward=round(base_r, 3), **base_stats))
        best = (base_r + 0.5*min(1, base_stats['peak_lat']/0.42), mean.copy(), base_stats)
        for gen in range(a.rl_generations):
            cand = mean + rng.normal(size=(a.rl_population, dim)) * sigma
            rewards, scores, stats_l = [], [], []
            for th in cand:
                mr, st = rl_evaluate(brain, est, base_model, th, rl_shots, a.sense_steps, a.rl_alpha)
                rewards.append(mr); stats_l.append(st)
                scores.append(mr + 0.5*min(1, st['peak_lat']/0.42))
            scores = np.array(scores); elite = np.argsort(scores)[-max(2, a.rl_population//3):]
            mean = cand[elite].mean(0); sigma = max(0.2, sigma*0.85)
            gi = int(np.argmax(scores)); bs = stats_l[gi]
            hist.append(dict(gen=gen, reward=round(float(max(rewards)), 3), **bs))
            print(f"[gen {gen}] reward={hist[-1]['reward']} save={bs['save']} peakLat={bs['peak_lat']} "
                  f"tk_by_h={bs['tk_by_h']} pz_by_h={bs['pz_by_h']}", flush=True)
            if scores.max() > best[0]:
                best = (float(scores.max()), cand[gi].copy(), bs)
        final_flat = best[1]
        base_flat = np.concatenate([base_model.w1.ravel(), base_model.b1,
                                    base_model.w2.ravel(), base_model.b2])
        final = ExecPolicy(*(x.copy() for x in (base_model.w1, base_model.b1, base_model.w2, base_model.b2)),
                           base_model.mean, base_model.scale)
        rl = RLExecPolicy(final.w1, final.b1, final.w2, final.b2, final.mean, final.scale)
        rl.set_flat(base_flat + a.rl_alpha * final_flat)
        rl_file = f"arcade_early_intent_{a.tag}_rl.npz"
        rl.save(rl_file, dict(stage="rl", sense_steps=a.sense_steps, alpha=a.rl_alpha,
                              base_policy=pol_file, history=hist,
                              obs_names=list(EXEC_OBS_NAMES)))
        summary["stages"].append(dict(stage="rl", policy=rl_file, history=hist))
        summary["final_policy"] = rl_file
    finally:
        brain.close()
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / f"arcade_early_intent_{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ei] done {summary['wall_seconds']}s")


if __name__ == "__main__":
    main()
