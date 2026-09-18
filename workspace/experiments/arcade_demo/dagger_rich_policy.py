"""Iteration-2 richer-observation DAgger for the binocular Arcade keeper.

Collects closed-loop trajectories on a BROADENED randomized CLEAN_GOAL shot
distribution (continuous lateral/height/speed jitter, not the fixed 3x3 grid),
records the 14-dim RICH observation (per-eye discriminative MaleCNS embedding +
proprioception + previous action), and teacher-labels each visited state.
Trains a compact RichPolicy, then runs 1-2 DAgger aggregation rounds rolling out
the CURRENT policy.

Targets can be 'full' (absolute teacher action) or 'residual' (teacher - base).
The teacher (privileged) is used ONLY for labels; the policy never sees ball truth.
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
from experiments.arcade_demo.arcade_runtime import (Arcade2AxisDecoder,
                                                    DECISION_MS, DECISION_S,
                                                    MAX_DECISIONS)
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_rich_encoder import RichBinocularEncoder
from experiments.arcade_demo.rich_action_policy import (RICH_OBS_NAMES, OBS_DIM,
                                                        RichPolicy, RichActionBridge)
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def rollout_and_label(brain, encoder, shots, policy=None, mode="full",
                      residual_scale=0.35):
    rows, targets, seeds_col = [], [], []
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        bridge = RichActionBridge(hybrid, encoder, policy=policy, mode=mode,
                                  residual_scale=residual_scale)
        vision = BinocularVisionBridge(world.fly, brain, condition="both",
                                       retina_map="fullframe")
        decoder = Arcade2AxisDecoder()
        world.reset(shot); brain.reset(); bridge.reset()
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            vision.perceive()
            bridge.inject(brain)
            brain.step(DECISION_MS)
            bridge.observe(brain)
            decoder.decode(brain.read(decoder.readout_ids()))
            base = np.asarray(hybrid.command(), float)
            obs = bridge.observation(world, base)          # rich 14-dim obs
            teacher = np.asarray(teacher_command(world)[0], float)
            tgt = teacher if mode == "full" else np.clip(teacher - base, -1., 1.)
            rows.append(obs.astype(np.float32)); targets.append(tgt.astype(np.float32))
            seeds_col.append(seed)
            u_lat, u_vert = bridge.command(world)
            world.fly.set_command(0.0, 0.0, 1.0, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S)
            n += 1
    return (np.asarray(rows, np.float32), np.asarray(targets, np.float32),
            np.asarray(seeds_col, np.int64))


def train_rich(X, Y, seeds, out, hidden=16, epochs=800, lr=0.003,
               split_seed=20260918, vert_low_weight=2.0):
    """Train RichPolicy. A mild extra weight on LOW-target vertical samples
    discourages the always-jump habit (graded-vertical objective, label-side)."""
    ids = np.unique(seeds); rng = np.random.default_rng(split_seed); rng.shuffle(ids)
    ntr = max(1, int(.7 * len(ids))); nval = max(1, int(.15 * len(ids)))
    tr_ids, val_ids, te_ids = ids[:ntr], ids[ntr:ntr+nval], ids[ntr+nval:]
    trm = np.isin(seeds, tr_ids); vam = np.isin(seeds, val_ids); tem = np.isin(seeds, te_ids)
    mean = X[trm].mean(0); scale = X[trm].std(0); scale[scale < 1e-6] = 1.0
    Z = (X - mean) / scale
    rng2 = np.random.default_rng(split_seed)
    w1 = rng2.normal(0, .2, (OBS_DIM, hidden)); b1 = np.zeros(hidden)
    w2 = rng2.normal(0, .2, (hidden, 2)); b2 = np.zeros(2)
    params = [w1, b1, w2, b2]; m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]
    # per-sample weight: emphasize samples whose teacher vertical target is LOW
    # (u_vert < 0.15) so the policy learns NOT to jump on low balls.
    sw = np.where(Y[:, 1] < 0.15, vert_low_weight, 1.0)
    best = None
    for step in range(1, epochs + 1):
        h = np.tanh(Z[trm] @ w1 + b1); p = np.tanh(h @ w2 + b2)
        wgt = sw[trm][:, None]
        d = 2 * (p - Y[trm]) * wgt / max(1, wgt.sum())
        dz2 = d * (1 - p * p); gw2 = h.T @ dz2; gb2 = dz2.sum(0)
        dh = (dz2 @ w2.T) * (1 - h * h); gw1 = Z[trm].T @ dh; gb1 = dh.sum(0)
        for i, (q, g) in enumerate(zip(params, (gw1, gb1, gw2, gb2))):
            m[i] = .9*m[i] + .1*g; v[i] = .999*v[i] + .001*g*g
            q -= lr * (m[i] / (1-.9**step)) / (np.sqrt(v[i]/(1-.999**step)) + 1e-8)
        hv = np.tanh(Z[vam] @ w1 + b1); pv = np.tanh(hv @ w2 + b2)
        loss = float(((pv - Y[vam])**2).mean())
        if best is None or loss < best[0]:
            best = (loss, *(q.copy() for q in params), step)
    _, w1, b1, w2, b2, chosen = best
    model = RichPolicy(w1, b1, w2, b2, mean, scale)

    def score(mask):
        pr = model.predict(X[mask])
        return dict(mae=round(float(np.abs(pr - Y[mask]).mean()), 4),
                    corr=[round(float(np.corrcoef(Y[mask][:, i], pr[:, i])[0, 1]), 3)
                          if Y[mask][:, i].std() > 1e-8 else None for i in range(2)])
    meta = dict(version="rich-binocular-action-policy", obs_names=list(RICH_OBS_NAMES),
                architecture=f"{OBS_DIM}->{hidden}->2 tanh", epochs_selected=int(chosen),
                n_steps=int(len(X)), vert_low_weight=vert_low_weight,
                metrics=dict(train=score(trm), val=score(vam), test=score(tem)))
    model.save(out, meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-shots", type=int, default=126)   # 14/cell x 9 strata
    p.add_argument("--base-seed", type=int, default=300000)
    p.add_argument("--encoder", default="binocular_rich_encoder_corrected.npz")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--mode", choices=("full", "residual"), default="full")
    p.add_argument("--residual-scale", type=float, default=0.35)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--tag", default="rich")
    a = p.parse_args()

    t0 = time.perf_counter()
    per_cell = max(1, a.n_shots // 9)
    print(f"[rich-dagger] building stratified CLEAN_GOAL shots "
          f"({per_cell}/cell x 9 = {per_cell*9})...")
    shots = SV.stratified_valid_shots(per_cell, a.base_seed)
    from collections import Counter
    print(f"[rich-dagger] groups: {dict(Counter(s.group for _, s in shots))}")
    encoder, enc_meta = RichBinocularEncoder.load(a.encoder)
    brain = MaleCNSBrain(backend="metal")
    summary = dict(n_shots=len(shots), rounds=a.rounds, mode=a.mode,
                   encoder=enc_meta, rounds_log=[])
    try:
        print("[round 0] collecting base-trajectory rich demos...")
        X, Y, S = rollout_and_label(brain, encoder, shots, policy=None, mode=a.mode)
        pol_file = f"arcade_rich_policy_{a.tag}_r0.npz"
        meta = train_rich(X, Y, S, pol_file, hidden=a.hidden, epochs=a.epochs)
        summary["rounds_log"].append(dict(round=0, n_steps=int(len(X)),
                                          policy=pol_file, metrics=meta["metrics"]))
        print(f"[round 0] {len(X)} steps -> {pol_file}  metrics={meta['metrics']}")
        for r in range(1, a.rounds + 1):
            print(f"[round {r}] rolling out current rich policy...")
            policy, _ = RichPolicy.load(pol_file)
            Xr, Yr, Sr = rollout_and_label(brain, encoder, shots, policy=policy,
                                           mode=a.mode, residual_scale=a.residual_scale)
            X = np.concatenate([X, Xr]); Y = np.concatenate([Y, Yr]); S = np.concatenate([S, Sr])
            pol_file = f"arcade_rich_policy_{a.tag}_r{r}.npz"
            meta = train_rich(X, Y, S, pol_file, hidden=a.hidden, epochs=a.epochs)
            summary["rounds_log"].append(dict(round=r, n_steps=int(len(X)),
                                              new_steps=int(len(Xr)),
                                              policy=pol_file, metrics=meta["metrics"]))
            print(f"[round {r}] +{len(Xr)} (total {len(X)}) -> {pol_file}  metrics={meta['metrics']}")
    finally:
        brain.close()
    summary["final_policy"] = pol_file
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / f"arcade_rich_dagger_{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rich-dagger] done {summary['wall_seconds']}s; final {pol_file}")


if __name__ == "__main__":
    main()
