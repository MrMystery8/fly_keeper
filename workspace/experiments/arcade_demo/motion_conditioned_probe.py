"""Step 1: motion-conditioned L/C/R decoding probe (cheap, episode-disjoint).

Hypothesis: self-motion corrupts the binocular MaleCNS visual representation, but
proprioception + previous action (all LEGITIMATE body/controller signals already
allowed in the deployed policy -- NOT privileged) may explain enough of that
transformation for a learned decoder to recover external ball direction from
CLOSED-LOOP (moving-keeper) trajectories.

Collect corrected-env CLOSED-LOOP trajectories (keeper acts on the frozen hybrid
base command) and record, per approach step:
    * rich binocular MaleCNS features (600-pool x 8 windows)
    * proprioception: fly_y, fly_z-stand, vy, vz, airborne
    * previous action: prev_u_lat, prev_u_vert
    * group label (L/C/R)  [training-only label]

Compare 3-way L/C/R decodability (episode-disjoint) for:
    A. neural features only
    B. neural + proprioception
    C. neural + proprioception + previous action

Report overall acc + per-class recall (esp. RIGHT). Pass criterion ~ overall>=0.45,
center>=0.30, right>=0.40. A biased single-class readout does NOT pass.
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
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_rich_encoder import RichBinocularEncoder
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
CLASSES = ("left", "center", "right")


def _load_pool():
    m = np.load(OUT / "binocular_pool.npz", allow_pickle=False)
    return [int(x) for x in m["body_ids"]], m["somaSide"].astype(str)


def collect_closed(brain, shots, pool_ids, n_windows=8, approach_x=2.0):
    feats, prop, prevact, groups, seeds = [], [], [], [], []
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        vision = BinocularVisionBridge(world.fly, brain, condition="both", retina_map="fullframe")
        decoder = Arcade2AxisDecoder()
        world.reset(shot); brain.reset(); hybrid.reset()
        buf = [np.zeros(len(pool_ids), np.float32) for _ in range(n_windows)]
        prev = np.zeros(2); n = 0
        while world.shot_live and n < MAX_DECISIONS:
            vision.perceive(); hybrid.inject(brain); brain.step(DECISION_MS)
            rd = brain.read(pool_ids)
            buf.append(np.array([rd[i]["spikes"] for i in pool_ids], np.float32)); buf.pop(0)
            hybrid.observe(brain); decoder.decode(brain.read(decoder.readout_ids()))
            fly = world.fly; vel = fly.data.qvel[fly.root_dofadr:fly.root_dofadr + 3]
            stand = float(getattr(fly, "_stand_z", None) or fly.position[2])
            bx = float(world._observe_ball()["pos"][0])
            if bx < approach_x:
                feats.append(np.concatenate(buf[::-1]).astype(np.float32))
                prop.append(np.array([float(fly.position[1]), float(fly.position[2]) - stand,
                                      float(vel[1]), float(vel[2]),
                                      float(getattr(fly, "_airborne", False))], np.float32))
                prevact.append(prev.astype(np.float32).copy())
                groups.append(shot.group); seeds.append(seed)
            u_lat, u_vert = hybrid.command()
            world.fly.set_command(0, 0, 1.0, lateral=u_lat, vertical=u_vert)
            world.step(DECISION_S); prev = np.array([u_lat, u_vert]); n += 1
    return (np.asarray(feats, np.float32), np.asarray(prop, np.float32),
            np.asarray(prevact, np.float32), np.asarray(groups), np.asarray(seeds, np.int64))


def encode_features(X, pool_side, n_windows, k=3, split_mask=None, ridge=5.0, groups=None):
    """Discriminative left/right neural-stream embedding, fit on split_mask=True."""
    npool = len(pool_side)
    Xr = X.reshape(len(X), n_windows, npool)
    li = np.where(pool_side == "L")[0]; ri = np.where(pool_side == "R")[0]
    XL = Xr[:, :, li].reshape(len(X), -1); XR = Xr[:, :, ri].reshape(len(X), -1)
    yi = np.array([CLASSES.index(v) for v in groups]); Y = np.eye(3)[yi]

    def fit(Xe):
        mu = Xe[split_mask].mean(0); sd = Xe[split_mask].std(0); sd[sd < 1e-6] = 1
        Z = (Xe[split_mask] - mu) / sd
        W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ Y[split_mask])[:, :k]
        return ((Xe - mu) / sd) @ W
    return np.c_[fit(XL), fit(XR)]


def softmax_eval(Ztr, ytr, Zte, yte, epochs=500, lr=0.15, l2=1e-3):
    mu = Ztr.mean(0); sd = Ztr.std(0); sd[sd < 1e-6] = 1
    Ztr = (Ztr - mu) / sd; Zte = (Zte - mu) / sd
    yi = np.array([CLASSES.index(v) for v in ytr]); yti = np.array([CLASSES.index(v) for v in yte])
    W = np.zeros((Ztr.shape[1], 3)); b = np.zeros(3); Y = np.eye(3)[yi]
    for _ in range(epochs):
        z = Ztr @ W + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        W -= lr * (Ztr.T @ (p - Y) / len(Ztr) + l2 * W); b -= lr * (p - Y).mean(0)
    pt = (Zte @ W + b).argmax(1)
    rec = {c: round(float((pt[yti == i] == i).mean()), 3) if (yti == i).any() else None
           for i, c in enumerate(CLASSES)}
    return round(float((pt == yti).mean()), 3), rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=320000)
    p.add_argument("--k-per-eye", type=int, default=3)
    a = p.parse_args()
    t0 = time.perf_counter()
    pool_ids, side = _load_pool()
    print(f"[motion-probe] building stratified CLEAN_GOAL shots ({a.per_cell}/cell x9)...")
    shots = SV.stratified_valid_shots(a.per_cell, a.base_seed)
    brain = MaleCNSBrain(backend="metal")
    try:
        print("[motion-probe] collecting CLOSED-LOOP trajectories w/ proprio+action...")
        X, prop, prevact, groups, seeds = collect_closed(brain, shots, pool_ids)
    finally:
        brain.close()
    print(f"[motion-probe] steps={len(X)}")
    uids = np.unique(seeds); rng = np.random.default_rng(7); rng.shuffle(uids)
    ntr = int(0.7 * len(uids)); tr_ids = set(uids[:ntr].tolist())
    tr = np.array([s in tr_ids for s in seeds]); te = ~tr
    # neural embedding fit on TRAIN closed-loop episodes
    emb = encode_features(X, side, 8, k=a.k_per_eye, split_mask=tr, groups=groups)

    def run(name, Z):
        acc, rec = softmax_eval(Z[tr], groups[tr], Z[te], groups[te])
        print(f"  {name:42s} acc={acc}  L/C/R={rec['left']}/{rec['center']}/{rec['right']}")
        return dict(acc=acc, recall=rec)

    print("\n=== MOTION-CONDITIONED CLOSED-LOOP L/C/R DECODING ===")
    res = {}
    res["A_neural_only"] = run("A. neural features only", emb)
    res["B_neural_proprio"] = run("B. neural + proprioception", np.c_[emb, prop])
    res["C_neural_proprio_prevaction"] = run("C. neural + proprio + prev action",
                                             np.c_[emb, prop, prevact])
    # pass criterion on the richest (C)
    c = res["C_neural_proprio_prevaction"]
    rr = c["recall"]["right"] or 0.0; cc = c["recall"]["center"] or 0.0
    genuine = c["acc"] >= 0.45 and cc >= 0.30 and rr >= 0.40
    verdict = ("PASS: motion-conditioned decoding recovers genuine multi-class "
               "direction (incl. right) -> use this representation in the policy"
               if genuine else
               "FAIL: direction stays near chance even with proprio+action -> "
               "switch to EARLY-INTENT (pre-motion sensing) architecture")
    res["verdict"] = verdict
    res["n_steps"] = int(len(X)); res["n_shots"] = len(shots)
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"\n  VERDICT: {verdict}")
    (OUT / "motion_conditioned_probe.json").write_text(json.dumps(res, indent=2))
    # save the collected closed-loop conditioned data for reuse
    np.savez(OUT / "motion_conditioned_data.npz", X=X, prop=prop, prevact=prevact,
             groups=groups, seeds=seeds, pool_body_ids=np.asarray(pool_ids, np.int64),
             pool_side=side, n_windows=np.array([8], np.int32))
    print(f"saved motion_conditioned_probe.json ({res['wall_seconds']}s)")


if __name__ == "__main__":
    main()
