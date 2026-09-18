"""Corrected-env validation of the richer neural-stream encoder BEFORE DAgger.

Prerequisite checks (Point 2):
  * Collect MaleCNS 600-pool features on the CORRECTED environment, stratified
    LEFT/CENTER/RIGHT x LOW/MID/HIGH with continuous jitter, all CLEAN_GOAL.
  * Two rollout regimes, episode-disjoint train/test:
       A. STATIONARY keeper (u=0)         -> clean, uncorrupted view
       B. CLOSED-LOOP moving keeper        -> keeper acts on the frozen hybrid
                                              base command (self-motion present)
  * Refit the discriminative left/right neural-stream encoder on CORRECTED
    stationary data, then measure 3-way L/C/R decodability (accuracy + per-class
    recall, esp. RIGHT) on held-out episodes for BOTH regimes.

The critical question: does RIGHT discrimination survive self-motion? If moving
RIGHT collapses toward chance while stationary RIGHT is ~0.6, we STOP before big
DAgger and address the observation/self-motion mismatch.

Outputs:
  * binocular_rich_features_corrected.npz  (features + labels + regime)
  * binocular_rich_encoder_corrected.npz   (refit encoder)
  * rich_encoder_probe.json                (decodability report)
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
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
CLASSES = ("left", "center", "right")


def _load_pool():
    m = np.load(OUT / "binocular_pool.npz", allow_pickle=False)
    return ([int(x) for x in m["body_ids"]], m["somaSide"].astype(str))


def collect(brain, shots, regime, pool_ids, n_windows=8, approach_x=2.0):
    """Return (features[N,nw*npool], groups[N], seeds[N]) for one regime.

    STATIONARY: keeper never moves (u=0). CLOSED-LOOP: keeper acts on the frozen
    hybrid base command (self-motion). Only steps while the ball is in the
    approach window (bx < approach_x) and the shot is live are recorded.
    """
    feats, groups, seeds_col = [], [], []
    for seed, shot in shots:
        world = ArcadeGoalkeeperWorld(seed=seed)
        vision = BinocularVisionBridge(world.fly, brain, condition="both",
                                       retina_map="fullframe")
        if regime == "closed":
            hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
            decoder = Arcade2AxisDecoder()
        world.reset(shot); brain.reset()
        if regime == "closed":
            hybrid.reset()
        buf = [np.zeros(len(pool_ids), np.float32) for _ in range(n_windows)]
        n = 0
        while world.shot_live and n < MAX_DECISIONS:
            vision.perceive()
            if regime == "closed":
                hybrid.inject(brain)
            brain.step(DECISION_MS)
            rd = brain.read(pool_ids)
            buf.append(np.array([rd[i]["spikes"] for i in pool_ids], np.float32))
            buf.pop(0)
            bx = float(world._observe_ball()["pos"][0])
            if bx < approach_x:
                feats.append(np.concatenate(buf[::-1]).astype(np.float32))
                groups.append(shot.group); seeds_col.append(seed)
            if regime == "closed":
                hybrid.observe(brain)
                decoder.decode(brain.read(decoder.readout_ids()))
                u_lat, u_vert = hybrid.command()
                world.fly.set_command(0, 0, 1.0, lateral=u_lat, vertical=u_vert)
            else:
                world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
            world.step(DECISION_S); n += 1
    return (np.asarray(feats, np.float32), np.asarray(groups),
            np.asarray(seeds_col, np.int64))


def _fit_encoder(X, groups, seeds, side, n_windows, k=3, split_seed=7, ridge=5.0):
    npool = len(side)
    Xr = X.reshape(len(X), n_windows, npool)
    li = np.where(side == "L")[0]; ri = np.where(side == "R")[0]
    XL = Xr[:, :, li].reshape(len(X), -1); XR = Xr[:, :, ri].reshape(len(X), -1)
    uids = np.unique(seeds); rng = np.random.default_rng(split_seed); rng.shuffle(uids)
    ntr = int(0.7 * len(uids)); tr_ids = set(uids[:ntr].tolist())
    tr = np.array([s in tr_ids for s in seeds])
    yi = np.array([CLASSES.index(v) for v in groups])
    Y = np.eye(len(CLASSES))[yi]

    def fit_eye(Xe):
        mu = Xe[tr].mean(0); sd = Xe[tr].std(0); sd[sd < 1e-6] = 1
        Z = (Xe[tr] - mu) / sd
        A = Z.T @ Z + ridge * np.eye(Z.shape[1])
        W = np.linalg.solve(A, Z.T @ Y[tr])[:, :k]
        return mu, sd, W
    muL, sdL, WL = fit_eye(XL); muR, sdR, WR = fit_eye(XR)
    return dict(li=li, ri=ri, muL=muL, sdL=sdL, WL=WL, muR=muR, sdR=sdR, WR=WR,
                n_windows=n_windows, tr=tr)


def _embed(X, enc, n_windows, side):
    npool = len(side)
    Xr = X.reshape(len(X), n_windows, npool)
    XL = Xr[:, :, enc["li"]].reshape(len(X), -1)
    XR = Xr[:, :, enc["ri"]].reshape(len(X), -1)
    eL = ((XL - enc["muL"]) / enc["sdL"]) @ enc["WL"]
    eR = ((XR - enc["muR"]) / enc["sdR"]) @ enc["WR"]
    return np.c_[eL, eR]


def _softmax_eval(Ztr, ytr, Zte, yte, epochs=400, lr=0.15, l2=1e-3):
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
    p.add_argument("--per-cell", type=int, default=8)   # 9 strata -> 72 shots/regime
    p.add_argument("--base-seed", type=int, default=310000)
    p.add_argument("--k-per-eye", type=int, default=3)
    a = p.parse_args()

    t0 = time.perf_counter()
    pool_ids, side = _load_pool()
    print(f"[probe] pool={len(pool_ids)} L={int((side=='L').sum())} R={int((side=='R').sum())}")
    print(f"[probe] building stratified CLEAN_GOAL shots ({a.per_cell}/cell x 9)...")
    shots = SV.stratified_valid_shots(a.per_cell, a.base_seed)
    from collections import Counter
    print(f"[probe] groups={dict(Counter(s.group for _,s in shots))} "
          f"n={len(shots)}")
    brain = MaleCNSBrain(backend="metal")
    report = dict(n_shots=len(shots), per_cell=a.per_cell, k_per_eye=a.k_per_eye)
    try:
        print("[probe] collecting STATIONARY features...")
        Xs, gs, ss = collect(brain, shots, "stationary", pool_ids)
        print(f"[probe] stationary steps={len(Xs)}")
        print("[probe] collecting CLOSED-LOOP features...")
        Xc, gc, sc = collect(brain, shots, "closed", pool_ids)
        print(f"[probe] closed-loop steps={len(Xc)}")
    finally:
        brain.close()

    # Fit encoder on STATIONARY train episodes, evaluate on held-out episodes of
    # BOTH regimes (same episode split so 'test' seeds are unseen in either).
    enc = _fit_encoder(Xs, gs, ss, side, n_windows=8, k=a.k_per_eye)
    tr = enc["tr"]; te = ~tr
    # stationary held-out
    Zs = _embed(Xs, enc, 8, side)
    acc_s, rec_s = _softmax_eval(Zs[tr], gs[tr], Zs[te], gs[te])
    # closed-loop: split by same seeds
    tr_ids = set(ss[tr].tolist())
    ctr = np.array([s in tr_ids for s in sc]); cte = ~ctr
    Zc = _embed(Xc, enc, 8, side)
    # train a fresh readout on stationary-train embeddings, test on closed-loop-test
    acc_c, rec_c = _softmax_eval(Zs[tr], gs[tr], Zc[cte], gc[cte])
    # also closed-loop trained on closed-loop-train (does the info EXIST at all when moving?)
    acc_cc, rec_cc = _softmax_eval(Zc[ctr], gc[ctr], Zc[cte], gc[cte])

    report["stationary_heldout"] = dict(acc=acc_s, recall=rec_s)
    report["closed_loop_stationaryreadout"] = dict(acc=acc_c, recall=rec_c)
    report["closed_loop_selfreadout"] = dict(acc=acc_cc, recall=rec_cc)
    print("\n=== CORRECTED-ENV L/C/R DECODABILITY ===")
    print(f"  A. stationary heldout        acc={acc_s} recall={rec_s}")
    print(f"  B. closed-loop (stat readout) acc={acc_c} recall={rec_c}")
    print(f"  B'. closed-loop (self readout) acc={acc_cc} recall={rec_cc}")
    # Honest verdict: require (i) genuine OVERALL closed-loop discrimination
    # above chance (0.33) -- a high single-class recall with a collapsed other
    # class is a degenerate biased readout, not real direction -- AND (ii) right
    # recall holding up. Use the self-trained closed-loop readout for the
    # "does the info EXIST when moving" question, but gate on overall accuracy.
    right_stat = rec_s.get("right") or 0.0
    right_move = rec_cc.get("right") or 0.0
    closed_overall = max(acc_c, acc_cc)
    genuine_move = (closed_overall >= 0.45 and (rec_cc.get("center") or 0.0) >= 0.30
                    and right_move >= 0.45)
    verdict = ("PROCEED: direction (incl. right) survives self-motion"
               if genuine_move else
               "STOP: closed-loop direction collapses toward chance under "
               "self-motion (high single-class recall is a degenerate biased "
               "readout, not genuine discrimination) -- fix observation/"
               "self-motion mismatch before large DAgger")
    report["right_stationary"] = right_stat
    report["right_closed_loop"] = right_move
    report["closed_loop_overall_acc"] = closed_overall
    report["verdict"] = verdict
    print(f"\n  RIGHT recall: stationary={right_stat}  closed-loop={right_move}")
    print(f"  VERDICT: {verdict}")

    # save corrected features + refit encoder (for downstream use if we proceed)
    np.savez(OUT / "binocular_rich_features_corrected.npz",
             X_stationary=Xs, groups_stationary=gs, seeds_stationary=ss,
             X_closed=Xc, groups_closed=gc, seeds_closed=sc,
             pool_body_ids=np.asarray(pool_ids, np.int64), pool_side=side,
             n_windows=np.array([8], np.int32))
    np.savez(OUT / "binocular_rich_encoder_corrected.npz",
             pool_body_ids=np.asarray(pool_ids, np.int64), pool_side=side,
             n_windows=np.array([8], np.int32),
             left_idx=enc["li"].astype(np.int64), right_idx=enc["ri"].astype(np.int64),
             muL=enc["muL"], sdL=enc["sdL"], WL=enc["WL"],
             muR=enc["muR"], sdR=enc["sdR"], WR=enc["WR"],
             metadata=np.array([json.dumps(dict(
                 dataset="corrected-env stationary", k_per_eye=a.k_per_eye,
                 n_windows=8, embed_dim=2 * a.k_per_eye,
                 note="left/right neural-stream discriminative encoder (corrected env)"))]))
    report["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "rich_encoder_probe.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved rich_encoder_probe.json ({report['wall_seconds']}s)")


if __name__ == "__main__":
    main()
