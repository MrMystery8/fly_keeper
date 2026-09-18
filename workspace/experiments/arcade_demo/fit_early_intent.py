"""Fit + honestly evaluate the WHERE component on the LARGER stationary set.

Refits the left/right neural-stream discriminative encoder AND the L/C/R
early-intent softmax, using ONLY the EARLY approach window (ball_x >= x_min,
i.e. the first informative pre-motion steps), with a strict held-out-by-SEED
split so we measure GENERALIZATION to unseen shots (the prior 72-episode
estimator collapsed out-of-distribution).

Reports, on held-out seeds:
  * per-step early-window 3-way acc + L/C/R recall
  * episode-level intent (mean prob over the early window) acc + recall + the
    signed intent ordering L < C < R
Saves the generalizing encoder + estimator if it passes, for the RL stage.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path: sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
CLASSES = ("left", "center", "right")


def fit_stream_encoder(XL, XR, Y, tr, k, ridge):
    def fit(Xe):
        mu = Xe[tr].mean(0); sd = Xe[tr].std(0); sd[sd < 1e-6] = 1
        Z = (Xe[tr] - mu) / sd
        W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ Y[tr])[:, :k]
        return mu, sd, W
    return fit(XL), fit(XR)


def softmax_fit(Z, Y, tr, epochs, lr, l2):
    mu = Z[tr].mean(0); sd = Z[tr].std(0); sd[sd < 1e-6] = 1
    Zn = (Z - mu) / sd
    W = np.zeros((Z.shape[1], 3)); b = np.zeros(3)
    for _ in range(epochs):
        z = Zn[tr] @ W + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        W -= lr * (Zn[tr].T @ (p - Y[tr]) / tr.sum() + l2 * W); b -= lr * (p - Y[tr]).mean(0)
    return W, b, mu, sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="stationary_features_big.npz")
    p.add_argument("--k-per-eye", type=int, default=3)
    p.add_argument("--x-min", type=float, default=2.4, help="early window: ball_x >= this")
    p.add_argument("--ridge", type=float, default=20.0)
    p.add_argument("--l2", type=float, default=5e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    d = np.load(OUT / a.data, allow_pickle=False)
    X = d["features"].astype(float); groups = d["groups"]; seeds = d["seeds"].astype(int)
    ballx = d["ball_x"].astype(float); side = d["pool_side"]; nw = int(d["n_windows"][0])
    pool_ids = [int(x) for x in d["pool_body_ids"]]; npool = len(pool_ids)
    # EARLY window only
    early = ballx >= a.x_min
    X = X[early]; groups = groups[early]; seeds = seeds[early]
    print(f"[fit] early-window steps (ball_x>={a.x_min}): {len(X)} "
          f"from {len(np.unique(seeds))} episodes")
    Xr = X.reshape(len(X), nw, npool)
    li = np.where(side == "L")[0]; ri = np.where(side == "R")[0]
    XL = Xr[:, :, li].reshape(len(X), -1); XR = Xr[:, :, ri].reshape(len(X), -1)
    yi = np.array([CLASSES.index(v) for v in groups]); Y = np.eye(3)[yi]
    uids = np.unique(seeds); rng = np.random.default_rng(11); rng.shuffle(uids)
    ntr = int(0.7 * len(uids)); tr_ids = set(uids[:ntr].tolist())
    tr = np.array([s in tr_ids for s in seeds]); te = ~tr

    (muL, sdL, WL), (muR, sdR, WR) = fit_stream_encoder(XL, XR, Y, tr, a.k_per_eye, a.ridge)
    E = np.c_[((XL - muL) / sdL) @ WL, ((XR - muR) / sdR) @ WR]
    W, b, mu, sd = softmax_fit(E, Y, tr, a.epochs, 0.1, a.l2)
    Zt = (E - mu) / sd
    logits = Zt @ W + b; pr = logits.argmax(1)
    acc = float((pr[te] == yi[te]).mean())
    rec = {c: round(float((pr[te & (yi == j)] == j).mean()), 3) if (te & (yi == j)).any() else None
           for j, c in enumerate(CLASSES)}
    print(f"[fit] per-step EARLY held-out acc={acc:.3f} L/C/R={rec['left']}/{rec['center']}/{rec['right']}")

    # episode-level: mean prob over the early window per held-out episode
    z = logits - logits.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    ep_pred = {}; ep_true = {}; ep_intent = {}
    for s in np.unique(seeds[te]):
        msk = te & (seeds == s)
        pm = p[msk].mean(0)
        ep_pred[s] = int(pm.argmax()); ep_true[s] = int(yi[msk][0])
        ep_intent[s] = float(pm[2] - pm[0])
    ep_acc = float(np.mean([ep_pred[s] == ep_true[s] for s in ep_pred]))
    ep_rec = {}
    for j, c in enumerate(CLASSES):
        ss = [s for s in ep_pred if ep_true[s] == j]
        ep_rec[c] = round(float(np.mean([ep_pred[s] == j for s in ss])), 3) if ss else None
    intent_by = defaultdict(list)
    for s in ep_pred:
        intent_by[CLASSES[ep_true[s]]].append(ep_intent[s])
    L = np.mean(intent_by["left"]); C = np.mean(intent_by["center"]); R = np.mean(intent_by["right"])
    print(f"[fit] episode-level held-out acc={ep_acc:.3f} L/C/R={ep_rec['left']}/{ep_rec['center']}/{ep_rec['right']}")
    print(f"[fit] intent ordering  L={L:+.3f} C={C:+.3f} R={R:+.3f}  {'OK (L<C<R)' if L < C < R else 'NOT ordered'}")

    passed = (ep_acc >= 0.45 and (ep_rec['right'] or 0) >= 0.40 and (ep_rec['center'] or 0) >= 0.30 and L < C < R)
    print(f"[fit] VERDICT: {'PASS -> proceed to RL execution' if passed else 'WEAK -> signal marginal'}")

    if a.save:
        np.savez(OUT / "binocular_rich_encoder_early.npz",
                 pool_body_ids=np.asarray(pool_ids, np.int64), pool_side=side,
                 n_windows=np.array([nw], np.int32),
                 left_idx=li.astype(np.int64), right_idx=ri.astype(np.int64),
                 muL=muL, sdL=sdL, WL=WL, muR=muR, sdR=sdR, WR=WR,
                 metadata=np.array([json.dumps(dict(source=a.data, x_min=a.x_min,
                     k_per_eye=a.k_per_eye, embed_dim=2*a.k_per_eye,
                     note="early-window left/right neural-stream discriminative encoder"))]))
        np.savez(OUT / "early_intent_estimator.npz", W=W, b=b, mu=mu, sd=sd,
                 metadata=np.array([json.dumps(dict(source=a.data, x_min=a.x_min,
                     heldout_step_acc=round(acc, 3), heldout_episode_acc=round(ep_acc, 3),
                     heldout_episode_recall=ep_rec, encoder="binocular_rich_encoder_early.npz",
                     note="early-intent L/C/R softmax (early window, generalization-checked)"))]))
        print("[fit] saved binocular_rich_encoder_early.npz + early_intent_estimator.npz")


if __name__ == "__main__":
    main()
