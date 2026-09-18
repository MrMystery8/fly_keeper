"""Train the small residual Arcade action policy from teacher rollouts.

The input archive is intentionally simple: ``observations`` are the ten
policy-visible values named in ``arcade_action_policy.OBS_NAMES``; ``targets``
are privileged teacher-minus-base residual actions captured only during
demonstration collection;
and ``episode_seeds`` keep validation splits episode-disjoint.  It has no ball
coordinates, shot labels, or future trajectory columns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
from experiments.arcade_demo.arcade_action_policy import OBS_NAMES, TinyPolicy

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _split_episode(seeds: np.ndarray, split_seed: int):
    ids = np.unique(seeds).copy()
    rng = np.random.default_rng(split_seed); rng.shuffle(ids)
    n_train = max(1, int(.70 * len(ids))); n_val = max(1, int(.15 * len(ids)))
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


def _score(y, p):
    return dict(mae=round(float(np.abs(y - p).mean()), 5),
                corr=[round(float(np.corrcoef(y[:, i], p[:, i])[0, 1]), 4)
                      if len(y) > 2 and y[:, i].std() > 1e-8 else None
                      for i in range(2)])


def train(dataset="arcade_binocular_action_demos.npz",
          out="arcade_binocular_action_policy.npz", hidden=12, epochs=500,
          lr=.003, split_seed=20260917, target_mode="residual"):
    data = np.load(OUT / dataset, allow_pickle=False)
    x = data["observations"].astype(float); y = data["targets"].astype(float)
    seeds = data["episode_seeds"].astype(int)
    if x.ndim != 2 or x.shape[1] != len(OBS_NAMES) or y.shape != (len(x), 2):
        raise ValueError("demo archive does not have the causal policy schema")
    train_ids, val_ids, test_ids = _split_episode(seeds, split_seed)
    train_mask, val_mask, test_mask = (np.isin(seeds, ids)
                                       for ids in (train_ids, val_ids, test_ids))
    mean = x[train_mask].mean(0); scale = x[train_mask].std(0)
    scale[scale < 1e-6] = 1.0; z = (x - mean) / scale
    rng = np.random.default_rng(split_seed)
    w1 = rng.normal(0, .18, (z.shape[1], hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, .18, (hidden, 2)); b2 = np.zeros(2)
    # Adam is implemented locally to keep this experiment dependency-light.
    params = [w1, b1, w2, b2]; m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]; best = None
    for step in range(1, epochs + 1):
        h = np.tanh(z[train_mask] @ w1 + b1); p = np.tanh(h @ w2 + b2)
        d = 2 * (p - y[train_mask]) / max(1, len(p))
        dz2 = d * (1 - p * p); gw2 = h.T @ dz2; gb2 = dz2.sum(0)
        dh = (dz2 @ w2.T) * (1 - h * h); gw1 = z[train_mask].T @ dh; gb1 = dh.sum(0)
        for i, (q, g) in enumerate(zip(params, (gw1, gb1, gw2, gb2))):
            m[i] = .9*m[i] + .1*g; v[i] = .999*v[i] + .001*g*g
            q -= lr * (m[i] / (1-.9**step)) / (np.sqrt(v[i]/(1-.999**step)) + 1e-8)
        hval = np.tanh(z[val_mask] @ w1 + b1); pval = np.tanh(hval @ w2 + b2)
        loss = float(((pval - y[val_mask])**2).mean())
        if best is None or loss < best[0]:
            best = (loss, *(q.copy() for q in params), step)
    _, w1, b1, w2, b2, chosen_epoch = best
    model = TinyPolicy(w1, b1, w2, b2, mean, scale)
    pred = model.predict(x)
    target = ("privileged_teacher_action" if target_mode == "full" else "clip(privileged_teacher_action - frozen_binocular_base_action)")
    metadata = dict(version="arcade-binocular-action-policy-v1", architecture=f"{len(OBS_NAMES)}->{hidden}->2 tanh MLP", policy_mode=target_mode, observation_names=list(OBS_NAMES), target=target, teacher="privileged Arcade teacher used only for supervised labels", split_seed=split_seed, epochs_selected=int(chosen_epoch), n_steps=int(len(x)), n_episodes=int(len(np.unique(seeds))), split_episodes=dict(train=list(map(int,train_ids)), val=list(map(int,val_ids)), test=list(map(int,test_ids))), metrics=dict(train=_score(y[train_mask],pred[train_mask]), val=_score(y[val_mask],pred[val_mask]), test=_score(y[test_mask],pred[test_mask])))
    model.save(OUT / out, metadata)
    (OUT / out.replace(".npz", "_train.json")).write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["metrics"], indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arcade_binocular_action_demos.npz")
    parser.add_argument("--out", default="arcade_binocular_action_policy.npz"); parser.add_argument("--target-mode",choices=("residual","full"),default="residual")
    parser.add_argument("--hidden", type=int, default=12); parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args(); train(args.dataset, args.out, args.hidden, args.epochs,target_mode=args.target_mode)
