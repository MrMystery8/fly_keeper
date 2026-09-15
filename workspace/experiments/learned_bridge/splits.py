"""Episode-level train/val/test splitting (Section 21 -- mandatory).

Splits are by COMPLETE EPISODE (shot seed). Every window of an episode goes to
exactly one split, so adjacent correlated windows never leak across splits.
Splits are stratified by shot group (left/center/right) so each split sees the
full penalty distribution, and are reproducible from a fixed seed. Episode
membership (seeds) is recorded for the report.
"""
from __future__ import annotations
import numpy as np


def episode_splits(dataset, *, frac=(0.6, 0.2, 0.2), seed=20260915):
    """Return dict with 'train','val','test' -> arrays of episode indices,
    stratified by group. Deterministic given the seed."""
    groups = np.array([str(g) for g in dataset["groups"]])
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for g in ("left", "center", "right"):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        test += idx[n_tr + n_va:].tolist()
    out = dict(train=np.array(sorted(train)), val=np.array(sorted(val)),
               test=np.array(sorted(test)))
    return out


def split_report(dataset, splits):
    seeds = dataset["seeds"]; groups = np.array([str(g) for g in dataset["groups"]])
    rep = {}
    for name, idx in splits.items():
        gcount = {g: int((groups[idx] == g).sum()) for g in ("left", "center", "right")}
        rep[name] = dict(n_episodes=int(len(idx)), by_group=gcount,
                         seeds=[int(s) for s in seeds[idx]])
    return rep
