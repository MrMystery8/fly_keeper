"""PHASE 7-8: structured separate-stream late-fused binocular LATERAL hybrid.

DEVELOPMENT HYBRID (experimental isolation tool -- NOT a final "V4"):

    LEFT eye  -> retina -> MaleCNS -> LEFT-somaSide optic-lobe features
                                       -> left  linear lateral estimator  \
                                                                           -> late
                                                                              linear
                                                                              fusion -> u_lat
    RIGHT eye -> retina -> MaleCNS -> RIGHT-somaSide optic-lobe features
                                       -> right linear lateral estimator  /

    fused u_lat -> REAL lateral DN opponent pathway (DNp20/DNpe017 L/R via
                   ArcadeDNBasis) -> Arcade locomotor controller

The VERTICAL path is the EXISTING best-validated V3 vertical head, loaded
FROZEN and unchanged (Phase 8). No new z*, vertical feature selection, vertical
teacher, RL, or vertical DN mapping is introduced. This isolates the single
question: does structured separate-stream late fusion improve LATERAL control?

Everything is LINEAR and interpretable (no MLP). Feature/window/alpha and the
late-fusion weight are selected on TRAIN/VAL only; TEST is read once.

Streams are defined by the connectome somaSide of the 600-neuron bilateral pool
(pool_side in the dataset), i.e. real left-hemisphere vs right-hemisphere
optic-lobe neurons -- not invented neuron-pair subtraction. Trained on the
geometry-corrected (retina_map="fullframe") binocular dataset.
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
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.arcade_demo.binocular_temporal_foundation import (
    _design, _corr, _ridge, _select, stratified_episode_split, WINDOWS, BUDGETS, ALPHAS)
from experiments.arcade_demo.binocular_v3 import V3Head, ArcadeBinocularV3Bridge
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _fit_stream(raw, y, masks, pool_n, sub_index, max_w):
    """Fit a linear lateral estimator on ONE hemisphere's neuron subset.

    raw: [N, max_w*pool] newest-first features (full pool).
    sub_index: indices (into the pool axis) of this stream's neurons.
    Selection (window/budget/alpha + which neurons) on TRAIN, chosen on VAL.
    Returns (pred_all, V3Head-config dict, info).
    """
    best = None
    for window in WINDOWS:
        if window > max_w:
            continue
        # design restricted to this stream's neurons only
        X_full = raw.reshape(len(raw), -1, pool_n)[:, :window, :]
        Xs_stream = X_full[:, :, sub_index].reshape(len(raw), window * len(sub_index))
        activity = Xs_stream.reshape(len(raw), window, len(sub_index)).sum(1)
        for budget in BUDGETS:
            chosen_local, _ = _select(activity, y, masks["train"], budget)
            xs = Xs_stream.reshape(len(raw), window, len(sub_index))[:, :, chosen_local]
            xs = xs.reshape(len(raw), -1)
            mu = xs[masks["train"]].mean(0)
            sd = np.where(xs[masks["train"]].std(0) < 1e-6, 1., xs[masks["train"]].std(0))
            z = (xs - mu) / sd
            for alpha in ALPHAS:
                w, b = _ridge(z[masks["train"]], y[masks["train"]], alpha)
                p = np.tanh(z @ w + b)
                score = _corr(y[masks["val"]], p[masks["val"]])
                if best is None or score > best[0]:
                    # global neuron indices this stream selected
                    global_idx = sub_index[chosen_local]
                    best = (score, window, budget, alpha, w, b, mu, sd, global_idx, p)
    _, window, budget, alpha, w, b, mu, sd, global_idx, p = best
    info = dict(window_bins=window, budget=budget, alpha=alpha,
                n_selected=int(len(global_idx)),
                validation_corr=round(_corr(y[masks["val"]], p[masks["val"]]), 4),
                test_corr=round(_corr(y[masks["test"]], p[masks["test"]]), 4))
    return p, dict(window=window, w=w, b=b, mu=mu, sd=sd, global_idx=global_idx), info


def _metrics(y, p):
    use = np.abs(y) > .05
    if use.sum() < 2:
        return dict(corr=round(_corr(y, p), 4))
    pos, neg = p[use & (y > 0)], p[use & (y < 0)]
    auc = ((pos[:, None] > neg[None, :]).mean()
           + .5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None
    return dict(corr=round(_corr(y, p), 4),
                direction_accuracy=round(float((np.sign(p[use]) == np.sign(y[use])).mean()), 4),
                auc_right_vs_left=(round(float(auc), 4) if auc is not None else None))


def train(dataset="binocular_balanced_dataset_fullframe.npz",
          vertical_from="arcade_binocular_bridge_v3.npz",
          out="arcade_binocular_lateral_hybrid.npz", split_seed=20260917):
    d = np.load(OUT / dataset, allow_pickle=False)
    raw = d["features"].astype(float)
    ids = d["pool_body_ids"].astype(np.int64)
    sides = d["pool_side"].astype(str)
    pool = len(ids)
    groups = d["groups"].astype(str); heights = d["height_classes"].astype(str)
    seeds = d["seeds"].astype(int); ul = d["u_lat"].astype(float)
    max_w = int(np.asarray(d["n_windows"]).ravel()[0])
    retmap = str(d["retina_map"][0]) if "retina_map" in d else "viewport"

    left_sub = np.where(sides == "L")[0]
    right_sub = np.where(sides == "R")[0]
    parts, _ = stratified_episode_split(seeds, groups, heights, split_seed)
    masks = {k: np.isin(seeds, list(v)) for k, v in parts.items()}

    pL, cfgL, infoL = _fit_stream(raw, ul, masks, pool, left_sub, max_w)
    pR, cfgR, infoR = _fit_stream(raw, ul, masks, pool, right_sub, max_w)

    # late linear fusion weight chosen on VALIDATION only
    alphas = np.linspace(0., 1., 21)
    fw = float(max(alphas, key=lambda a: _corr(ul[masks["val"]],
                                                ((1 - a) * pL + a * pR)[masks["val"]])))
    fused = (1 - fw) * pL + fw * pR

    report = dict(
        analysis="structured separate-stream late-fused binocular LATERAL hybrid",
        development_hybrid=True, is_v4=False, dataset=dataset, retina_map=retmap,
        vertical_path=f"FROZEN existing V3 vertical head from {vertical_from}",
        left_stream=infoL, right_stream=infoR,
        fusion_right_weight=round(fw, 3),
        left_only_test=_metrics(ul[masks["test"]], pL[masks["test"]]),
        right_only_test=_metrics(ul[masks["test"]], pR[masks["test"]]),
        fused_validation=_metrics(ul[masks["val"]], fused[masks["val"]]),
        fused_test=_metrics(ul[masks["test"]], fused[masks["test"]]),
        n_left_neurons=int(len(left_sub)), n_right_neurons=int(len(right_sub)),
        split={k: sorted(map(int, v)) for k, v in parts.items()},
    )

    # ---- package the deployable hybrid bridge weights ----
    def head_ids(cfg):
        return ids[cfg["global_idx"]]
    np.savez(
        OUT / out,
        # left lateral stream
        left_ids=head_ids(cfgL).astype(np.int64),
        left_windows=np.array([cfgL["window"]]), left_W=cfgL["w"],
        left_b=np.array([cfgL["b"]]), left_mu=cfgL["mu"], left_sd=cfgL["sd"],
        # right lateral stream
        right_ids=head_ids(cfgR).astype(np.int64),
        right_windows=np.array([cfgR["window"]]), right_W=cfgR["w"],
        right_b=np.array([cfgR["b"]]), right_mu=cfgR["mu"], right_sd=cfgR["sd"],
        fusion_right_weight=np.array([fw]),
        vertical_from=np.array([vertical_from]),
        metadata=np.array([json.dumps(report)]))
    (OUT / out.replace(".npz", "_train.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("left_stream", "right_stream", "fusion_right_weight",
                       "left_only_test", "right_only_test", "fused_test")}, indent=2))
    return report


# ------------------------------- closed-loop bridge -------------------------
class StructuredLateralHybridBridge:
    """Left+right linear lateral estimators -> late fusion -> lateral DN;
    frozen V3 vertical head -> vertical DN. Real DNs stay in the causal path.

    Mirrors ArcadeBinocularV3Bridge's interface (reset/observe/command/inject)
    so it drops into the existing ArcadeController loop unchanged.
    """
    def __init__(self, left_head, right_head, fusion_right_weight, vertical_head,
                 basis, lat_gain=3.5, vert_gain=4.5, smoothing=.2):
        self.left = left_head
        self.right = right_head
        self.fw = float(fusion_right_weight)
        self.vertical = vertical_head
        self.basis = basis
        self.body_ids = list(dict.fromkeys(
            left_head.body_ids + right_head.body_ids + vertical_head.body_ids))
        self._index = {b: i for i, b in enumerate(self.body_ids)}
        self.n_windows = max(left_head.n_windows, right_head.n_windows,
                             vertical_head.n_windows)
        self.lat_gain = float(lat_gain); self.vert_gain = float(vert_gain)
        self.smoothing = float(smoothing); self.enabled = True
        # per-step diagnostics for the closed-loop trace
        self.last_pL = 0.0; self.last_pR = 0.0
        self.reset()

    def reset(self):
        self._buf = [np.zeros(len(self.body_ids), np.float32)
                     for _ in range(self.n_windows)]
        self._raw = np.zeros(2); self._ema = np.zeros(2)
        self.last_pL = 0.0; self.last_pR = 0.0

    def observe(self, brain):
        rd = brain.read(self.body_ids)
        self._buf.append(np.array([rd[i]["spikes"] for i in self.body_ids], np.float32))
        self._buf.pop(0)

    def command(self):
        pL = self.left.predict(self._buf, self._index)
        pR = self.right.predict(self._buf, self._index)
        self.last_pL = pL; self.last_pR = pR
        u_lat = (1 - self.fw) * pL + self.fw * pR
        u_vert = self.vertical.predict(self._buf, self._index)
        self._raw[:] = [u_lat, u_vert]
        self._ema = self.smoothing * self._ema + (1 - self.smoothing) * self._raw
        return float(self._ema[0]), float(max(0., self._ema[1]))

    def inject(self, brain):
        return self.basis.inject(brain, float(self._ema[0]) * self.lat_gain,
                                 float(max(0., self._ema[1])) * self.vert_gain)


def load(path, basis, lat_gain=3.5, vert_gain=4.5, smoothing=.2):
    d = np.load(OUT / path if not str(path).startswith("/") else path,
                allow_pickle=False)
    left = V3Head(d["left_ids"], int(d["left_windows"][0]), d["left_W"],
                  d["left_b"][0], d["left_mu"], d["left_sd"])
    right = V3Head(d["right_ids"], int(d["right_windows"][0]), d["right_W"],
                   d["right_b"][0], d["right_mu"], d["right_sd"])
    fw = float(d["fusion_right_weight"][0])
    vfrom = str(d["vertical_from"][0])
    vd = np.load(OUT / vfrom, allow_pickle=False)
    vertical = V3Head(vd["vert_ids"], int(vd["vert_windows"][0]), vd["vert_W"],
                      vd["vert_b"][0], vd["vert_mu"], vd["vert_sd"])
    bridge = StructuredLateralHybridBridge(left, right, fw, vertical, basis,
                                           lat_gain, vert_gain, smoothing)
    meta = json.loads(str(d["metadata"][0]))
    return bridge, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="binocular_balanced_dataset_fullframe.npz")
    p.add_argument("--vertical-from", default="arcade_binocular_bridge_v3.npz")
    p.add_argument("--out", default="arcade_binocular_lateral_hybrid.npz")
    a = p.parse_args()
    train(a.dataset, a.vertical_from, a.out)
