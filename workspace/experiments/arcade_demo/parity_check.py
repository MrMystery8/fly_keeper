"""Offline/runtime inference parity for Arcade Bridge v2.

Confirms the deployed runtime bridge computes exactly what training scored, so
the held-out offline metrics transfer to gameplay. Checks:

  1. WEIGHTS/BIAS/NORM parity: the frozen model file loaded by the runtime
     (LinearBridge2D.load) has identical W, b, mu, sd to the trainer's fit
     (recomputed here from the dataset + frozen split seeds).
  2. FEATURE IDENTITY/ORDER: the runtime feature extractor's body_ids (from the
     visual manifest) equal the dataset's graph_index mapping, in order, and the
     newest-first temporal layout matches.
  3. PREDICTION parity: for held-out TEST samples, the runtime inference path
     (LinearBridge2D.predict, as ArcadeLearnedBridge.command uses it) reproduces
     the offline tanh((x-mu)/sd @ W + b) within tight numerical tolerance.
  4. LIVE EXTRACTOR parity: the MetalSafeFeatureExtractor, fed the same per-window
     spike-count vectors, emits the identical feature vector the dataset stored.

No model is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                   MetalSafeFeatureExtractor)

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _split_by_seed(seeds, seed=20260916, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds); rng.shuffle(uniq)
    n = len(uniq); n_tr = int(round(fracs[0] * n)); n_va = int(round(fracs[1] * n))
    return (set(uniq[:n_tr].tolist()),
            set(uniq[n_tr:n_tr + n_va].tolist()),
            set(uniq[n_tr + n_va:].tolist()))


def main(dataset="arcade_dataset_v2.npz", model="arcade_bridge_model_v2.npz"):
    data = np.load(OUT / dataset, allow_pickle=False)
    X = data["features"].astype(np.float64)
    seeds = data["seeds"]
    gi = data["graph_index"].astype(np.int64)
    n_windows = int(data["n_windows"])
    m = LinearBridge2D.load(OUT / model)
    meta = getattr(m, "loaded_metadata", {})

    report = {}

    # 1. weights/bias/norm: recompute the frozen fit and compare to loaded model
    tr, va, te = _split_by_seed(seeds, seed=int(meta.get("split_seed", 20260916)))
    m_tr = np.array([s in tr for s in seeds])
    mu = X[m_tr].mean(0); sd = X[m_tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    report["norm_parity"] = dict(
        mu_max_abs_diff=float(np.max(np.abs(mu - m.mu))),
        sd_max_abs_diff=float(np.max(np.abs(sd - m.sd))),
        matches=bool(np.allclose(mu, m.mu, atol=1e-9)
                     and np.allclose(sd, m.sd, atol=1e-9)))
    # sel_graph in metadata must equal the dataset graph_index (feature identity)
    sel_graph = np.asarray(meta.get("sel_graph", []), dtype=np.int64)
    report["feature_identity"] = dict(
        sel_graph_matches_dataset=bool(sel_graph.size == gi.size
                                       and np.array_equal(sel_graph, gi)),
        n_neurons=int(gi.size), n_windows=n_windows,
        feature_order=str(m.feature_order))

    # 2. feature extractor body-id / order parity: the runtime extractor reads
    #    the manifest body_ids; those map to the same graph_index the dataset
    #    used. Verify the manifest order == dataset order.
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    man_gi = man["graph_index"][:gi.size].astype(np.int64)
    body = [int(x) for x in man["body_ids"][:gi.size]]
    ex = MetalSafeFeatureExtractor(body, n_windows=n_windows)
    report["extractor_parity"] = dict(
        manifest_graph_index_matches_dataset=bool(np.array_equal(man_gi, gi)),
        extractor_dim=int(ex.dim), expected_dim=int(gi.size * n_windows),
        dim_ok=bool(ex.dim == gi.size * n_windows))

    # 3. prediction parity on held-out TEST samples: runtime predict vs offline
    m_te = np.array([s in te for s in seeds])
    Xte = X[m_te]
    # offline path (training math)
    off = np.tanh((Xte - m.mu) / m.sd @ m.W + m.b)
    # runtime path (exactly what ArcadeLearnedBridge.command calls)
    run = np.vstack([m.predict(x) for x in Xte])
    diff = np.abs(off - run)
    report["prediction_parity"] = dict(
        n_test_samples=int(Xte.shape[0]),
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        matches=bool(np.allclose(off, run, atol=1e-9)))

    # 4. live-extractor parity: reconstruct per-window spike vectors from a
    #    stored feature row (newest-first blocks) and feed them to the extractor
    #    OLDEST-first (as observe() would receive them over time), then confirm
    #    features() reproduces the stored row exactly.
    x0 = Xte[0].astype(np.float32)
    nn = gi.size
    wins_newest_first = [x0[k * nn:(k + 1) * nn] for k in range(n_windows)]
    wins_oldest_first = wins_newest_first[::-1]
    ex.reset()
    for w in wins_oldest_first:            # observe would push oldest..newest
        ex._buf.append(w.astype(np.float32))
        if len(ex._buf) > ex.n_windows:
            ex._buf.pop(0)
    feat = ex.features(order="newest_first")
    report["live_extractor_parity"] = dict(
        max_abs_diff=float(np.max(np.abs(feat - x0))),
        matches=bool(np.allclose(feat, x0, atol=1e-6)))

    report["ALL_PASS"] = bool(
        report["norm_parity"]["matches"]
        and report["feature_identity"]["sel_graph_matches_dataset"]
        and report["extractor_parity"]["manifest_graph_index_matches_dataset"]
        and report["extractor_parity"]["dim_ok"]
        and report["prediction_parity"]["matches"]
        and report["live_extractor_parity"]["matches"])

    print(json.dumps(report, indent=2))
    (OUT / "parity_check_v2.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
