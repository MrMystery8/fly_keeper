"""Offline bridge training + diagnostics (Stages 1-2, alignment, feature sweep).

Runs entirely on the saved dataset (no brain sim):

  Stage 1  LEFT/RIGHT sanity classifier (Section 20.1): does the exact feature
           pipeline preserve the direction info seen in diagnostics? Uses only
           left/right episodes, logistic regression, held-out by episode.

  Alignment sweep (Section 15): pick the causal target offset (t..t+80ms) that
           best predicts the teacher, on train/val ONLY.

  Feature-count sweep (Section 13): ridge on 16/32/64/128/207 top neurons.

  Stage 2  Continuous teacher regression (Section 20.2): ridge, alpha chosen on
           val, evaluated on the untouched test split. Reports corr / MAE / MSE
           / signed-dir-acc and L/C/R confusion.

Feature selection + normalization + alpha/offset selection use train (+val)
ONLY. The test split is untouched until the final model is frozen. The frozen
LinearBridgeModel is saved for closed-loop use.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))

from experiments.learned_bridge import features as F
from experiments.learned_bridge.splits import episode_splits, split_report
from experiments.learned_bridge.bridge import LinearBridgeModel

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
DECISION_MS = 20.0


def load_dataset(name="dataset_main.npz"):
    return np.load(OUT / name, allow_pickle=True)


# --------------------------------------------------------------- Stage 1
def stage1_sanity(ds, splits, n_windows=4):
    """LEFT vs RIGHT logistic classifier on selected temporal features."""
    groups = np.array([str(g) for g in ds["groups"]])
    def lr_eps(idx):
        return [i for i in idx if groups[i] in ("left", "right")]
    tr = lr_eps(splits["train"]); te = lr_eps(splits["test"])
    Xtr, ytr, _ = F.build_matrix(ds, tr, n_windows=n_windows, label_kind="sign")
    Xte, yte, _ = F.build_matrix(ds, te, n_windows=n_windows, label_kind="sign")
    # labels -1/+1 -> 0/1
    ytr01 = (ytr > 0).astype(float); yte01 = (yte > 0).astype(float)
    mu, sd = F.fit_normalizer(Xtr)
    Xtrn = F.apply_normalizer(Xtr, mu, sd); Xten = F.apply_normalizer(Xte, mu, sd)
    w, b = F.logistic_fit(Xtrn, ytr01, alpha=1.0, lr=0.5, iters=800)
    ptr = F.logistic_predict_proba(Xtrn, w, b)
    pte = F.logistic_predict_proba(Xten, w, b)
    acc_tr = float(((ptr > 0.5) == ytr01).mean())
    acc_te = float(((pte > 0.5) == yte01).mean())
    # AUC on test
    def auc(p, y):
        pos = p[y == 1]; neg = p[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        allv = np.concatenate([pos, neg])
        order = allv.argsort(); ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(allv) + 1)
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))
    return dict(n_windows=n_windows, train_acc=round(acc_tr, 4),
                test_acc=round(acc_te, 4), test_auc=round(auc(pte, yte01), 4),
                n_train=int(len(ytr)), n_test=int(len(yte)))


# --------------------------------------------------------------- alignment
def alignment_sweep(ds, splits, n_windows=4, offsets=(0, 1, 2, 3, 4), alpha=10.0):
    """Choose the causal target offset on train/val only (predict u at t+off)."""
    tr = splits["train"]; va = splits["val"]
    res = []
    for off in offsets:
        Xtr, ytr, _ = F.build_matrix(ds, tr, n_windows=n_windows,
                                     target_offset=off, label_kind="continuous")
        Xva, yva, _ = F.build_matrix(ds, va, n_windows=n_windows,
                                     target_offset=off, label_kind="continuous")
        mu, sd = F.fit_normalizer(Xtr)
        w, b = F.ridge_fit(F.apply_normalizer(Xtr, mu, sd), ytr, alpha=alpha)
        pred = F.ridge_predict(F.apply_normalizer(Xva, mu, sd), w, b)
        m = F.regression_metrics(yva, pred)
        res.append(dict(offset_windows=off, offset_ms=off * DECISION_MS, **m))
    best = max(res, key=lambda r: r["corr"])
    return dict(sweep=res, best_offset_windows=best["offset_windows"],
                best_offset_ms=best["offset_ms"], best_val_corr=best["corr"])


# --------------------------------------------------------------- feature sweep
def feature_count_sweep(ds, splits, *, n_windows=4, target_offset=0,
                        counts=(16, 32, 64, 128, 207), alpha=10.0):
    """Ridge on the top-k selected neurons (by manifest score order)."""
    tr = splits["train"]; va = splits["val"]
    n_sel = int(ds["n_sel"])
    res = []
    for k in counts:
        k = min(k, n_sel)
        fidx = np.arange(k)                      # manifest is score-ordered
        Xtr, ytr, _ = F.build_matrix(ds, tr, n_windows=n_windows,
                                     feature_idx=fidx, target_offset=target_offset)
        Xva, yva, _ = F.build_matrix(ds, va, n_windows=n_windows,
                                     feature_idx=fidx, target_offset=target_offset)
        mu, sd = F.fit_normalizer(Xtr)
        w, b = F.ridge_fit(F.apply_normalizer(Xtr, mu, sd), ytr, alpha=alpha)
        pred = F.ridge_predict(F.apply_normalizer(Xva, mu, sd), w, b)
        m = F.regression_metrics(yva, pred)
        res.append(dict(n_neurons=k, n_features=k * n_windows,
                        n_params=k * n_windows + 1, **m))
    return res


# --------------------------------------------------------------- Stage 2 + freeze
def stage2_train_and_freeze(ds, splits, *, n_windows=4, target_offset=0,
                            n_neurons=207, alphas=(1, 10, 30, 100, 300)):
    """Train ridge, pick alpha on val, evaluate on test, freeze the model."""
    tr = splits["train"]; va = splits["val"]; te = splits["test"]
    n_sel = int(ds["n_sel"]); k = min(n_neurons, n_sel)
    fidx = np.arange(k)
    Xtr, ytr, _ = F.build_matrix(ds, tr, n_windows=n_windows, feature_idx=fidx,
                                 target_offset=target_offset)
    Xva, yva, _ = F.build_matrix(ds, va, n_windows=n_windows, feature_idx=fidx,
                                 target_offset=target_offset)
    Xte, yte, ete = F.build_matrix(ds, te, n_windows=n_windows, feature_idx=fidx,
                                   target_offset=target_offset)
    mu, sd = F.fit_normalizer(Xtr)
    Xtrn, Xvan, Xten = (F.apply_normalizer(Xtr, mu, sd),
                        F.apply_normalizer(Xva, mu, sd),
                        F.apply_normalizer(Xte, mu, sd))
    # pick alpha on val
    val_scores = []
    for al in alphas:
        w, b = F.ridge_fit(Xtrn, ytr, alpha=al)
        m = F.regression_metrics(yva, F.ridge_predict(Xvan, w, b))
        val_scores.append((al, m["corr"], m))
    best_alpha = max(val_scores, key=lambda z: z[1])[0]
    # final fit on train (val kept separate; could fold in, we keep it clean)
    w, b = F.ridge_fit(Xtrn, ytr, alpha=best_alpha)
    test_pred = F.ridge_predict(Xten, w, b)
    # tanh is applied at inference by LinearBridgeModel; report both raw & tanh
    test_m_raw = F.regression_metrics(yte, test_pred)
    test_m_tanh = F.regression_metrics(yte, np.tanh(test_pred))
    val_m = F.regression_metrics(yva, F.ridge_predict(Xvan, w, b))

    # L/C/R confusion on the test set (by predicted sign vs group)
    groups = np.array([str(g) for g in ds["groups"]])
    ep_group = groups[ete]
    def rate(g, sgn):
        mask = ep_group == g
        if not mask.any():
            return None
        return round(float((np.sign(np.tanh(test_pred[mask])) == sgn).mean()), 3)
    confusion = dict(
        left_pred_left=rate("left", -1), right_pred_right=rate("right", +1),
        center_pred_neutralish=round(float((np.abs(np.tanh(test_pred[ep_group == "center"])) < 0.5).mean()), 3)
        if (ep_group == "center").any() else None,
    )

    # freeze the model (features indexed 0..k-1 of the manifest order)
    model = LinearBridgeModel(w=w, b=b, mu=mu, sd=sd, feature_order="newest_first")
    meta = dict(n_windows=n_windows, target_offset=target_offset,
                n_neurons=k, best_alpha=best_alpha,
                sel_body=ds["sel_body"][:k].tolist(),
                sel_graph=ds["sel_graph"][:k].tolist(),
                n_params=model.n_params)
    return dict(model=model, meta=meta, best_alpha=best_alpha,
                val=val_m, test_raw=test_m_raw, test_tanh=test_m_tanh,
                confusion=confusion, n_params=model.n_params)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="dataset_main.npz")
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--freeze", action="store_true", help="save the frozen model")
    a = p.parse_args()

    ds = load_dataset(a.dataset)
    splits = episode_splits(ds)
    rep = split_report(ds, splits)
    print("[splits]", json.dumps({k: v["n_episodes"] for k, v in rep.items()}))

    out = dict(dataset=a.dataset, n_windows=a.n_windows, splits=rep)

    print("\n=== Stage 1: LEFT/RIGHT sanity ===")
    out["stage1"] = stage1_sanity(ds, splits, n_windows=a.n_windows)
    print(json.dumps(out["stage1"], indent=2))

    print("\n=== Alignment sweep (train/val only) ===")
    out["alignment"] = alignment_sweep(ds, splits, n_windows=a.n_windows)
    print(json.dumps(out["alignment"], indent=2))
    best_off = out["alignment"]["best_offset_windows"]

    print("\n=== Feature-count sweep (val) ===")
    out["feature_sweep"] = feature_count_sweep(ds, splits, n_windows=a.n_windows,
                                               target_offset=best_off)
    for r in out["feature_sweep"]:
        print(f"  {r['n_neurons']:>4} neurons  corr={r['corr']:.3f} "
              f"mae={r['mae']:.3f} dir_acc={r['signed_dir_acc']:.3f} "
              f"({r['n_params']} params)")

    print("\n=== Stage 2: continuous regression + freeze ===")
    s2 = stage2_train_and_freeze(ds, splits, n_windows=a.n_windows,
                                 target_offset=best_off, n_neurons=207)
    out["stage2"] = dict(best_alpha=s2["best_alpha"], val=s2["val"],
                         test_raw=s2["test_raw"], test_tanh=s2["test_tanh"],
                         confusion=s2["confusion"], n_params=s2["n_params"],
                         meta=s2["meta"])
    print(json.dumps(out["stage2"], indent=2))

    if a.freeze:
        path = s2["model"].save(OUT / "bridge_model.npz", metadata=s2["meta"])
        out["frozen_model"] = path
        print(f"\n[frozen] saved {path}  ({s2['n_params']} trainable params)")

    (OUT / "train_offline.json").write_text(json.dumps(out, indent=2))
    print("\nsaved train_offline.json")


if __name__ == "__main__":
    main()
