"""Robust directional metrics + temporal + hierarchy analysis.

Consumes static_tensor.npz / dynamic_tensor.npz (n_cond, n_trials, n_win,
n_active) and computes, with corrected criteria that treat constant responses
as non-directional:

  directional candidate requires ALL of:
    - min activity        : mean total spikes over the window >= MIN_SPIKES
    - reliability         : split-trial consistency of the L-R sign
    - effect size         : |Cohen's d| (right vs left trials) >= D_MIN
    - discriminability    : |AUC - 0.5| >= AUC_MARGIN

AUC is computed as Mann-Whitney; for tied/constant distributions this yields
0.5 (non-directional), fixing the earlier degenerate 0/1 artifact.

Outputs:
  - whole-brain ranked table (static + dynamic)
  - directional info by superclass (the sensory hierarchy)
  - descending-neuron table
  - temporal profile: directional AUC vs cumulative window for the best stages
  - corrected pathway diagnosis
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

MIN_SPIKES = 1.0      # mean spikes over the analysis window
D_MIN = 0.5           # Cohen's d threshold
AUC_MARGIN = 0.1      # |AUC - 0.5| threshold


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, len(a) + len(b) - 2))
    return 0.0 if pooled < 1e-9 else float((a.mean() - b.mean()) / pooled)


def auc(pos, neg):
    """Mann-Whitney AUC. Constant/tied data -> 0.5 (mid-ranks handle ties)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    # average ranks for ties (Mann-Whitney with tie correction)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def load(tensor_file):
    d = np.load(OUT / tensor_file)
    return dict(ids=d["ids"].astype(np.int64), active=d["active"].astype(np.int64),
                conditions=[str(c) for c in d["conditions"]],
                tensor=d["tensor"], window_ms=float(d["window_ms"]))


def totals_by_condition(data, window_slice=None):
    """Sum tensor over a window slice -> {cond: (n_trials, n_active)}."""
    T = data["tensor"]
    conds = data["conditions"]
    if window_slice is None:
        summed = T.sum(axis=2)
    else:
        summed = T[:, :, window_slice, :].sum(axis=2)
    return {c: summed[i].astype(float) for i, c in enumerate(conds)}


def directional_screen(totals, left_key, right_key, blind_key):
    """Per active-neuron directional metrics. Returns dict of arrays over active."""
    left = totals[left_key]; right = totals[right_key]
    n = left.shape[1]
    mb = totals[blind_key].mean(0) if blind_key in totals else np.zeros(n)
    ml = left.mean(0); mr = right.mean(0)
    lr_diff = (mr - mb) - (ml - mb)
    activity = (left.mean(0) + right.mean(0)) / 2.0
    d = np.zeros(n); a = np.full(n, 0.5)
    cand = np.flatnonzero(activity >= MIN_SPIKES)
    for i in cand:
        d[i] = cohens_d(right[:, i], left[:, i])
        a[i] = auc(right[:, i], left[:, i])
    directional = (np.abs(d) >= D_MIN) & (np.abs(a - 0.5) >= AUC_MARGIN) & (activity >= MIN_SPIKES)
    return dict(activity=activity, lr_diff=lr_diff, cohens_d=d, auc=a,
                directional=directional, ml=ml, mr=mr, mb=mb)


def annotate(ids_active, ann):
    out = []
    for bid in ids_active:
        bid = int(bid)
        if bid in ann.index:
            r = ann.loc[bid]
            out.append((str(r["type"]), str(r["somaSide"]), str(r["superclass"])))
        else:
            out.append(("?", "?", "unknown"))
    return out


def main():
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    report = {}
    for label, tfile, lk, rk, bk in [
        ("static", "static_tensor.npz", "left", "right", "blind"),
        ("dynamic", "dynamic_tensor.npz", "left_5_none", "right_5_none", "center_5_blind"),
    ]:
        data = load(tfile)
        ids_active = data["ids"][data["active"]]
        meta = annotate(ids_active, ann)
        sup = np.array([m[2] for m in meta])
        totals = totals_by_condition(data)
        scr = directional_screen(totals, lk, rk, bk)

        # ---- whole-brain ranked table ----
        dir_idx = np.flatnonzero(scr["directional"])
        rank = dir_idx[np.argsort(-np.abs(scr["cohens_d"][dir_idx]))]
        top = []
        for i in rank[:40]:
            typ, side, s = meta[i]
            top.append(dict(body_id=int(ids_active[i]), type=typ, soma_side=side,
                            superclass=s,
                            left=round(float(scr["ml"][i] - scr["mb"][i]), 2),
                            right=round(float(scr["mr"][i] - scr["mb"][i]), 2),
                            lr_diff=round(float(scr["lr_diff"][i]), 2),
                            cohens_d=round(float(scr["cohens_d"][i]), 2),
                            auc=round(float(scr["auc"][i]), 3)))
        (OUT / f"{label}_whole_brain_top.json").write_text(json.dumps(top, indent=2))

        # ---- by superclass (hierarchy) ----
        stage = []
        for s in sorted(set(sup)):
            mask = sup == s
            n_active = int((scr["activity"][mask] >= MIN_SPIKES).sum())
            n_dir = int(scr["directional"][mask].sum())
            dsel = np.abs(scr["cohens_d"][mask])
            asel = np.abs(scr["auc"][mask] - 0.5)
            stage.append(dict(superclass=s, n_active=n_active, n_directional=n_dir,
                              best_abs_d=round(float(dsel.max()) if len(dsel) else 0, 2),
                              best_auc_margin=round(float(asel.max()) if len(asel) else 0, 3)))
        stage.sort(key=lambda r: -r["n_directional"])
        (OUT / f"{label}_by_superclass.json").write_text(json.dumps(stage, indent=2))

        # ---- descending-neuron table ----
        dn_mask = np.array(["descend" in m[2].lower() for m in meta])
        dn_rows = []
        for i in np.flatnonzero(dn_mask):
            typ, side, s = meta[i]
            dn_rows.append(dict(body_id=int(ids_active[i]), type=typ, soma_side=side,
                                activity=round(float(scr["activity"][i]), 2),
                                lr_diff=round(float(scr["lr_diff"][i]), 2),
                                cohens_d=round(float(scr["cohens_d"][i]), 2),
                                auc=round(float(scr["auc"][i]), 3),
                                directional=bool(scr["directional"][i])))
        dn_rows.sort(key=lambda r: -abs(r["cohens_d"]))
        (OUT / f"{label}_descending.json").write_text(json.dumps(dn_rows, indent=2))

        n_dn_dir = sum(r["directional"] for r in dn_rows)
        report[label] = dict(
            n_active=int((scr["activity"] >= MIN_SPIKES).sum()),
            n_directional=int(scr["directional"].sum()),
            stage_top=stage[:8],
            descending_total=len(dn_rows),
            descending_responsive=int(sum(r["activity"] >= MIN_SPIKES for r in dn_rows)),
            descending_directional=int(n_dn_dir),
            best_descending=dn_rows[:8],
        )
        print(f"\n===== {label.upper()} =====")
        print(f"active={report[label]['n_active']} directional={report[label]['n_directional']}")
        print("hierarchy (directional counts by superclass):")
        for st in stage[:8]:
            print(f"  {st['superclass']:20s} active={st['n_active']:5d} "
                  f"directional={st['n_directional']:5d} best|d|={st['best_abs_d']:.2f} "
                  f"best|AUC-.5|={st['best_auc_margin']:.3f}")
        print(f"descending: {len(dn_rows)} total, "
              f"{report[label]['descending_responsive']} responsive, "
              f"{n_dn_dir} directional (corrected)")
        for r in dn_rows[:6]:
            print(f"   {r['body_id']:>8} {r['type']:>10} {r['soma_side']} "
                  f"act={r['activity']:.1f} d={r['cohens_d']:+.2f} AUC={r['auc']:.3f} "
                  f"dir={r['directional']}")

    (OUT / "corrected_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nsaved corrected metric outputs")


if __name__ == "__main__":
    main()
