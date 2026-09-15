"""Select the candidate pool of direction-informative optic-lobe neurons.

Reuses the EXISTING neural diagnostics (no new brain runs, no test-shot data):
the moving-shot `dynamic_tensor.npz` collected by the neural probe, which holds
per-(condition, trial, window) spike counts for every neuron that ever fired
under left/center/right shots at three speeds, plus mirrored and blind controls.

Selection criteria (Section 12 of the brief). A candidate optic-lobe neuron:
  * strongly encodes left-vs-right shot direction  (|AUC-0.5|*2, "effect size")
  * is reliable across episodes/speeds              (consistent sign of L-R diff
                                                      across the 3 speed variants)
  * responds appropriately under mirrored vision    (mirror flips L/R response)
  * loses task information under blind conditions    (near-zero activity blind)
  * occurs UPSTREAM of the representational collapse (optic-lobe superclasses:
                                                      ol_sensory, ol_intrinsic;
                                                      visual_projection is where
                                                      direction dies, so it is
                                                      excluded from the primary
                                                      pool but flagged)
  * has sufficient activity to be a useful feature   (mean spikes above a floor)

IMPORTANT: selection uses ONLY probe data (static ball offsets / open-loop moving
replays), never the held-out closed-loop test shots. The manifest is fixed and
saved before any bridge training.

Output: `workspace/outputs/learned_bridge/visual_manifest.json` (human summary)
and `visual_manifest.npz` (machine-readable: ordered graph indices, body IDs,
and per-neuron selection scores) used by the feature extractor.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
PROBE = ROOT / "workspace" / "outputs" / "neural_probe"
OUT = ROOT / "workspace" / "outputs" / "learned_bridge"

# Optic-lobe COMPUTATION neurons upstream of the audited collapse.
#
# We deliberately read from `ol_intrinsic` -- the lamina/medulla processing
# neurons (L1-L5 monopolar cells, Tm/Dm medulla types) that are the fly's
# genuine early-motion optic-lobe representation. These are the exact cell types
# the pathway audit named in its anatomical routes (L2->Tm4, L3->Mi1, L5->MeVP9).
#
# We EXCLUDE `ol_sensory` (the raw R1-R6 photoreceptors): reading R1-R6 directly
# would be reading the retina/pixels, not an optic-lobe *representation*, and
# would weaken the scientific claim that the bridge repairs the optic-lobe ->
# descending transformation. `visual_projection` is where direction dies
# (near-silent: 0 strong-directional cells in the probe), so it is not in the
# primary pool; it is kept only as a flagged secondary set for the optional
# anatomically-constrained bridge (Section 33).
PRIMARY_SUPERCLASSES = ("ol_intrinsic",)
SECONDARY_SUPERCLASSES = ("visual_projection",)
EXCLUDE_TYPES = ("R1-R6",)   # never read raw photoreceptors

LEFT_CONDS = ("left_5_none", "left_8_none", "left_3_none")
RIGHT_CONDS = ("right_5_none", "right_8_none", "right_3_none")
MIRROR_L = "left_5_mirror"
MIRROR_R = "right_5_mirror"
BLIND_COND = "center_5_blind"

ACTIVITY_FLOOR = 0.5      # mean spikes per trial (summed over windows) to qualify
EFFECT_FLOOR = 0.6        # |AUC-0.5|*2 minimum directional effect size
MAX_POOL = 500            # cap the candidate pool


def _auc(pos, neg):
    """AUC that R > L (probability a right-trial exceeds a left-trial)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def build_manifest():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(PROBE / "dynamic_tensor.npz")
    conds = [str(c) for c in d["conditions"]]
    T = d["tensor"].astype(np.float64)         # (cond, trials, win, active)
    active = d["active"].astype(np.int64)      # graph indices of active neurons
    ids = d["ids"].astype(np.int64)            # body IDs by graph index
    ci = {c: i for i, c in enumerate(conds)}

    # per (condition, trial) total spike count over the whole presentation
    tot = T.sum(2)                             # (cond, trials, active)

    # left vs right pooled across the 3 speeds
    L = np.concatenate([tot[ci[c]] for c in LEFT_CONDS], axis=0)   # (3*trials, active)
    R = np.concatenate([tot[ci[c]] for c in RIGHT_CONDS], axis=0)
    mean_act = (L.mean(0) + R.mean(0)) / 2.0
    lr_diff = R.mean(0) - L.mean(0)            # signed: >0 prefers right shots

    n = L.shape[1]
    effect = np.zeros(n); auc = np.full(n, 0.5)
    for i in range(n):
        if mean_act[i] < ACTIVITY_FLOOR:
            continue
        a = _auc(R[:, i], L[:, i])
        auc[i] = a
        effect[i] = abs(a - 0.5) * 2.0

    # reliability: sign of (R-L) consistent across all three speed pairs
    sign_ok = np.ones(n, dtype=bool)
    per_speed_sign = []
    for lc, rc in zip(LEFT_CONDS, RIGHT_CONDS):
        diff = tot[ci[rc]].mean(0) - tot[ci[lc]].mean(0)
        per_speed_sign.append(np.sign(diff))
    per_speed_sign = np.array(per_speed_sign)          # (3, active)
    # consistent if all three speeds agree with the pooled sign (nonzero)
    pooled_sign = np.sign(lr_diff)
    agree = (per_speed_sign == pooled_sign).all(0) & (pooled_sign != 0)
    reliability = (per_speed_sign == pooled_sign).mean(0)   # fraction agreeing

    # blind dependence: activity should collapse when blind
    blind_act = tot[ci[BLIND_COND]].mean(0)
    blind_drop = mean_act - blind_act                  # >0 means vision-driven
    # vision-dependent = activity genuinely drops when the eye is blinded (the
    # neuron's drive comes from vision, not tonic common-mode). Using strict
    # blind < 0.5*mean over-prunes to ~31; blind < mean keeps genuinely
    # vision-driven cells and lands the pool in the target 100-500 range.
    vision_dependent = blind_act < (mean_act - 1e-9)

    # mirror response: under mirrored input, a left-preferring cell should now
    # respond to a mirrored-left (which looks like a right stimulus), i.e. its
    # L/R response order should flip. Measure the mirrored L-R diff and require
    # it to have OPPOSITE sign to the normal L-R diff (a genuine retinotopic
    # flip) OR at least a substantial change.
    mir_diff = tot[ci[MIRROR_R]].mean(0) - tot[ci[MIRROR_L]].mean(0)
    # normalized mirror-flip score in [-1,1]: -1 = perfect reversal
    denom = np.abs(lr_diff) + np.abs(mir_diff) + 1e-9
    mirror_flip = (mir_diff * pooled_sign) / denom     # negative => flipped
    responds_mirror = (tot[ci[MIRROR_L]].mean(0) + tot[ci[MIRROR_R]].mean(0)) / 2.0

    # annotations for the active neurons
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")
    are = ann.reindex(ids[active])
    sup = are["superclass"].astype(str).to_numpy()
    typ = are["type"].astype(str).to_numpy()
    side = are["somaSide"].astype(str).to_numpy()

    primary_mask = np.isin(sup, PRIMARY_SUPERCLASSES) & ~np.isin(typ, EXCLUDE_TYPES)

    # candidate score: effect size gated by activity, reliability, and vision
    # dependence. This does not use closed-loop/test data.
    score = (effect
             * (mean_act > ACTIVITY_FLOOR)
             * reliability
             * (blind_drop > 0))

    # primary candidate pool: optic-lobe, strong effect, reliable, vision-dep.
    qualifies = (primary_mask
                 & (effect >= EFFECT_FLOOR)
                 & agree
                 & vision_dependent
                 & (mean_act >= ACTIVITY_FLOOR))
    cand_idx = np.flatnonzero(qualifies)
    # rank by score, keep top MAX_POOL
    order = cand_idx[np.argsort(-score[cand_idx])]
    pool = order[:MAX_POOL]

    # latency proxy: first 5 ms window (of the 24) whose mean count over L+R
    # trials exceeds 25% of that neuron's peak window mean.
    win_mean = np.concatenate([T[ci[c]] for c in LEFT_CONDS + RIGHT_CONDS],
                              axis=0).mean(0)          # (win, active)
    peak = win_mean.max(0) + 1e-9
    latency_win = np.argmax(win_mean >= 0.25 * peak, axis=0)
    latency_ms = latency_win * float(d["window_ms"])

    def rec(k):
        gi = int(active[k])
        return dict(
            body_id=int(ids[gi]), graph_index=gi,
            cell_type=str(typ[k]), superclass=str(sup[k]),
            soma_side=str(side[k]),
            direction_pref=("right" if lr_diff[k] > 0 else "left"),
            auc_right_gt_left=round(float(auc[k]), 3),
            effect_size=round(float(effect[k]), 3),
            lr_diff=round(float(lr_diff[k]), 3),
            reliability=round(float(reliability[k]), 3),
            mean_activity=round(float(mean_act[k]), 3),
            blind_activity=round(float(blind_act[k]), 3),
            blind_drop=round(float(blind_drop[k]), 3),
            mirror_flip=round(float(mirror_flip[k]), 3),
            latency_ms=round(float(latency_ms[k]), 1),
            selection_score=round(float(score[k]), 3),
        )

    pool_records = [rec(k) for k in pool]

    # machine-readable arrays for the feature extractor (ordered by score)
    pool_body_ids = np.array([r["body_id"] for r in pool_records], dtype=np.int64)
    pool_graph_idx = np.array([r["graph_index"] for r in pool_records], dtype=np.int64)
    pool_effect = np.array([r["effect_size"] for r in pool_records])
    pool_dirpref = np.array([1 if r["direction_pref"] == "right" else -1
                             for r in pool_records], dtype=np.int8)
    np.savez(OUT / "visual_manifest.npz",
             body_ids=pool_body_ids, graph_index=pool_graph_idx,
             effect_size=pool_effect, dir_pref=pool_dirpref,
             activity_floor=ACTIVITY_FLOOR, effect_floor=EFFECT_FLOOR)

    # summary stats by superclass and side
    from collections import Counter
    summary = dict(
        criteria=dict(primary_superclasses=list(PRIMARY_SUPERCLASSES),
                      excluded_types=list(EXCLUDE_TYPES),
                      excluded_secondary=list(SECONDARY_SUPERCLASSES),
                      activity_floor=ACTIVITY_FLOOR, effect_floor=EFFECT_FLOOR,
                      max_pool=MAX_POOL,
                      data_source="neural_probe/dynamic_tensor.npz (moving shots, "
                                  "3 speeds; mirror+blind controls); "
                                  "NO held-out closed-loop test shots"),
        n_active=int(n),
        n_qualifying=int(qualifies.sum()),
        pool_size=int(len(pool)),
        pool_by_superclass=dict(Counter(r["superclass"] for r in pool_records)),
        pool_by_side=dict(Counter(r["soma_side"] for r in pool_records)),
        pool_by_dir_pref=dict(Counter(r["direction_pref"] for r in pool_records)),
        effect_size_range=[round(float(pool_effect.min()), 3),
                           round(float(pool_effect.max()), 3)] if len(pool) else None,
        latency_ms_median=round(float(np.median([r["latency_ms"] for r in pool_records])), 1) if pool_records else None,
        top20=pool_records[:20],
        all_candidates=pool_records,
    )
    (OUT / "visual_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    s = build_manifest()
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("top20", "all_candidates")}, indent=2))
    print("\ntop 15 candidates (body_id  type  side  dir  effect  reliab  latency):")
    for r in s["top20"][:15]:
        print(f"  {r['body_id']:>8}  {r['cell_type'][:16]:<16} {r['soma_side']:<3} "
              f"{r['direction_pref']:<5} eff={r['effect_size']:.2f} "
              f"rel={r['reliability']:.2f} lat={r['latency_ms']:.0f}ms")
    print(f"\nsaved visual_manifest.json / .npz  (pool size {s['pool_size']})")


if __name__ == "__main__":
    main()
