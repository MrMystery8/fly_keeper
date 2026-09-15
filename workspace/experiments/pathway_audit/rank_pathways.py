"""Rank candidate visual -> intermediate -> descending routes.

Combines:
  - per-neuron directional strength (|AUC-0.5| for left vs right) from the
    static probe tensor, so we prefer routes whose SOURCE actually carries
    direction, and
  - the anatomical graph (CSR ptr/post/weight) to enumerate short (<=3 hop)
    paths from strong directional visual neurons to descending neurons,
    recording intermediates, edge weights (synaptic contact counts), and
    laterality.

Output: a manageable ranked candidate-pathway set with body IDs, annotations,
laterality and connection strengths, saved to
workspace/outputs/pathway_audit/candidate_pathways.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "pathway_audit"
GRAPH = ROOT / "upstream/doomfly/outputs/doom/malecns_v1/graph.npz"


def auc(pos, neg):
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


def directional_strength(tensor_file="static_tensor.npz"):
    """|AUC-0.5|*2 for left-vs-right per active neuron, plus signed lr_diff."""
    npz = ROOT / "workspace" / "outputs" / "neural_probe" / tensor_file
    d = np.load(npz)
    conds = [str(c) for c in d["conditions"]]
    T = d["tensor"]  # (cond, trials, win, active)
    tot = T.sum(2)   # (cond, trials, active) total spikes over presentation
    ci = {c: i for i, c in enumerate(conds)}
    L = tot[ci["left"]].astype(float); R = tot[ci["right"]].astype(float)
    n = L.shape[1]
    strength = np.zeros(n); lr = np.zeros(n); act = (L.mean(0) + R.mean(0)) / 2
    for i in range(n):
        if act[i] < 0.5:
            continue
        a = auc(R[:, i], L[:, i])
        strength[i] = abs(a - 0.5) * 2
        lr[i] = R[:, i].mean() - L[:, i].mean()
    return d["ids"].astype(np.int64), d["active"].astype(np.int64), strength, lr, act


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    g = np.load(GRAPH)
    ptr, post, weight, ids = g["ptr"], g["post"], g["weight"], g["ids"].astype(np.int64)
    n = len(ids)
    id_to_idx = {int(b): i for i, b in enumerate(ids)}

    # superclass/type/side per graph index (vectorized reindex by body ID)
    are = ann.reindex(ids)
    sup = are["superclass"].astype(str).to_numpy()
    typ = are["type"].astype(str).to_numpy()
    side = are["somaSide"].astype(str).to_numpy()

    # directional strength mapped to graph indices
    tids, active, strength, lr, act = directional_strength()
    strength_full = np.zeros(n); lr_full = np.zeros(n)
    for k, a in enumerate(active):
        strength_full[a] = strength[k]; lr_full[a] = lr[k]

    dn_mask = np.array(["descend" in str(s).lower() for s in sup])
    dn_idx = np.flatnonzero(dn_mask)
    responsive_dn = [10059, 10162, 10527, 555871]

    # strong directional visual sources
    visual_mask = np.array([s in ("ol_sensory", "ol_intrinsic", "visual_projection")
                            for s in sup])
    src_idx = np.flatnonzero(visual_mask & (strength_full >= 0.8))
    print(f"strong directional visual sources (strength>=0.8): {len(src_idx)}")

    # Build reverse adjacency for tracing backward from DNs is expensive; instead
    # do forward: for each DN, find which strong sources reach it in <=2 hops and
    # via which intermediates, tracking edge weights.
    # Precompute out-neighbors set for sources for hop-1.
    src_set = set(int(x) for x in src_idx)

    # neurons that project directly to any DN (1-hop predecessors of DNs).
    # Vectorized: find edges whose TARGET is a DN, then map each edge to its
    # source neuron via searchsorted on ptr.
    dn_target_edges = np.flatnonzero(dn_mask[post])          # edge indices -> DN
    src_of_edge = np.searchsorted(ptr, dn_target_edges, side="right") - 1
    projects_to_dn = {}
    for eidx, u in zip(dn_target_edges, src_of_edge):
        projects_to_dn.setdefault(int(u), []).append((int(post[eidx]), float(weight[eidx])))
    print(f"neurons projecting directly onto a DN: {len(projects_to_dn)}")

    # Now: source -> mid (mid in projects_to_dn) -> DN
    routes = []
    proj_keys = set(projects_to_dn.keys())
    for s in src_idx:
        a, b = ptr[s], ptr[s + 1]
        for k in range(a, b):
            mid = int(post[k]); w_sm = float(weight[k])
            if mid in proj_keys:
                for dn_t, w_md in projects_to_dn[mid]:
                    routes.append((int(s), mid, dn_t, w_sm, w_md))

    print(f"2-hop source->mid->DN routes found: {len(routes)}")

    # aggregate/rank by (source strength * min edge weight), keep opponent info
    def rec(i):
        return dict(body_id=int(ids[i]), type=str(typ[i]), side=str(side[i]),
                    superclass=str(sup[i]), dir_strength=round(float(strength_full[i]), 3),
                    lr=round(float(lr_full[i]), 2))

    scored = []
    for s, mid, dn_t, w_sm, w_md in routes:
        score = strength_full[s] * min(w_sm, w_md)
        scored.append((score, s, mid, dn_t, w_sm, w_md))
    scored.sort(key=lambda x: -x[0])

    top_routes = []
    seen = set()
    for score, s, mid, dn_t, w_sm, w_md in scored:
        key = (s, mid, dn_t)
        if key in seen:
            continue
        seen.add(key)
        top_routes.append(dict(
            score=round(float(score), 2),
            source=rec(s), intermediate=rec(mid), descending=rec(dn_t),
            w_source_mid=round(w_sm, 1), w_mid_dn=round(w_md, 1),
            dn_responsive=int(ids[dn_t]) in responsive_dn))
        if len(top_routes) >= 60:
            break

    # summary: which DNs are reachable, how many routes, responsive coverage
    dn_route_counts = {}
    for _, s, mid, dn_t, _, _ in scored:
        b = int(ids[dn_t]); dn_route_counts[b] = dn_route_counts.get(b, 0) + 1

    result = dict(
        n_strong_sources=len(src_idx),
        n_mid_projecting_to_dn=len(projects_to_dn),
        n_2hop_routes=len(routes),
        n_distinct_dn_reached=len(dn_route_counts),
        responsive_dn_route_counts={b: dn_route_counts.get(b, 0) for b in responsive_dn},
        top_routes=top_routes,
    )
    (OUT / "candidate_pathways.json").write_text(json.dumps(result, indent=2))
    print(f"\ndistinct DNs reached by strong directional sources (2 hops): {len(dn_route_counts)}")
    print("responsive-DN route counts:", result["responsive_dn_route_counts"])
    print("\ntop 12 candidate routes (source -> intermediate -> DN):")
    for r in top_routes[:12]:
        s = r["source"]; m = r["intermediate"]; dd = r["descending"]
        print(f"  [{r['score']:.1f}] {s['type']}({s['side']},dir={s['dir_strength']}) "
              f"-> {m['type']}({m['side']}) -> {dd['type']}({dd['side']}) "
              f"w={r['w_source_mid']}/{r['w_mid_dn']} resp={r['dn_responsive']}")


if __name__ == "__main__":
    main()
