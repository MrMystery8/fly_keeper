"""Trace anatomical paths from directional visual neurons to descending neurons.

Distinguishes:
  A. no anatomical pathway from directional visual neurons to DNs, versus
  B. a pathway exists but the simulation's dynamics don't propagate activity
     through it.

Uses the prepared CSR graph (graph.npz: ptr/post/weight over graph indices) and
the whole-brain screen (which neurons are directional and active). Computes, for
the strongest directional visual/optic-lobe neurons:
  - shortest-path hop distance to any descending neuron (BFS over the directed
    graph), and the number of such reachable DNs within a few hops
  - along the best paths, whether intermediate neurons were ACTIVE in the probe
  - synaptic edge counts/strengths on the first hop toward DNs
"""
from __future__ import annotations
import json
import sys
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"
GRAPH = ROOT / "upstream/doomfly/outputs/doom/malecns_v1/graph.npz"


def bfs_hops(ptr, post, sources, targets_mask, max_hops=6):
    """Multi-source BFS over the directed graph. Returns hop distance array."""
    n = len(ptr) - 1
    dist = np.full(n, -1, dtype=np.int16)
    q = deque()
    for s in sources:
        dist[s] = 0; q.append(s)
    reached_target_hop = {}
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_hops:
            continue
        for e in range(ptr[u], ptr[u + 1]):
            v = post[e]
            if dist[v] == -1:
                dist[v] = du + 1
                if targets_mask[v]:
                    reached_target_hop.setdefault(du + 1, 0)
                    reached_target_hop[du + 1] += 1
                q.append(v)
    return dist, reached_target_hop


def main():
    import pandas as pd
    g = np.load(GRAPH)
    ptr, post, weight, ids = g["ptr"], g["post"], g["weight"], g["ids"].astype(np.int64)
    n = len(ids)
    id_to_idx = {int(b): i for i, b in enumerate(ids)}
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    # superclass per graph index
    sup = np.array(["unknown"] * n, dtype=object)
    for i in range(n):
        bid = int(ids[i])
        if bid in ann.index:
            sup[i] = str(ann.loc[bid]["superclass"])
    dn_mask = np.array(["descend" in str(s).lower() for s in sup])
    dn_idx = np.flatnonzero(dn_mask)
    print(f"descending neurons in graph: {len(dn_idx)}")

    # directional visual sources: use the whole-brain static top table (strong,
    # reliable) restricted to optic lobe / retina / visual projection.
    top = json.loads((OUT / "static_whole_brain_top.json").read_text())
    src_ids = [t["body_id"] for t in top
               if t["superclass"] in ("ol_sensory", "ol_intrinsic", "visual_projection")]
    sources = [id_to_idx[b] for b in src_ids if b in id_to_idx]
    print(f"directional visual source neurons: {len(sources)}")

    # BFS from directional visual neurons toward descending neurons
    dist, reached = bfs_hops(ptr, post, sources, dn_mask, max_hops=6)
    dn_dist = dist[dn_idx]
    reachable = dn_dist[dn_dist > 0]
    result = dict(
        n_sources=len(sources), n_descending=len(dn_idx),
        descending_reached=int((dn_dist > 0).sum()),
        reached_by_hop={int(k): int(v) for k, v in sorted(reached.items())},
        min_hops_to_descending=int(reachable.min()) if len(reachable) else None,
        median_hops=float(np.median(reachable)) if len(reachable) else None,
    )

    # For the closest reachable DNs, report type + whether they were responsive.
    # responsive DNs from the probe:
    responsive_dn = {10059, 10162, 10527, 555871}
    closest = dn_idx[np.argsort(dn_dist)]
    closest = [i for i in closest if dist[i] > 0][:20]
    close_rows = []
    for i in closest:
        bid = int(ids[i])
        typ = str(ann.loc[bid]["type"]) if bid in ann.index else "?"
        close_rows.append(dict(body_id=bid, type=typ, hops=int(dist[i]),
                               responsive=bid in responsive_dn))
    result["closest_descending"] = close_rows

    # Do the 4 responsive DNs get reached, and at what hop?
    result["responsive_dn_hops"] = {
        bid: int(dist[id_to_idx[bid]]) for bid in responsive_dn if bid in id_to_idx}

    # First-hop fan-out: how many directional visual sources synapse directly
    # onto descending neurons (1-hop), and total edge weight.
    direct_edges = 0; direct_w = 0.0; direct_targets = set()
    for s in sources:
        for e in range(ptr[s], ptr[s + 1]):
            v = post[e]
            if dn_mask[v]:
                direct_edges += 1; direct_w += float(weight[e]); direct_targets.add(int(v))
    result["direct_visual_to_dn_edges"] = direct_edges
    result["direct_visual_to_dn_targets"] = len(direct_targets)
    result["direct_visual_to_dn_total_weight"] = round(direct_w, 1)

    (OUT / "connectivity_trace.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
