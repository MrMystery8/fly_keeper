"""Build a BILATERAL candidate neuron pool for the binocular Arcade bridge.

The frozen 207-neuron set is anatomically ~206 right / 1 left somaSide -- fine for
the one-eye science experiment, but a poor basis for genuinely bilateral control.
For the binocular experiment we build a candidate pool that is represented on BOTH
sides, then let feature selection (from TRAINING DATA ONLY, in binocular_train.py)
choose the final features. We do NOT hand-pick the final set here.

Construction (connectome-grounded, no data peeking):
  * take the cell TYPES of the frozen 207 optic-lobe neurons;
  * collect ALL neurons of those types on BOTH somaSides (L and R) that exist in
    the loaded graph;
  * that is the candidate pool (bilateral by construction, opponent L/R structure
    available because most types have both an L and an R member).

Writes binocular_pool.npz: body_ids, somaSide, type for the pool.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

import pandas as pd

from adapters.brain import MaleCNSBrain

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def build_pool(max_pool=600):
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    frozen207 = [int(x) for x in man["body_ids"][:207]]

    ann = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/"
                          "malecns_v1/annotations.feather")
    id2type = dict(zip(ann["bodyId"].astype(int), ann["type"].astype(str)))
    id2side = dict(zip(ann["bodyId"].astype(int), ann["somaSide"].astype(str)))

    # cell types spanned by the frozen 207
    types = sorted({id2type.get(b, "?") for b in frozen207} - {"?", "None", "nan"})

    brain = MaleCNSBrain(backend="cpu")
    graph_ids = set(int(x) for x in brain._brain.ids)
    brain.close()

    # all neurons of those types, both sides, present in the graph
    pool = ann[ann["type"].astype(str).isin(types)]
    cand = []
    for b, s in zip(pool["bodyId"].astype(int), pool["somaSide"].astype(str)):
        if int(b) in graph_ids and s in ("L", "R"):
            cand.append((int(b), s, id2type.get(int(b), "?")))
    # de-dup, stable order: frozen 207 first (so parity with v2 is possible), then
    # the rest sorted by (type, side, id) for determinism
    seen = set()
    ordered = []
    for b in frozen207:
        if b in graph_ids and b not in seen:
            ordered.append((b, id2side.get(b, "?"), id2type.get(b, "?")))
            seen.add(b)
    for b, s, t in sorted(cand, key=lambda x: (x[2], x[1], x[0])):
        if b not in seen:
            ordered.append((b, s, t)); seen.add(b)
    if len(ordered) > max_pool:
        # keep all frozen207 + fill remaining slots balanced across sides
        head = ordered[:207]
        rest = ordered[207:]
        rest_sorted = sorted(rest, key=lambda x: (x[1] != "L", x[2], x[0]))  # favor L to balance
        ordered = head + rest_sorted[:max_pool - 207]

    body_ids = np.array([o[0] for o in ordered], dtype=np.int64)
    sides = np.array([o[1] for o in ordered], dtype="U4")
    ptypes = np.array([o[2] for o in ordered], dtype="U64")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "binocular_pool.npz", body_ids=body_ids, somaSide=sides,
             type=ptypes)
    summary = dict(pool_size=int(len(body_ids)), n_types=len(types),
                   somaSide=dict(Counter(sides.tolist())),
                   n_frozen207_included=int(sum(b in set(frozen207)
                                                for b in body_ids.tolist())),
                   max_pool=max_pool)
    (OUT / "binocular_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-pool", type=int, default=600)
    a = p.parse_args()
    build_pool(max_pool=a.max_pool)
