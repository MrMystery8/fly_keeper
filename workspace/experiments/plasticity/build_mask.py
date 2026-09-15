"""Build the explicit plasticity mask for the visual -> projection -> descending
pathway. ONLY synapses on this route become mutable; everything else is frozen.

Target descending neurons (opponent pairs that (a) fire under visual drive and
(b) were shown last phase to move the body correctly/symmetrically):
    DNp20   : 10162 (L) / 10059 (R)
    DNpe017 : 10527 (L) / 555871 (R)
We additionally include DNp11 (the DN most connected to strongly directional
visual sources in the audit) as a candidate steering opponent pair, if present.

Mask construction (over graph indices, using the CSR graph):
  Stage "mid->DN"     : every incoming edge to a target DN whose presynaptic
                        neuron is in the optic lobe / visual projection / central
                        brain (i.e. plausible pre-descending intermediates).
  Stage "source->mid" : every incoming edge to those intermediate neurons whose
                        presynaptic neuron is a directional visual neuron
                        (ol_sensory / ol_intrinsic / visual_projection).

For every plastic synapse we record edge index, pre/post body IDs, annotations,
sides, baseline weight, and pathway stage. Output:
  workspace/outputs/plasticity/plasticity_mask.npz   (machine-readable)
  workspace/outputs/plasticity/pathway_table.json    (human-readable summary)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "plasticity"
GRAPH = ROOT / "upstream/doomfly/outputs/doom/malecns_v1/graph.npz"

TARGET_DN = {
    10162: ("DNp20", "L"), 10059: ("DNp20", "R"),
    10527: ("DNpe017", "L"), 555871: ("DNpe017", "R"),
}
# candidate additional opponent DN types to include if present (by type/side)
EXTRA_DN_TYPES = {"DNp11"}

VISUAL_SUP = {"ol_sensory", "ol_intrinsic", "visual_projection"}
INTERMEDIATE_SUP = {"ol_intrinsic", "visual_projection", "visual_centrifugal"}
# intermediates that may feed a DN: optic-lobe / visual-projection / central
PRE_DN_SUP = {"ol_intrinsic", "visual_projection", "visual_centrifugal",
              "cb_intrinsic", "central_brain"}


def incoming_edges(ptr, post, target_idx_set):
    """Return edge indices whose POST (target) is in target_idx_set, plus their
    source neuron index. Vectorized over the edge arrays."""
    mask = np.isin(post, np.fromiter(target_idx_set, dtype=np.int64,
                                     count=len(target_idx_set)))
    eidx = np.flatnonzero(mask)
    src = np.searchsorted(ptr, eidx, side="right") - 1
    return eidx, src


def load_directional_sources():
    """Body IDs of strongly directional visual neurons from the pathway audit's
    directional-strength screen (|AUC-0.5|*2 >= threshold)."""
    import numpy as np
    from experiments.pathway_audit.rank_pathways import directional_strength
    ids_active, active, strength, lr, act = directional_strength()
    # active indexes into the probe tensor's active set; map back to body IDs
    strong_local = active[strength >= 0.6]
    return set(int(ids_active[a]) for a in strong_local)


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
    are = ann.reindex(ids)
    sup = are["superclass"].astype(str).to_numpy().astype("U32")
    typ = are["type"].astype(str).to_numpy().astype("U32")
    side = are["somaSide"].astype(str).to_numpy().astype("U4")

    # restrict sources to the strongly DIRECTIONAL visual neurons (not all OL)
    directional_ids = load_directional_sources()
    directional_idx = set(id_to_idx[b] for b in directional_ids if b in id_to_idx)
    print(f"strongly directional visual source neurons: {len(directional_idx)}")
    MIN_EDGE_W = 1.0  # ignore negligible synapses to keep the mask focused

    # resolve target DN indices (named pairs + extra types present in graph)
    dn_idx = set()
    for bid in TARGET_DN:
        if bid in id_to_idx:
            dn_idx.add(id_to_idx[bid])
    for i in range(n):
        if typ[i] in EXTRA_DN_TYPES and "descend" in sup[i].lower():
            dn_idx.add(i)
    print(f"target DNs: {len(dn_idx)}")

    # STAGE mid->DN: incoming edges to target DNs from pre-DN populations,
    # keeping only reasonably strong synapses.
    eidx_md, src_md = incoming_edges(ptr, post, dn_idx)
    keep_md = np.array([sup[s] in PRE_DN_SUP and weight[e] >= MIN_EDGE_W
                        for e, s in zip(eidx_md, src_md)])
    eidx_md, src_md = eidx_md[keep_md], src_md[keep_md]
    mid_idx = set(int(s) for s in src_md)
    print(f"mid->DN plastic edges: {len(eidx_md)}, distinct intermediates: {len(mid_idx)}")

    # STAGE source->mid: incoming edges to those intermediates, but ONLY from the
    # strongly DIRECTIONAL visual source neurons (not the whole optic lobe) and
    # only reasonably strong synapses. This keeps the mask focused on the route
    # that actually carries direction.
    eidx_sm, src_sm = incoming_edges(ptr, post, mid_idx)
    keep_sm = np.array([int(s) in directional_idx and weight[e] >= MIN_EDGE_W
                        for e, s in zip(eidx_sm, src_sm)])
    eidx_sm, src_sm = eidx_sm[keep_sm], src_sm[keep_sm]
    print(f"source->mid plastic edges (directional sources only): {len(eidx_sm)}")

    # assemble mask
    all_eidx = np.concatenate([eidx_md, eidx_sm])
    all_src = np.concatenate([src_md, src_sm])
    all_stage = np.array(["mid_to_dn"] * len(eidx_md) + ["source_to_mid"] * len(eidx_sm))
    # dedupe edges (an edge could appear once; stages are disjoint by target set)
    _, uniq = np.unique(all_eidx, return_index=True)
    all_eidx, all_src, all_stage = all_eidx[uniq], all_src[uniq], all_stage[uniq]
    all_post = post[all_eidx]
    baseline_w = weight[all_eidx].astype(np.float32)

    np.savez_compressed(
        OUT / "plasticity_mask.npz",
        edge_index=all_eidx.astype(np.int64),
        pre_index=all_src.astype(np.int64),
        post_index=all_post.astype(np.int64),
        pre_body_id=ids[all_src], post_body_id=ids[all_post],
        stage=all_stage, baseline_weight=baseline_w,
    )

    # human-readable summary
    from collections import Counter
    def summarize(eidx, src):
        pre_sup = Counter(sup[s] for s in src)
        post_types = Counter(typ[post[e]] for e in eidx)
        return dict(n_edges=int(len(eidx)),
                    pre_superclass=dict(pre_sup.most_common(8)),
                    post_types=dict(post_types.most_common(10)),
                    weight_mean=round(float(weight[eidx].mean()), 3),
                    weight_max=round(float(weight[eidx].max()), 3))
    table = dict(
        target_dn={f"{TARGET_DN.get(int(ids[i]), (typ[i], side[i]))}": int(ids[i])
                   for i in dn_idx},
        n_target_dn=len(dn_idx),
        n_intermediates=len(mid_idx),
        stage_mid_to_dn=summarize(eidx_md, src_md),
        stage_source_to_mid=summarize(eidx_sm, src_sm),
        total_plastic_synapses=int(len(all_eidx)),
        total_synapses_in_graph=int(len(post)),
        fraction_plastic=round(len(all_eidx) / len(post), 6),
    )
    (OUT / "pathway_table.json").write_text(json.dumps(table, indent=2, default=str))
    print(json.dumps(table, indent=2, default=str))
    print(f"\nplastic synapses: {len(all_eidx)} / {len(post)} "
          f"({100*len(all_eidx)/len(post):.4f}% of graph)")


if __name__ == "__main__":
    main()
