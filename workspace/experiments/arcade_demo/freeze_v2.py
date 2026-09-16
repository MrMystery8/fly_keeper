"""Freeze the Arcade Bridge v2 artifact + a complete provenance manifest.

Writes arcade_bridge_v2_freeze.json capturing everything needed to reproduce or
audit the frozen v2 controller, WITHOUT touching the v1 artifacts or Science
Mode. Records: dataset manifest reference + checksum, train/val/test split seeds,
selected neuron IDs (graph_index + body_ids), temporal-window definition,
normalization checksum, weights/bias checksum + shapes, the DN mapping (lateral +
vertical real MaleCNS descending neurons), and the motor-interface scaling.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_bridge import LinearBridge2D
from experiments.arcade_demo.vertical_dn import (ArcadeDNBasis, LATERAL_LEFT_DN,
                                                 LATERAL_RIGHT_DN, VERTICAL_DN)
from experiments.arcade_demo.arcade_runtime import DECISION_MS

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def main(dataset="arcade_dataset_v2.npz", model="arcade_bridge_model_v2.npz"):
    dpath = OUT / dataset
    mpath = OUT / model
    data = np.load(dpath, allow_pickle=False)
    gi = data["graph_index"].astype(np.int64)
    n_windows = int(data["n_windows"])
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = man["body_ids"][:gi.size].astype(np.int64)

    m = LinearBridge2D.load(mpath)
    meta = getattr(m, "loaded_metadata", {})
    basis = ArcadeDNBasis()

    freeze = dict(
        model_version="arcade-bridge-v2",
        description=("Metal-based 2-output (lateral, vertical) linear Arcade "
                     "goalkeeper bridge, retrained on the CORRECTED (non-padded) "
                     "Arcade dataset. One-eye (eye_left) visual path, same 207 "
                     "optic-lobe neurons and 4x20ms causal windows as v1; only "
                     "the training DATA was corrected. Separate from frozen "
                     "Science Mode."),
        dataset=dict(
            file=dataset, sha256=_sha(dpath.read_bytes()),
            manifest_file=dataset.replace(".npz", "_manifest.json"),
            n_episodes=int(np.unique(data["seeds"]).size),
            n_samples=int(data["features"].shape[0]),
            camera="eye_left"),
        split=dict(
            seed=int(meta.get("split_seed", 20260916)),
            train_seeds=meta.get("split_seeds", {}).get("train"),
            val_seeds=meta.get("split_seeds", {}).get("val"),
            test_seeds=meta.get("split_seeds", {}).get("test"),
            method="by_complete_episode/shot_seed (no window-level split)"),
        features=dict(
            n_neurons=int(gi.size), n_windows=n_windows,
            window_ms=float(DECISION_MS), feature_dim=int(gi.size * n_windows),
            feature_order=str(m.feature_order),
            graph_index=gi.tolist(), body_ids=body.tolist(),
            note=("newest-first concatenation of the last n_windows per-window "
                  "spike-count vectors of the selected neurons, read by BODY ID "
                  "(CPU+Metal safe).")),
        normalization=dict(
            mu_sha256=_sha(m.mu.astype(np.float64).tobytes()),
            sd_sha256=_sha(m.sd.astype(np.float64).tobytes()),
            mu_shape=list(m.mu.shape), sd_shape=list(m.sd.shape),
            note="train-set mean/std; frozen with the model."),
        weights=dict(
            model_file=model, model_sha256=_sha(mpath.read_bytes()),
            W_shape=list(m.W.shape), b_shape=list(m.b.shape),
            W_sha256=_sha(m.W.astype(np.float64).tobytes()),
            b=[float(x) for x in m.b], best_alpha=int(meta.get("best_alpha", 0)),
            n_params=int(m.W.size + m.b.size),
            activation="tanh(((x-mu)/sd)@W + b); u_vert = max(0, out[1])"),
        dn_mapping=dict(
            lateral_left_dn=list(LATERAL_LEFT_DN),
            lateral_right_dn=list(LATERAL_RIGHT_DN),
            vertical_dn=list(VERTICAL_DN),
            vertical_type="DNp01 (giant-fiber / escape-takeoff) bilateral pair",
            drive_mv=basis.drive_mv, max_abs_mv=basis.max_abs_mv,
            note=("real MaleCNS descending neurons driven by bounded additive "
                  "current; command->body mapping is the engineered arcade "
                  "decoder (lateral via flight strafe, vertical via arcade "
                  "z-drive). Native weights unchanged.")),
        motor_interface=dict(
            lat_gain=float(meta.get("lat_gain", 3.5)),
            vert_gain=float(meta.get("vert_gain", 3.5)),
            cmd_smoothing=float(meta.get("cmd_smoothing", 0.7)),
            note="runtime gains applied to the bridge command before DN drive."),
        offline_test_metrics=meta.get("test", {}),
        privileged_state_at_runtime=False,
        science_mode_touched=False,
        v1_preserved=dict(
            model="arcade_bridge_model_v1.npz",
            dataset="arcade_dataset_v1.npz"),
    )
    (OUT / "arcade_bridge_v2_freeze.json").write_text(json.dumps(freeze, indent=2))
    print("FROZEN arcade-bridge-v2")
    print("  model sha256:", freeze["weights"]["model_sha256"][:16])
    print("  dataset sha256:", freeze["dataset"]["sha256"][:16])
    print("  n_params:", freeze["weights"]["n_params"],
          " features:", freeze["features"]["feature_dim"])
    print("  split (ep):", len(freeze["split"]["train_seeds"] or []), "/",
          len(freeze["split"]["val_seeds"] or []), "/",
          len(freeze["split"]["test_seeds"] or []))
    print("  saved arcade_bridge_v2_freeze.json")
    return freeze


if __name__ == "__main__":
    main()
