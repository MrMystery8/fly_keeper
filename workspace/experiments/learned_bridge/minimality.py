"""Minimality experiment (Section 32): how small an intervention suffices?

VISUAL minimality: retrain the linear bridge on the top-k selected optic-lobe
neurons (k = 16/32/64/128/207) and evaluate CLOSED-LOOP save rate vs feature
count / trainable-parameter count.

MOTOR minimality: shrink the DN motor basis from the 2+2 opponent populations to
a single opponent pair (1 left DN + 1 right DN), and to a single-DN axis, and
evaluate closed-loop.

Each reduced model is trained on the SAME train split with the frozen
hyperparameters (n_windows=4, offset=4, alpha=300), normalization on train only,
then evaluated on held-out seeds. Runtime gain/smoothing are the frozen values.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.learned_bridge import features as F
from experiments.learned_bridge.splits import episode_splits
from experiments.learned_bridge.bridge import (VisualFeatureExtractor,
                                               LinearBridgeModel, LearnedBridge)
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN
from experiments.learned_bridge.evaluate import make_shot_set, heldout_seeds, summarize

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
N_WINDOWS = 4
OFFSET = 4
ALPHA = 300.0


def train_reduced_model(ds, splits, n_neurons):
    """Ridge model on the top-n_neurons manifest neurons (train split)."""
    fidx = np.arange(min(n_neurons, int(ds["n_sel"])))
    Xtr, ytr, _ = F.build_matrix(ds, splits["train"], n_windows=N_WINDOWS,
                                 feature_idx=fidx, target_offset=OFFSET)
    mu, sd = F.fit_normalizer(Xtr)
    w, b = F.ridge_fit(F.apply_normalizer(Xtr, mu, sd), ytr, alpha=ALPHA)
    model = LinearBridgeModel(w=w, b=b, mu=mu, sd=sd)
    return model, len(fidx)


def eval_bridge_closed(shots, model, gain, smoothing, n_neurons,
                       left_dn=LEFT_DN, right_dn=RIGHT_DN):
    from adapters.brain import MaleCNSBrain
    from embodiment.mujoco_world import GoalkeeperWorld
    from embodiment.vision_bridge import VisionBridge
    from embodiment.motor_decoder import DescendingMotorDecoder
    from experiments.learned_bridge.runtime import BridgeController, run_episode

    man = np.load(OUT / "visual_manifest.npz")
    gi = man["graph_index"][:n_neurons].astype(np.int64)
    world = GoalkeeperWorld(seed=1)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left", condition="normal")
    decoder = DescendingMotorDecoder(left_ids=list(left_dn), right_ids=list(right_dn),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)
    ex = VisualFeatureExtractor(gi, n_windows=N_WINDOWS)
    basis = DNMotorBasis(left_ids=left_dn, right_ids=right_dn)
    bridge = LearnedBridge(ex, model, basis, gain=gain, enabled=True,
                           cmd_smoothing=smoothing)
    ctrl = BridgeController(brain, vision, decoder, bridge=bridge)
    rows = []
    for _, group, shot in shots:
        out, _ = run_episode(world, ctrl, shot)
        rows.append(dict(group=group, result=out["result"]))
    brain.close()
    return summarize(rows)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=96000)  # held-out, disjoint
    p.add_argument("--visual", action="store_true")
    p.add_argument("--motor", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ds = np.load(OUT / "dataset_main.npz", allow_pickle=True)
    splits = episode_splits(ds)
    # frozen runtime params
    fm = getattr(LinearBridgeModel.load(OUT / "bridge_model.npz"),
                 "loaded_metadata", {})
    gain = fm.get("runtime_gain", 3.5); sm = fm.get("cmd_smoothing", 0.8)
    shots = make_shot_set(heldout_seeds(a.per_group, base_seed=a.base_seed))
    out = dict(gain=gain, cmd_smoothing=sm, n_shots=len(shots))
    t0 = time.perf_counter()

    if a.visual or not (a.visual or a.motor):
        print("=== VISUAL minimality (closed-loop) ===")
        vis = []
        for k in (16, 32, 64, 128, 207):
            model, nk = train_reduced_model(ds, splits, k)
            s = eval_bridge_closed(shots, model, gain, sm, nk)
            rec = dict(n_neurons=nk, n_params=nk * N_WINDOWS + 1,
                       save_rate=s["save_rate"],
                       by_group={g: s["by_group"][g]["save_rate"] for g in ("left", "center", "right")})
            vis.append(rec)
            print(f"  {nk:>4} neurons ({rec['n_params']} params): {rec['save_rate']}  {rec['by_group']}")
        out["visual_minimality"] = vis

    if a.motor:
        print("\n=== MOTOR minimality (closed-loop) ===")
        model, nk = train_reduced_model(ds, splits, 207)
        motor = []
        configs = [
            ("4dn_2plus2", LEFT_DN, RIGHT_DN),
            ("2dn_1plus1", (LEFT_DN[0],), (RIGHT_DN[0],)),
        ]
        for name, ld, rd in configs:
            s = eval_bridge_closed(shots, model, gain, sm, nk, left_dn=ld, right_dn=rd)
            rec = dict(config=name, n_dn=len(ld) + len(rd),
                       save_rate=s["save_rate"],
                       by_group={g: s["by_group"][g]["save_rate"] for g in ("left", "center", "right")})
            motor.append(rec)
            print(f"  {name} ({rec['n_dn']} DNs): {rec['save_rate']}  {rec['by_group']}")
        out["motor_minimality"] = motor

    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "minimality.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved minimality.json ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
