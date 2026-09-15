"""Ablations, temporal-history tests, and the minimality experiment.

Two families:

OFFLINE (fast, on the dataset test split -- no brain sim): re-fit the linear
model under feature/window ablations and measure the change in held-out teacher
prediction (corr / signed-dir-acc). These test whether the bridge USES genuine
temporal/motion information and how prediction degrades as we shrink the visual
feature set.

  * temporal-history: single-current-window only | full 4-window history |
    reversed window order | remove oldest window | remove newest window.
  * feature-count minimality: 16 / 32 / 64 / 128 / 207 top neurons.

CLOSED-LOOP (brain sim, small matched shot set): causal behavioural ablations
of the frozen bridge -- zero the whole bridge, zero the left-preferring visual
population, zero the right-preferring population, zero the left-DN drive, zero
the right-DN drive -- and verify the behavioural change matches the expected
laterality. Also the motor-side minimality (fewer DNs / single opponent axis).

All offline model selection uses train/val; closed-loop uses held-out seeds.
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

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
DECISION_MS = 20.0


# ---------------------------------------------------------------- offline
def _fit_eval(ds, splits, *, n_windows=4, order="newest_first", feature_idx=None,
              window_mask=None, target_offset=4, alpha=300.0):
    Xtr, ytr, _ = F.build_matrix(ds, splits["train"], n_windows=n_windows,
                                 order=order, feature_idx=feature_idx,
                                 window_mask=window_mask, target_offset=target_offset)
    Xte, yte, _ = F.build_matrix(ds, splits["test"], n_windows=n_windows,
                                 order=order, feature_idx=feature_idx,
                                 window_mask=window_mask, target_offset=target_offset)
    mu, sd = F.fit_normalizer(Xtr)
    w, b = F.ridge_fit(F.apply_normalizer(Xtr, mu, sd), ytr, alpha=alpha)
    pred = F.ridge_predict(F.apply_normalizer(Xte, mu, sd), w, b)
    return F.regression_metrics(yte, pred)


def _fit_frozen_then_eval_reordered(ds, splits, *, n_windows=4, target_offset=4,
                                    alpha=300.0):
    """Train on newest-first, then FEED reversed-window inputs at test WITHOUT
    refitting. This is the meaningful temporal-order test: if the frozen weights
    rely on the temporal ordering, scrambling it at inference should hurt."""
    Xtr, ytr, _ = F.build_matrix(ds, splits["train"], n_windows=n_windows,
                                 order="newest_first", target_offset=target_offset)
    mu, sd = F.fit_normalizer(Xtr)
    w, b = F.ridge_fit(F.apply_normalizer(Xtr, mu, sd), ytr, alpha=alpha)
    Xte, yte, _ = F.build_matrix(ds, splits["test"], n_windows=n_windows,
                                 order="reversed", target_offset=target_offset)
    pred = F.ridge_predict(F.apply_normalizer(Xte, mu, sd), w, b)
    return F.regression_metrics(yte, pred)


def temporal_history(ds, splits):
    res = {}
    res["full_4_windows"] = _fit_eval(ds, splits, n_windows=4)
    res["single_current_window"] = _fit_eval(ds, splits, n_windows=1)
    # refit under reversed order (invariant for a linear model, shown for
    # completeness) vs FROZEN weights fed reversed inputs (the real test).
    res["reversed_order_refit"] = _fit_eval(ds, splits, n_windows=4, order="reversed")
    res["reversed_order_frozen"] = _fit_frozen_then_eval_reordered(ds, splits)
    # remove oldest (window index 3 in newest-first) / newest (index 0)
    res["remove_oldest"] = _fit_eval(ds, splits, n_windows=4,
                                     window_mask=[True, True, True, False])
    res["remove_newest"] = _fit_eval(ds, splits, n_windows=4,
                                     window_mask=[False, True, True, True])
    return res


def feature_count_minimality(ds, splits, counts=(16, 32, 64, 128, 207)):
    res = []
    n_sel = int(ds["n_sel"])
    for k in counts:
        k = min(k, n_sel)
        m = _fit_eval(ds, splits, feature_idx=np.arange(k))
        res.append(dict(n_neurons=k, n_params=k * 4 + 1, **m))
    return res


# ---------------------------------------------------------------- closed-loop
def _eval_closed(shots, bridge, condition="normal", enabled=True,
                 zero_left_pop=False, zero_right_pop=False,
                 zero_left_dn=False, zero_right_dn=False):
    """Closed-loop with optional visual-population / DN-drive ablations."""
    from adapters.brain import MaleCNSBrain
    from embodiment.mujoco_world import GoalkeeperWorld
    from embodiment.vision_bridge import VisionBridge
    from embodiment.motor_decoder import DescendingMotorDecoder
    from experiments.learned_bridge.runtime import (BridgeController, run_episode)

    world = GoalkeeperWorld(seed=1)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left", condition=condition)
    decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)
    bridge.enabled = enabled

    # visual-population ablation: zero the selected-neuron feature columns whose
    # direction preference is on the ablated side, at the extractor level.
    man = np.load(OUT / "visual_manifest.npz")
    dir_pref = man["dir_pref"][:bridge.extractor.n_neurons]  # +1 right, -1 left
    zero_cols = np.zeros(bridge.extractor.n_neurons, dtype=bool)
    if zero_left_pop:
        zero_cols |= (dir_pref < 0)
    if zero_right_pop:
        zero_cols |= (dir_pref > 0)

    orig_observe = bridge.extractor.observe
    if zero_cols.any():
        def patched_observe(b):
            orig_observe(b)
            bridge.extractor._buf[-1][zero_cols] = 0.0
        bridge.extractor.observe = patched_observe

    # DN-drive ablation: monkeypatch the basis to drop one side's current.
    basis = bridge.basis
    orig_currents = basis.currents
    if zero_left_dn or zero_right_dn:
        def patched_currents(u):
            ids, vals = orig_currents(u)
            out = []
            for i, v in zip(ids, vals):
                if zero_left_dn and i in LEFT_DN:
                    v = 0.0
                if zero_right_dn and i in RIGHT_DN:
                    v = 0.0
                out.append(v)
            return ids, out
        basis.currents = patched_currents

    rows = []
    for _, group, shot in shots:
        out, _ = run_episode(world, BridgeController(brain, vision, decoder, bridge=bridge),
                             shot, condition=condition)
        rows.append(out)
    brain.close()
    # restore patches
    bridge.extractor.observe = orig_observe
    basis.currents = orig_currents
    return rows


def _rate(rows, groups=("left", "center", "right")):
    by = {g: [0, 0] for g in groups}
    for r in rows:
        by[r["group"]][0] += int(r["result"] == "SAVE"); by[r["group"]][1] += 1
    tot_s = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    disp = np.mean([r["final_fly_y"] for r in rows])
    return dict(save_rate=round(tot_s / tot_n, 3),
                by_group={g: round(v[0] / v[1], 3) if v[1] else None for g, v in by.items()},
                mean_final_y=round(float(disp), 3))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=95000)  # held-out, != test/val
    p.add_argument("--model", type=str, default="bridge_model.npz")
    p.add_argument("--closed-loop", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ds = np.load(OUT / "dataset_main.npz", allow_pickle=True)
    splits = episode_splits(ds)
    out = {}

    print("=== temporal-history ablations (offline, test split) ===")
    out["temporal_history"] = temporal_history(ds, splits)
    for k, v in out["temporal_history"].items():
        print(f"  {k:>22}: corr={v['corr']:.3f} dir_acc={v['signed_dir_acc']:.3f}")

    print("\n=== feature-count minimality (offline, test split) ===")
    out["feature_minimality"] = feature_count_minimality(ds, splits)
    for r in out["feature_minimality"]:
        print(f"  {r['n_neurons']:>4} neurons ({r['n_params']} params): "
              f"corr={r['corr']:.3f} dir_acc={r['signed_dir_acc']:.3f}")

    if a.closed_loop:
        from experiments.learned_bridge.evaluate import make_shot_set, build_bridge
        m = LinearBridgeModel.load(OUT / a.model)
        fm = getattr(m, "loaded_metadata", {})
        gain = fm.get("runtime_gain", 3.5); sm = fm.get("cmd_smoothing", 0.8)
        sg = []
        s = a.base_seed
        for g in ("left", "center", "right"):
            for _ in range(a.per_group):
                sg.append((s, g)); s += 1
        shots = make_shot_set(sg)
        print(f"\n=== closed-loop causal ablations ({len(shots)} shots) ===")
        t0 = time.perf_counter()
        cl = {}
        def run(name, **kw):
            b, _ = build_bridge(OUT / a.model, gain=gain, cmd_smoothing=sm)
            enabled = kw.pop("enabled", True)
            rows = _eval_closed(shots, b, enabled=enabled, **kw)
            cl[name] = _rate(rows)
            print(f"  {name:>18}: {cl[name]['save_rate']}  by={cl[name]['by_group']} "
                  f"meanY={cl[name]['mean_final_y']}")
        run("intact")
        run("zero_bridge", enabled=False)
        run("zero_left_visual", zero_left_pop=True)
        run("zero_right_visual", zero_right_pop=True)
        run("zero_left_dn", zero_left_dn=True)
        run("zero_right_dn", zero_right_dn=True)
        out["closed_loop_ablations"] = cl
        out["closed_loop_wall_s"] = round(time.perf_counter() - t0, 1)

    (OUT / "ablations.json").write_text(json.dumps(out, indent=2))
    print("\nsaved ablations.json")


if __name__ == "__main__":
    main()
