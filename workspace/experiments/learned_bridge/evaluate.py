"""Closed-loop evaluation of the frozen learned bridge + controls + comparisons.

Runs matched held-out shot sets (same seeds across conditions/controllers) and
reports, per condition/controller: overall + per-direction save rates with
binomial CIs, plus behavioural/neural metrics (mean lateral displacement,
movement-direction accuracy, teacher-command correlation, bridge-command
distribution, selected-DN firing, left/right DN opponent asymmetry).

Controllers:  passive | random | natural (MaleCNS, bridge off) | bridge (learned)
              | heuristic (oracle ceiling)
Bridge controls (bridge controller only): normal | blind | mirrored | shuffled
              | bridge_off

Feature selection / model are FROZEN (loaded from bridge_model.npz). No tuning
here. Held-out seeds are DISJOINT from the dataset seeds used for training.
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.embodied_flykeeper.controllers import (PassiveController,
                                                        RandomController,
                                                        HeuristicController)
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN
from experiments.learned_bridge.bridge import (VisualFeatureExtractor,
                                               LinearBridgeModel, LearnedBridge)
from experiments.learned_bridge.runtime import (BridgeController, run_episode,
                                                MAX_DECISIONS, DECISION_S)
from experiments.learned_bridge.dataset import teacher_command

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
GROUPS = ("left", "center", "right")


# ------------------------------------------------------------- statistics
def wilson_ci(k, n, z=1.96):
    """Wilson score binomial CI for k successes of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def summarize(rows):
    by = {g: [0, 0] for g in GROUPS}
    for r in rows:
        by[r["group"]][0] += int(r["result"] == "SAVE")
        by[r["group"]][1] += 1
    tot_s = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    out = dict(save_rate=round(tot_s / tot_n, 4) if tot_n else None,
               saves=tot_s, n=tot_n, ci=wilson_ci(tot_s, tot_n))
    out["by_group"] = {g: dict(save_rate=round(v[0] / v[1], 4) if v[1] else None,
                               saves=v[0], n=v[1], ci=wilson_ci(v[0], v[1]))
                       for g, v in by.items()}
    return out


# ------------------------------------------------------------- shot set
def make_shot_set(seeds_groups):
    """Deterministic held-out shots: list of (seed, group) -> ShotSpec via a
    per-seed world (same construction the dataset used)."""
    shots = []
    for seed, group in seeds_groups:
        w = GoalkeeperWorld(seed=seed)
        shots.append((seed, group, w.sample_shot(group)))
    return shots


def heldout_seeds(n_per_group, base_seed=90000):
    sg = []
    s = base_seed
    for g in GROUPS:
        for _ in range(n_per_group):
            sg.append((s, g)); s += 1
    return sg


# ------------------------------------------------------------- simple controllers
def eval_simple(kind, shots, seed=7):
    ctrl = {"passive": PassiveController(), "random": RandomController(seed),
            "heuristic": HeuristicController(GOAL_LINE_X)}[kind]
    world = GoalkeeperWorld(seed=1)
    rows = []
    for _, group, shot in shots:
        world.reset(shot); ctrl.reset()
        disp = []
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            cmd, _ = ctrl.act(world)
            world.fly.set_command(cmd["forward"], cmd.get("turn", 0), cmd["gait_on"],
                                  lateral=cmd.get("lateral", 0))
            world.step(DECISION_S); disp.append(float(world.fly.position[1])); n += 1
        rows.append(dict(group=group, result=world.result or "GOAL",
                         final_fly_y=float(world.fly.position[1]),
                         mean_abs_disp=float(np.mean(np.abs(disp))) if disp else 0.0))
    return rows


# ------------------------------------------------------------- neural controllers
def build_bridge(model_path, gain=1.0, n_neurons=None, feature_idx=None,
                 cmd_smoothing=0.0):
    model = LinearBridgeModel.load(model_path)
    meta = getattr(model, "loaded_metadata", {})
    man = np.load(OUT / "visual_manifest.npz")
    k = n_neurons or meta.get("n_neurons", len(man["graph_index"]))
    gi = man["graph_index"][:k].astype(np.int64)
    ex = VisualFeatureExtractor(gi, n_windows=meta.get("n_windows", 4))
    basis = DNMotorBasis()
    return LearnedBridge(ex, model, basis, gain=gain, enabled=True,
                         cmd_smoothing=cmd_smoothing), meta


def eval_neural(shots, *, bridge=None, condition="normal", enabled=True,
                collect_metrics=True):
    """Natural (bridge=None) or learned-bridge closed loop under a condition."""
    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=1)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left", condition=condition)
    decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)
    if bridge is not None:
        bridge.enabled = enabled
    ctrl = BridgeController(brain, vision, decoder, bridge=bridge)
    rows = []
    for _, group, shot in shots:
        world.reset(shot); ctrl.reset()
        if condition == "static_ball":
            vision.set_static_reference(vision.render())
        disp, us, dn_l, dn_r, teacher_us, moved_dir = [], [], [], [], [], []
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            cmd, diag = ctrl.act(world)
            world.fly.set_command(cmd["forward"], 0.0, cmd["gait_on"],
                                  lateral=cmd.get("lateral", 0.0))
            # teacher command for correlation (privileged, metric only)
            tu, _ = teacher_command(world)
            world.step(DECISION_S)
            disp.append(float(world.fly.position[1]))
            us.append(float(diag.get("bridge_u", 0.0)))
            teacher_us.append(tu)
            dn_l.append(float(diag.get("left_spikes", 0.0)))
            dn_r.append(float(diag.get("right_spikes", 0.0)))
            n += 1
        rows.append(dict(group=group, result=world.result or "GOAL",
                         final_fly_y=float(world.fly.position[1]),
                         mean_abs_disp=float(np.mean(np.abs(disp))) if disp else 0.0,
                         bridge_u_mean=float(np.mean(us)) if us else 0.0,
                         bridge_u_absmean=float(np.mean(np.abs(us))) if us else 0.0,
                         teacher_corr_num=(us, teacher_us),
                         dn_left=float(np.sum(dn_l)), dn_right=float(np.sum(dn_r))))
    brain.close()
    return rows


def behavioural_metrics(rows):
    """Aggregate behavioural/neural metrics across rows."""
    # teacher-command correlation across all steps
    all_u, all_t = [], []
    for r in rows:
        if "teacher_corr_num" in r:
            u, t = r["teacher_corr_num"]; all_u += u; all_t += t
    all_u = np.array(all_u); all_t = np.array(all_t)
    corr = (float(np.corrcoef(all_u, all_t)[0, 1])
            if len(all_u) > 2 and all_u.std() > 1e-9 and all_t.std() > 1e-9 else 0.0)
    # movement-direction accuracy: sign(final displacement) vs sign(teacher intent
    # by group): left group wants +y (u<0), right group wants -y (u>0).
    want = {"left": +1, "right": -1, "center": 0}
    md_hits, md_n = 0, 0
    for r in rows:
        if r["group"] in ("left", "right"):
            md_n += 1
            md_hits += int(np.sign(r["final_fly_y"]) == want[r["group"]])
    dn_l = sum(r.get("dn_left", 0.0) for r in rows)
    dn_r = sum(r.get("dn_right", 0.0) for r in rows)
    asym = (dn_r - dn_l) / (dn_r + dn_l) if (dn_r + dn_l) > 0 else 0.0
    return dict(
        mean_abs_disp=round(float(np.mean([r["mean_abs_disp"] for r in rows])), 4),
        teacher_corr=round(corr, 4),
        move_dir_acc=round(md_hits / md_n, 4) if md_n else None,
        bridge_u_absmean=round(float(np.mean([r.get("bridge_u_absmean", 0.0) for r in rows])), 4),
        dn_left_total=round(dn_l, 1), dn_right_total=round(dn_r, 1),
        dn_opponent_asym=round(float(asym), 4),
    )


def clean_rows(rows):
    """Strip the bulky per-step corr payload for JSON."""
    out = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "teacher_corr_num"}
        out.append(rr)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--gain", type=float, default=None,
                   help="override frozen gain (default: use frozen metadata)")
    p.add_argument("--model", type=str, default="bridge_model.npz")
    p.add_argument("--controls", action="store_true",
                   help="run blind/mirrored/shuffled/bridge_off controls")
    p.add_argument("--out", type=str, default="evaluate.json")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sg = heldout_seeds(a.per_group, base_seed=a.base_seed)
    shots = make_shot_set(sg)
    print(f"[eval] {len(shots)} held-out shots ({a.per_group}/group), "
          f"seeds {a.base_seed}..")

    result = dict(config=vars(a), n_shots=len(shots),
                  seeds=[s for s, _ in sg])

    t0 = time.perf_counter()
    # --- baselines
    for kind in ("passive", "random", "heuristic"):
        rows = eval_simple(kind, shots)
        result[kind] = summarize(rows)
        result[kind]["behaviour"] = dict(
            mean_abs_disp=round(float(np.mean([r["mean_abs_disp"] for r in rows])), 4))
        print(f"  {kind:>10}: {result[kind]['save_rate']}")

    # --- natural MaleCNS (bridge off, no bridge object)
    rows = eval_neural(shots, bridge=None, condition="normal")
    result["natural"] = summarize(rows)
    result["natural"]["behaviour"] = behavioural_metrics(rows)
    print(f"  {'natural':>10}: {result['natural']['save_rate']}")

    # --- learned bridge (normal). Gain + command smoothing are FROZEN in the
    # model metadata (selected on validation seeds, never on the test set).
    _m = LinearBridgeModel.load(OUT / a.model)
    fm = getattr(_m, "loaded_metadata", {})
    gain = a.gain if a.gain is not None else fm.get("runtime_gain", 1.0)
    smoothing = fm.get("cmd_smoothing", 0.0)
    print(f"  [frozen] gain={gain} cmd_smoothing={smoothing}")
    bridge, meta = build_bridge(OUT / a.model, gain=gain, cmd_smoothing=smoothing)
    result["bridge_meta"] = meta
    result["frozen_runtime"] = dict(gain=gain, cmd_smoothing=smoothing)
    rows = eval_neural(shots, bridge=bridge, condition="normal", enabled=True)
    result["bridge_normal"] = summarize(rows)
    result["bridge_normal"]["behaviour"] = behavioural_metrics(rows)
    print(f"  {'bridge':>10}: {result['bridge_normal']['save_rate']}  "
          f"(L={result['bridge_normal']['by_group']['left']['save_rate']} "
          f"C={result['bridge_normal']['by_group']['center']['save_rate']} "
          f"R={result['bridge_normal']['by_group']['right']['save_rate']})")

    if a.controls:
        for cond in ("blind", "mirrored", "shuffled"):
            bridge, _ = build_bridge(OUT / a.model, gain=gain, cmd_smoothing=smoothing)
            rows = eval_neural(shots, bridge=bridge, condition=cond, enabled=True)
            result[f"bridge_{cond}"] = summarize(rows)
            result[f"bridge_{cond}"]["behaviour"] = behavioural_metrics(rows)
            print(f"  bridge/{cond}: {result[f'bridge_{cond}']['save_rate']}")
        # bridge OFF (disabled) = must reproduce natural
        bridge, _ = build_bridge(OUT / a.model, gain=gain, cmd_smoothing=smoothing)
        rows = eval_neural(shots, bridge=bridge, condition="normal", enabled=False)
        result["bridge_off"] = summarize(rows)
        result["bridge_off"]["behaviour"] = behavioural_metrics(rows)
        print(f"  bridge/off: {result['bridge_off']['save_rate']}")

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / a.out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved {a.out}  ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
