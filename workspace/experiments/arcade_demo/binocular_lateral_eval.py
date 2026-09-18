"""PHASE 9-10: closed-loop evaluation of the structured binocular LATERAL hybrid.

Matched, reset-corrected 3x3 shots (brain reset per episode inside run_episode)
compared across:

    Oracle                         (privileged physical ceiling)
    v2 one-eye                     (current strong baseline, 74.1%)
    current V3 (stale viewport)    (frozen binocular V3)
    structured-binocular-lateral   (this hybrid, retina_map="fullframe")

PHASE 10 eye-blinding ablation on the hybrid:
    both / left_blind / right_blind / both_blind

Per-step lateral diagnostics are logged for the hybrid (both-eyes) run:
    fused u_lat, left-stream prediction, right-stream prediction,
    lateral DN drive, fly lateral velocity, fly lateral displacement.

The vertical path is held FIXED (frozen V3 vertical head) across every hybrid
condition so the lateral comparison is fair.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.vision_bridge import VisionBridge
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (ArcadeController,
                                                    Arcade2AxisDecoder, run_episode,
                                                    DECISION_S, MAX_DECISIONS)
from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                   ArcadeLearnedBridge,
                                                   MetalSafeFeatureExtractor)
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.arcade_oracle import (ArcadeOracleController,
                                                   run_oracle_episode)
from experiments.arcade_demo.binocular_v3 import load as load_v3
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))


def _cells(per_cell, base_seed):
    cells, seed = [], base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            for _ in range(per_cell):
                cells.append((seed, g, hname, hf)); seed += 1
    return cells


def _summ(res, events=None, n_override=None):
    s = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())

    def axis(idx, keys):
        return {k: round(sum(v[0] for (g, h), v in res.items()
                             if (g if idx == 0 else h) == k)
                         / max(1, sum(v[1] for (g, h), v in res.items()
                                      if (g if idx == 0 else h) == k)), 3)
                for k in keys}
    ans = dict(overall=round(s / n, 3), saves=s, n=n,
               by_group=axis(0, GROUPS),
               by_height=axis(1, ("low", "mid", "high")),
               matrix_3x3={f"{g}/{h}": f"{v[0]}/{v[1]}"
                           for (g, h), v in sorted(res.items())})
    if events:
        ans.update({k: round(v / n, 3) for k, v in events.items()})
    return ans


def _record(out, res, events):
    res[(out["group"], out["_hname"])][0] += int(out["result"] == "SAVE")
    res[(out["group"], out["_hname"])][1] += 1
    events["keeper_contact_rate"] += int(out.get("keeper_contact", False))
    events["touch_but_goal_rate"] += int(out.get("touch_but_goal", False))
    events["deflected_save_rate"] += int(out.get("deflected", False)
                                         and out["result"] == "SAVE")


def eval_oracle(cells):
    ctrl = ArcadeOracleController(lat_gain=3.0)
    res = defaultdict(lambda: [0, 0]); events = defaultdict(int)
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        out["_hname"] = hname
        _record(out, res, events)
    return _summ(res, events)


def _make_v2_bridge(brain):
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = [int(x) for x in man["body_ids"][:207]]
    model = LinearBridge2D.load(OUT / "arcade_bridge_model_v2.npz")
    fm = getattr(model, "loaded_metadata", {})
    ex = MetalSafeFeatureExtractor(body, n_windows=int(fm.get("n_windows", 4)))
    return ArcadeLearnedBridge(ex, model, ArcadeDNBasis(),
                               lat_gain=fm.get("lat_gain", 3.5),
                               vert_gain=fm.get("vert_gain", 4.5), enabled=True,
                               cmd_smoothing=fm.get("cmd_smoothing", 0.2))


def eval_controller(cells, brain, make_bridge, make_vision, trace_signals=False):
    """Generic neural closed-loop evaluation with optional per-step lateral log."""
    res = defaultdict(lambda: [0, 0]); events = defaultdict(int)
    traces = []
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        bridge = make_bridge()
        ctrl = ArcadeController(brain, make_vision(w), Arcade2AxisDecoder(),
                                bridge=bridge)
        if trace_signals:
            out, tr = _run_with_lateral_trace(w, ctrl, bridge,
                                              w.sample_shot(g, height_frac=hf),
                                              seed, g, hname)
            traces.append(tr)
        else:
            out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        out["_hname"] = hname; out["group"] = g
        _record(out, res, events)
    summ = _summ(res, events)
    if trace_signals:
        summ["_traces"] = traces
    return summ


def _run_with_lateral_trace(world, controller, bridge, shot, seed, g, hname):
    """Like run_episode but logs the hybrid's per-step lateral signals."""
    world.reset(shot)
    if hasattr(controller, "brain"):
        controller.brain.reset()
    controller.reset()
    log = []
    n = 0
    prev_y = float(world.fly.position[1])
    y0 = prev_y
    while world.result is None and n < MAX_DECISIONS:
        command, diag = controller.act(world)
        world.fly.set_command(command["forward"], command.get("turn", 0.0),
                              command["gait_on"],
                              lateral=command.get("lateral", 0.0),
                              vertical=command.get("vertical", 0.0))
        world.step(DECISION_S)
        fy = float(world.fly.position[1])
        vlat = float(world.fly.velocity[1]) if hasattr(world.fly, "velocity") else (fy - prev_y) / DECISION_S
        # lateral DN drive: opponent (right - left) of the injected lateral current
        pL = float(getattr(bridge, "last_pL", 0.0))
        pR = float(getattr(bridge, "last_pR", 0.0))
        u_lat = float(diag.get("bridge_u_lat", 0.0))
        log.append(dict(step=n, u_lat=round(u_lat, 4),
                        left_pred=round(pL, 4), right_pred=round(pR, 4),
                        lateral_dn_drive=round(u_lat * bridge.lat_gain, 4),
                        fly_lat_vel=round(vlat, 4),
                        fly_lat_disp=round(fy - y0, 4)))
        prev_y = fy
        n += 1
    diag_out = world.outcome_diagnostics() if hasattr(world, "outcome_diagnostics") else {}
    out = dict(group=g, result=world.result or "GOAL",
               keeper_contact=bool(getattr(world, "keeper_contact", False)),
               deflected=bool(getattr(world, "deflected", False)),
               touch_but_goal=diag_out.get("touch_but_goal"))
    tr = dict(seed=int(seed), group=g, height=hname, result=out["result"],
              steps=log)
    return out, tr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--hybrid", default="arcade_binocular_lateral_hybrid.npz")
    p.add_argument("--skip-baselines", action="store_true")
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="metal")
    t0 = time.perf_counter()
    result = {"n_shots": len(cells), "per_cell": a.per_cell,
              "base_seed": a.base_seed,
              "note": ("matched reset-corrected shots; hybrid uses "
                       "retina_map=fullframe (geometry-corrected); vertical path "
                       "frozen (V3 vertical head) across all hybrid conditions")}

    # --- baselines ---
    result["oracle"] = eval_oracle(cells)
    print(f"  oracle       {result['oracle']['overall']:.1%}")
    if not a.skip_baselines:
        result["v2_one_eye"] = eval_controller(
            cells, brain, lambda: _make_v2_bridge(brain),
            lambda w: VisionBridge(w.fly, brain, camera="eye_left", condition="normal"))
        print(f"  v2_one_eye   {result['v2_one_eye']['overall']:.1%} {result['v2_one_eye']['by_group']}")
        result["v3_stale_viewport"] = eval_controller(
            cells, brain,
            lambda: load_v3(OUT / "arcade_binocular_bridge_v3.npz", ArcadeDNBasis())[0],
            lambda w: BinocularVisionBridge(w.fly, brain, condition="both",
                                            retina_map="viewport"))
        print(f"  v3_stale     {result['v3_stale_viewport']['overall']:.1%} {result['v3_stale_viewport']['by_group']}")

    # --- structured binocular lateral hybrid (Phase 9) + blinding matrix (Phase 10) ---
    def make_hybrid():
        return load_hybrid(a.hybrid, ArcadeDNBasis())[0]

    conditions = ["both", "left_blind", "right_blind", "both_blind"]
    result["hybrid"] = {}
    for cond in conditions:
        trace = (cond == "both")
        summ = eval_controller(
            cells, brain, make_hybrid,
            lambda w, c=cond: BinocularVisionBridge(w.fly, brain, condition=c,
                                                    retina_map="fullframe"),
            trace_signals=trace)
        traces = summ.pop("_traces", None)
        result["hybrid"][cond] = summ
        print(f"  hybrid[{cond:11s}] {summ['overall']:.1%} {summ['by_group']}")
        if traces is not None:
            # keep a stratified sample: a few traces from each shot group
            by_g = {}
            for tr in traces:
                by_g.setdefault(tr["group"], []).append(tr)
            sample = []
            for g in GROUPS:
                sample += by_g.get(g, [])[:6]
            (OUT / "binocular_lateral_hybrid_traces.json").write_text(
                json.dumps(sample, indent=2))
    brain.close()
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "binocular_lateral_eval.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved binocular_lateral_eval.json ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
