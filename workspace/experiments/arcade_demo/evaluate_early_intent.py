"""Held-out evaluation of the early-intent controller.

Reports on held-out stratified CLEAN_GOAL shots (seeds disjoint from training):
  * overall save %, 3x3 matrix, by-group (esp RIGHT), by-height
  * per-band takeoff rate, mean u_vert, mean peak_z  (verify LOW<MID<HIGH)
  * movement (peak lateral/vertical, diagonal-jump, onset)
  * natural-miss (absent-keeper) sanity
  * eye ablations (both / left_blind / right_blind / both_blind)
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path: sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path: sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import Arcade2AxisDecoder, DECISION_MS, DECISION_S, MAX_DECISIONS
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.early_intent import EarlyIntentEstimator
from experiments.arcade_demo.early_intent_bridge import EarlyIntentBridge, ExecPolicy
from experiments.arcade_demo import shot_validity as SV
OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
AIR = ("TAKEOFF", "FLIGHT", "DIVE"); HOP = ("LOW_HOP", "TAKEOFF", "FLIGHT", "DIVE")


def hband(h): return "low" if h < 0.52 else "mid" if h < 0.72 else "high"


def rollout(brain, est, model, sense_steps, seed, shot, condition="both"):
    world = ArcadeGoalkeeperWorld(seed=seed)
    hybrid, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
    bridge = EarlyIntentBridge(hybrid, est, policy=model, sense_steps=sense_steps)
    vision = BinocularVisionBridge(world.fly, brain, condition=condition, retina_map="fullframe")
    dec = Arcade2AxisDecoder()
    world.reset(shot); brain.reset(); bridge.reset()
    tr = []; n = 0
    while world.result is None and n < MAX_DECISIONS:
        vision.perceive(); bridge.inject(brain); brain.step(DECISION_MS); bridge.observe(brain)
        dec.decode(brain.read(dec.readout_ids()))
        g = bridge.gait_on(); u_lat, u_vert = bridge.command(world)
        world.fly.set_command(0, 0, g, lateral=u_lat, vertical=u_vert); world.step(DECISION_S)
        tr.append(dict(fly_y=round(float(world.fly.position[1]), 4), fly_z=round(float(world.fly.position[2]), 4),
                   u_lat=round(u_lat, 4), u_vert=round(u_vert, 4),
                   st=str(getattr(world.fly, "state", "?")))); n += 1
    d = world.outcome_diagnostics()
    return world.result or "GOAL", tr, d


def mv(tr):
    y = np.array([t["fly_y"] for t in tr]); z = np.array([t["fly_z"] for t in tr]); st = [t["st"] for t in tr]
    lat = np.abs(y - y[0]); ver = np.abs(z - z[0])
    onset = next((i for i in range(len(tr)) if lat[i] > 0.05 or ver[i] > 0.05), -1)
    airborne_steps = sum(1 for s in st if s in AIR)
    return dict(peak_lat=float(lat.max()), peak_vert=float(ver.max()),
                takeoff=float(any(s in AIR for s in st)), hop=float(any(s in HOP for s in st)),
                diag=float(any((s in AIR) and lat[i] > 0.3 for i, s in enumerate(st))),
                onset=onset, mean_uvert=float(np.mean([t["u_vert"] for t in tr])),
                airborne_duration=round(airborne_steps * DECISION_S, 3))


def evaluate(brain, est, model, sense_steps, shots, condition="both"):
    res = defaultdict(lambda: [0, 0]); ms = []; by_h = defaultdict(lambda: [0, 0, [], [], []])
    contacts = 0; deflected = 0; touch_goal = 0
    for seed, shot in shots:
        r, tr, d = rollout(brain, est, model, sense_steps, seed, shot, condition)
        g = shot.group; hb = hband(shot.height)
        res[(g, hb)][0] += int(r == "SAVE"); res[(g, hb)][1] += 1
        m = mv(tr); ms.append(m)
        by_h[hb][0] += int(r == "SAVE"); by_h[hb][1] += 1
        by_h[hb][2].append(m["takeoff"]); by_h[hb][3].append(m["mean_uvert"]); by_h[hb][4].append(m["peak_vert"])
        contacts += int(bool(d.get("keeper_contact")))
        deflected += int(bool(d.get("deflected_save")))
        touch_goal += int(bool(d.get("touch_but_goal")))
    n = sum(v[1] for v in res.values()); s = sum(v[0] for v in res.values())

    def axis(idx, keys):
        o = {}
        for k in keys:
            ss = sum(v[0] for (g, h), v in res.items() if (g if idx == 0 else h) == k)
            nn = sum(v[1] for (g, h), v in res.items() if (g if idx == 0 else h) == k)
            o[k] = round(ss / nn, 3) if nn else None
        return o

    def axis_counts(idx, keys):
        o = {}
        for k in keys:
            ss = sum(v[0] for (g, h), v in res.items() if (g if idx == 0 else h) == k)
            nn = sum(v[1] for (g, h), v in res.items() if (g if idx == 0 else h) == k)
            o[k] = f"{ss}/{nn}"
        return o

    return dict(overall=round(s / n, 3), saves=s, n=n,
                by_group=axis(0, ("left", "center", "right")),
                counts_by_group=axis_counts(0, ("left", "center", "right")),
                by_height=axis(1, ("low", "mid", "high")),
                counts_by_height=axis_counts(1, ("low", "mid", "high")),
                by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}" for (g, h), v in sorted(res.items())},
                diagnostics=dict(keeper_contacts=f"{contacts}/{n}",
                                 deflected_saves=f"{deflected}/{n}",
                                 touch_but_goal=f"{touch_goal}/{n}"),
                takeoff_by_height={h: round(float(np.mean(v[2])), 2) for h, v in sorted(by_h.items())},
                mean_uvert_by_height={h: round(float(np.mean(v[3])), 3) for h, v in sorted(by_h.items())},
                peakz_by_height={h: round(float(np.mean(v[4])), 3) for h, v in sorted(by_h.items())},
                movement={k: round(float(np.mean([m[k] for m in ms])), 3)
                          for k in ("peak_lat", "peak_vert", "takeoff", "hop", "diag", "onset", "airborne_duration")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="arcade_early_intent_vert_refined_rl.npz")
    p.add_argument("--sense-steps", type=int, default=8)
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--base-seed", type=int, default=500000)
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--out", default="arcade_early_intent_eval.json")
    a = p.parse_args()
    t0 = time.perf_counter()
    est, est_meta = EarlyIntentEstimator.load()
    model, pol_meta = ExecPolicy.load(a.policy)
    shots = SV.stratified_valid_shots(a.per_cell, a.base_seed)
    # natural-miss sanity
    absent = sum(int(SV.simulate_absent_keeper(ArcadeGoalkeeperWorld(seed=s), sh)[0] == "SAVE")
                 for s, sh in shots)
    print(f"[eval] {len(shots)} HELD-OUT shots; natural-miss saves {absent}/{len(shots)}", flush=True)
    brain = MaleCNSBrain(backend="metal")
    result = dict(policy=a.policy, sense_steps=a.sense_steps, n_shots=len(shots),
                  natural_miss=f"{absent}/{len(shots)}", estimator=est_meta.get("heldout_episode_acc"))
    try:
        r = evaluate(brain, est, model, a.sense_steps, shots)
        result["both"] = r
        print(f"  BOTH: {r['overall']:.1%} ({r['saves']}/{r['n']})", flush=True)
        print(f"        by_group={r['by_group']} counts_by_g={r['counts_by_group']}", flush=True)
        print(f"        by_height={r['by_height']} counts_by_h={r['counts_by_height']}", flush=True)
        print(f"        diagnostics={r['diagnostics']}", flush=True)
        print(f"        takeoff_by_h={r['takeoff_by_height']} mean_uvert_by_h={r['mean_uvert_by_height']} "
              f"peakz_by_h={r['peakz_by_height']}", flush=True)
        print(f"        movement={r['movement']}", flush=True)
        print(f"        3x3={r['by_cell']}", flush=True)
        if a.ablations:
            for cond in ("left_blind", "right_blind", "both_blind"):
                rc = evaluate(brain, est, model, a.sense_steps, shots, condition=cond)
                result[cond] = dict(overall=rc["overall"], saves=rc["saves"], n=rc["n"],
                                    by_group=rc["by_group"], counts_by_group=rc["counts_by_group"])
                print(f"  {cond}: {rc['overall']:.1%} ({rc['saves']}/{rc['n']}) by_group={rc['by_group']}", flush=True)
    finally:
        brain.close()
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / a.out).write_text(json.dumps(result, indent=2))
    print(f"saved {a.out} ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
