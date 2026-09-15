"""Motor perturbation + locomotion-bridge symmetry.

Two independent tests:

1) BODY SYMMETRY (no brain): command equal-strength engineered left vs right
   strafe / turn directly and verify the body moves symmetrically. This checks
   the locomotion bridge itself is unbiased (the 98% heuristic already implies
   capability, but this isolates left/right symmetry).

2) DESCENDING MOTOR PERTURBATION (no vision): inject external current into the
   candidate LEFT descending population vs the RIGHT population, read their
   spikes through the SAME decoder used in the closed loop, drive the body, and
   measure the resulting displacement. This establishes whether these DNs can
   actually produce useful left/right body movement through the existing
   hierarchical locomotion layer, independent of whether vision reaches them.

Uses an explicit diagnostic stimulation interface (MaleCNSBrain.stimulate);
the MaleCNS scientific core is not modified.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

from embodiment.fly_body import FlyBody
from embodiment.motor_decoder import DescendingMotorDecoder

LEFT_DN = [10162, 10527]     # DNp20_L, DNpe017_L
RIGHT_DN = [10059, 555871]   # DNp20_R, DNpe017_R


def body_symmetry():
    """Directly command equal L/R strafe and turn; measure displacement."""
    fb = FlyBody()
    out = {}
    for name, (fwd, lat, turn) in [
        ("strafe_left", (0, +1, 0)), ("strafe_right", (0, -1, 0)),
        ("turn_left", (0, 0, -1)), ("turn_right", (0, 0, +1)),
        ("forward", (1, 0, 0)),
    ]:
        fb.set_pose(xy=(0, 0), yaw=0.0)
        fb.set_command(fwd, turn, 1.0, lateral=lat)
        ys, xs, hs = [], [], []
        for _ in range(60):  # 1.2 s
            fb.step(0.02)
            xs.append(float(fb.position[0])); ys.append(float(fb.position[1]))
            hs.append(float(fb.heading))
        out[name] = dict(dx=round(xs[-1], 3), dy=round(ys[-1], 3),
                         dheading=round(hs[-1], 3),
                         # latency to reach 50% of final |dy| (strafe) or |dh| (turn)
                         )
    # symmetry metrics
    sym = dict(
        strafe_dy_left=out["strafe_left"]["dy"],
        strafe_dy_right=out["strafe_right"]["dy"],
        strafe_symmetry=round(abs(out["strafe_left"]["dy"] + out["strafe_right"]["dy"])
                              / (abs(out["strafe_left"]["dy"]) + abs(out["strafe_right"]["dy"]) + 1e-9), 3),
        turn_left=out["turn_left"]["dheading"],
        turn_right=out["turn_right"]["dheading"],
        turn_symmetry=round(abs(out["turn_left"]["dheading"] + out["turn_right"]["dheading"])
                            / (abs(out["turn_left"]["dheading"]) + abs(out["turn_right"]["dheading"]) + 1e-9), 3),
    )
    return dict(per_command=out, symmetry=sym)


def descending_perturbation():
    """Inject current into L vs R DNs, decode, drive body, measure motion."""
    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="cpu")
    fb = FlyBody()
    decoder = DescendingMotorDecoder(left_ids=LEFT_DN, right_ids=RIGHT_DN,
                                     turn_gain=2.5, smoothing=0.5)
    results = {}
    for label, drive_ids in [("drive_left_DNs", LEFT_DN), ("drive_right_DNs", RIGHT_DN),
                             ("drive_none", [])]:
        brain.reset()
        fb.set_pose(xy=(0, 0), yaw=0.0)
        decoder.reset()
        moves = []; lateral_cmds = []
        for _ in range(60):  # 60 x 20 ms decisions = 1.2 s
            if drive_ids:
                # strong external current into the chosen DN population (no vision)
                brain.stimulate(drive_ids, [25.0] * len(drive_ids))
            brain.step(20.0)
            act = brain.read(decoder.readout_ids())
            cmd, diag = decoder.decode(act)
            fb.set_command(cmd["forward"], cmd.get("turn", 0), cmd["gait_on"],
                           lateral=cmd.get("lateral", 0))
            fb.step(0.02)
            moves.append(cmd["move"]); lateral_cmds.append(cmd.get("lateral", 0))
        from collections import Counter
        results[label] = dict(
            final_dy=round(float(fb.position[1]), 3),
            final_dx=round(float(fb.position[0]), 3),
            move_dist=dict(Counter(moves)),
            mean_lateral_cmd=round(float(np.mean(lateral_cmds)), 3),
            left_spikes=round(float(sum(act[i]["spikes"] for i in LEFT_DN)), 1),
            right_spikes=round(float(sum(act[i]["spikes"] for i in RIGHT_DN)), 1),
        )
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== body symmetry (direct commands, no brain) ===")
    sym = body_symmetry()
    print(json.dumps(sym["symmetry"], indent=2))
    print("\n=== descending motor perturbation (inject current, no vision) ===")
    pert = descending_perturbation()
    for k, v in pert.items():
        print(f"  {k}: dy={v['final_dy']:+.3f} dx={v['final_dx']:+.3f} "
              f"Lspk={v['left_spikes']} Rspk={v['right_spikes']} "
              f"lat_cmd={v['mean_lateral_cmd']:+.3f} moves={v['move_dist']}")
    result = dict(body_symmetry=sym, descending_perturbation=pert,
                  left_dn=LEFT_DN, right_dn=RIGHT_DN)
    (OUT / "motor_perturbation.json").write_text(json.dumps(result, indent=2))
    print("\nsaved motor_perturbation.json")


if __name__ == "__main__":
    main()
