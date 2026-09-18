"""Phase 4/5: continuous vertical (and lateral) action-authority sweep.

Drives the ARCADE BODY MOTOR LAYER directly with held commands (no vision, no
brain, no ball) to characterise the PHYSICAL response of the force-driven body:

  for u_vert in {0.0, 0.1, ..., 1.0}  x  u_lat in {0.0, 0.5, 1.0}:
      hold (u_lat, u_vert) for a fixed window, then release and let it recover.

Measured per (u_lat, u_vert):
  takeoff        : did the body leave the ground (FLIGHT/TAKEOFF state) at all
  takeoff_latency: decisions until first airborne state
  peak_z         : max thorax height above stand (cm)
  peak_vz        : max upward velocity (cm/s)
  airborne_steps : decisions spent airborne
  peak_lateral   : max |y| displacement (cm)
  recovered      : returned near stand height after release

Goal: NO severe vertical deadband. We want smooth, monotone continuous
authority: small u_vert -> small hop / stay low, medium -> moderate jump, large
-> strong rise. All via bounded MuJoCo forces (no teleport / no velocity hacks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_body import ArcadeFlyBody
from experiments.arcade_demo.arcade_runtime import DECISION_S

AIRBORNE_STATES = ("TAKEOFF", "FLIGHT", "DIVE")


def probe(u_lat, u_vert, hold_decisions=30, release_decisions=25):
    """Hold a command, then release; return the physical response metrics."""
    body = ArcadeFlyBody()
    body.reset()
    body.set_pose(xy=(0.0, 0.0), yaw=0.0)
    import mujoco
    mujoco.mj_forward(body.model, body.data)
    # settle a few steps so _stand_z is captured
    for _ in range(3):
        body.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
        body.step(DECISION_S)
    stand_z = float(body._stand_z if body._stand_z is not None else body.position[2])

    zs, vzs, ys, states = [], [], [], []
    first_air = None
    for t in range(hold_decisions):
        body.set_command(0, 0, 1.0, lateral=u_lat, vertical=u_vert)
        body.step(DECISION_S)
        z = float(body.position[2]) - stand_z
        vz = float(body.data.qvel[body.root_dofadr + 2])
        y = float(body.position[1])
        zs.append(z); vzs.append(vz); ys.append(y); states.append(str(body.state))
        if first_air is None and body.state in AIRBORNE_STATES:
            first_air = t
    # release and let it recover
    rec_z = []
    for _ in range(release_decisions):
        body.set_command(0, 0, 1.0, lateral=0.0, vertical=0.0)
        body.step(DECISION_S)
        rec_z.append(float(body.position[2]) - stand_z)

    airborne_steps = sum(s in AIRBORNE_STATES for s in states)
    recovered = bool(abs(rec_z[-1]) < 0.06) if rec_z else False
    return dict(
        u_lat=round(u_lat, 2), u_vert=round(u_vert, 2),
        takeoff=bool(first_air is not None),
        takeoff_latency=(int(first_air) if first_air is not None else None),
        peak_z=round(float(np.max(zs)), 4),
        peak_vz=round(float(np.max(vzs)), 4),
        airborne_steps=int(airborne_steps),
        peak_lateral=round(float(np.max(np.abs(ys))), 4),
        recovered=recovered,
        final_recover_z=round(float(rec_z[-1]) if rec_z else 0.0, 4),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=30)
    p.add_argument("--release", type=int, default=25)
    a = p.parse_args()
    vert_grid = [round(0.1 * i, 1) for i in range(0, 11)]
    lat_grid = [0.0, 0.5, 1.0]
    rows = []
    print("=== VERTICAL/LATERAL ACTION-AUTHORITY SWEEP (body motor layer) ===")
    print(f"hold={a.hold} decisions ({a.hold*DECISION_S*1000:.0f} ms), "
          f"release={a.release}\n")
    for u_lat in lat_grid:
        print(f"--- u_lat = {u_lat} ---")
        print(f"  {'u_vert':>6} {'takeoff':>8} {'lat':>4} {'peak_z':>8} "
              f"{'peak_vz':>8} {'air':>4} {'peak_|y|':>9} {'recov':>6}")
        for u_vert in vert_grid:
            r = probe(u_lat, u_vert, a.hold, a.release)
            rows.append(r)
            print(f"  {r['u_vert']:>6} {str(r['takeoff']):>8} "
                  f"{str(r['takeoff_latency']):>4} {r['peak_z']:>8.3f} "
                  f"{r['peak_vz']:>8.3f} {r['airborne_steps']:>4} "
                  f"{r['peak_lateral']:>9.3f} {str(r['recovered']):>6}")
        print()

    # deadband analysis on the u_lat=0 column (pure vertical)
    pure = [r for r in rows if r["u_lat"] == 0.0]
    print("=== VERTICAL DEADBAND ANALYSIS (u_lat=0) ===")
    peak_z = [r["peak_z"] for r in pure]
    # monotonicity of peak_z vs u_vert
    diffs = np.diff(peak_z)
    print(f"peak_z by u_vert: {[round(z,3) for z in peak_z]}")
    print(f"monotone non-decreasing: {bool(np.all(diffs >= -0.02))}")
    # find smallest u_vert with a non-trivial hop (>0.1 cm)
    hop = next((r['u_vert'] for r in pure if r['peak_z'] > 0.1), None)
    strong = next((r['u_vert'] for r in pure if r['peak_z'] > 0.8), None)
    print(f"smallest u_vert with hop (peak_z>0.1): {hop}")
    print(f"smallest u_vert with strong rise (peak_z>0.8): {strong}")

    out = ROOT / "workspace" / "outputs" / "arcade_demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vertical_authority.json").write_text(json.dumps(dict(
        hold=a.hold, release=a.release, rows=rows,
        pure_vertical_peak_z=peak_z,
        smallest_hop_uvert=hop, smallest_strong_uvert=strong), indent=2))
    print("\nsaved vertical_authority.json")


if __name__ == "__main__":
    main()
