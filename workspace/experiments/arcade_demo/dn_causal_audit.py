"""Causal, no-vision audit of the real DN-to-motor Arcade path.

Each condition injects a bounded current into one real DN basis, steps MaleCNS,
decodes only those DN spike counts, and applies that decoded command to the
articulated Arcade fly.  It is deliberately independent of any bridge/model,
so an asymmetry here cannot be blamed on visual feature selection.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path: sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path: sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import Arcade2AxisDecoder, DECISION_MS, DECISION_S
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def run(backend="metal", steps=12, drive=25.0):
    basis = ArcadeDNBasis(drive_mv=drive); brain = MaleCNSBrain(backend=backend)
    conditions = {"left_dn": (-1.0, 0.0), "right_dn": (1.0, 0.0), "vertical_dn": (0.0, 1.0), "zero": (0.0, 0.0)}
    result = {"backend": backend, "steps": steps, "drive_mv": drive,
              "real_dn_ids": basis.describe(), "conditions": {}}
    for name, (u_lat, u_vert) in conditions.items():
        w = ArcadeGoalkeeperWorld(seed=81000); w.reset(w.sample_shot("center", height_frac=.5)); brain.reset()
        dec = Arcade2AxisDecoder(); y0, z0 = float(w.fly.position[1]), float(w.fly.position[2]); rows=[]
        for t in range(steps):
            basis.inject(brain, u_lat, u_vert); brain.step(DECISION_MS)
            activity = brain.read(dec.readout_ids()); cmd, diag = dec.decode(activity)
            w.fly.set_command(cmd["forward"], cmd["turn"], cmd["gait_on"], lateral=cmd["lateral"], vertical=cmd["vertical"])
            w.step(DECISION_S)
            rows.append(dict(step=t, left_dn_spikes=diag["left_spikes"], right_dn_spikes=diag["right_spikes"],
                             vertical_dn_spikes=diag["vert_spikes"], lateral_command=round(diag["lateral"], 5),
                             vertical_command=round(diag["vertical"], 5), fly_y=round(float(w.fly.position[1]), 5), fly_z=round(float(w.fly.position[2]), 5)))
        result["conditions"][name] = dict(delta_y=round(float(w.fly.position[1])-y0, 5), delta_z=round(float(w.fly.position[2])-z0, 5), trace=rows)
    brain.close()
    L=result["conditions"]["left_dn"]["delta_y"]; R=result["conditions"]["right_dn"]["delta_y"]
    result["lateral_symmetry"] = dict(left_delta_y=L, right_delta_y=R,
        same_sign=bool(np.sign(L)==np.sign(R)), magnitude_ratio=round(abs(L)/max(abs(R),1e-9),4))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "dn_causal_audit.json").write_text(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    r=run(); print(json.dumps({"lateral_symmetry":r["lateral_symmetry"], "vertical":r["conditions"]["vertical_dn"]}, indent=2))
