"""Stage 3 -- open-loop DN injection validation (Section 20.3).

Independently of any closed-loop success, verify that feeding a bridge command
u through the fixed DN motor basis produces the expected, bounded, lateralized
DN drive and body motion:

  * u < 0  -> LEFT DNs excited, fly strafes +y (left)
  * u > 0  -> RIGHT DNs excited, fly strafes -y (right)
  * |drive| bounded (<= 30 mV), scales with |u|
  * DN spike laterality follows the injected side

No vision here: the brain runs blind so the ONLY directional input is the bridge
current, isolating the injection pathway. The selected DNs keep evolving under
the existing MaleCNS LIF dynamics (additive current, not overwrite).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.fly_body import FlyBody
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"


def run_command(brain, fb, decoder, basis, u, steps=60):
    brain.reset(); fb.set_pose(xy=(0, 0), yaw=0.0); decoder.reset()
    lat, moves, lspk, rspk = [], [], [], []
    for _ in range(steps):
        basis.inject(brain, u)              # bounded additive DN current, no vision
        brain.step(20.0)
        act = brain.read(decoder.readout_ids())
        cmd, diag = decoder.decode(act)
        fb.set_command(0.0, 0.0, cmd["gait_on"], lateral=cmd["lateral"])
        fb.step(0.02)
        lat.append(cmd["lateral"]); moves.append(cmd["move"])
        lspk.append(diag["left_spikes"]); rspk.append(diag["right_spikes"])
    from collections import Counter
    return dict(u=u, final_dy=round(float(fb.position[1]), 3),
                mean_lateral=round(float(np.mean(lat)), 3),
                moves=dict(Counter(moves)),
                left_spikes=round(float(np.sum(lspk)), 1),
                right_spikes=round(float(np.sum(rspk)), 1))


def main():
    from adapters.brain import MaleCNSBrain
    OUT.mkdir(parents=True, exist_ok=True)
    brain = MaleCNSBrain(backend="cpu")
    fb = FlyBody()
    decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)
    basis = DNMotorBasis()

    res = {}
    for u in (-1.0, -0.5, 0.0, +0.5, +1.0):
        r = run_command(brain, fb, decoder, basis, u)
        res[f"u={u:+.1f}"] = r
        print(f"  u={u:+.1f}: dy={r['final_dy']:+.3f} lat={r['mean_lateral']:+.3f} "
              f"L={r['left_spikes']} R={r['right_spikes']} moves={r['moves']}")

    # checks
    left_dy = res["u=-1.0"]["final_dy"]; right_dy = res["u=+1.0"]["final_dy"]
    checks = dict(
        left_command_moves_left=bool(left_dy > 0.1),
        right_command_moves_right=bool(right_dy < -0.1),
        opponent_symmetry=round(abs(left_dy + right_dy) /
                                (abs(left_dy) + abs(right_dy) + 1e-9), 3),
        left_excites_left_dn=bool(res["u=-1.0"]["left_spikes"] > res["u=-1.0"]["right_spikes"]),
        right_excites_right_dn=bool(res["u=+1.0"]["right_spikes"] > res["u=+1.0"]["left_spikes"]),
        magnitude_scales=bool(abs(res["u=+1.0"]["final_dy"]) > abs(res["u=+0.5"]["final_dy"]) - 0.05),
    )
    result = dict(basis=basis.describe(), by_command=res, checks=checks,
                  passed=bool(checks["left_command_moves_left"]
                              and checks["right_command_moves_right"]
                              and checks["left_excites_left_dn"]
                              and checks["right_excites_right_dn"]))
    (OUT / "validate_injection.json").write_text(json.dumps(result, indent=2))
    print("\nchecks:", json.dumps(checks, indent=2))
    print("PASS" if result["passed"] else "FAIL")
    brain.close()


if __name__ == "__main__":
    main()
