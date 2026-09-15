"""Shuffled-retina controls for the learned bridge (Section 26).

The project's built-in `condition="shuffled"` applies a FIXED spatial permutation
to the R1-R6 luminance vector. We discovered this preserves LEFT/RIGHT
*separability*: a fixed permutation is a bijection, so the distinct luminance
vectors for a left vs a right ball remain distinct and CONSISTENT after
permutation. A learned readout can therefore still latch onto the (spatially
scrambled but consistent) direction signal -- so the fixed shuffle removes
retinotopy but NOT discriminability, and does not, on its own, collapse a learned
controller. (The earlier plasticity experiment never noticed this because its
controller was vision-independent regardless.)

We therefore evaluate TWO shuffle controls and report both honestly:

  * shuffle_fixed   : the project's fixed permutation (retinotopy destroyed,
                      L/R separability PRESERVED -- expected NOT to collapse).
  * shuffle_perstep : a fresh random permutation every 20 ms window (Section 26's
                      explicitly-allowed temporal-noise variant). This destroys
                      any CONSISTENT spatial->neuron mapping while preserving the
                      per-frame marginal luminance distribution, so no consistent
                      directional signal can survive. A genuinely
                      structure-dependent bridge must collapse toward baseline
                      here; a bridge exploiting a non-structural artifact would
                      not.

This wraps VisionBridge WITHOUT modifying the shared baseline module: we subclass
and override the shuffle branch of perceive().
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.vision_bridge import VisionBridge
from doom.game import retinal_samples


class ShuffleVisionBridge(VisionBridge):
    """VisionBridge with selectable shuffle mode: 'fixed' or 'perstep'."""

    def __init__(self, *args, shuffle_mode="fixed", shuffle_seed=1234, **kw):
        super().__init__(*args, **kw)
        self.shuffle_mode = shuffle_mode
        self._rng = np.random.default_rng(shuffle_seed)
        self._fixed_perm = np.random.default_rng(shuffle_seed).permutation(len(self.uv))

    def perceive(self, rgb=None):
        if self.condition != "shuffled":
            return super().perceive(rgb)
        if rgb is None:
            rgb = self.render()
        used = self._apply_condition(rgb)          # no-op for 'shuffled'
        luminance = retinal_samples(used, self.uv)
        if self.shuffle_mode == "perstep":
            perm = self._rng.permutation(len(luminance))
        else:
            perm = self._fixed_perm
        luminance = luminance[perm]
        self.brain.stimulate_retinal_luminance(self.retina_ids, luminance)
        return luminance, used


def eval_shuffle(shots, model_path, *, mode, gain, smoothing):
    """Closed-loop eval of the frozen bridge under a shuffle mode."""
    from adapters.brain import MaleCNSBrain
    from embodiment.mujoco_world import GoalkeeperWorld
    from embodiment.motor_decoder import DescendingMotorDecoder
    from experiments.learned_bridge.runtime import BridgeController, run_episode
    from experiments.learned_bridge.evaluate import build_bridge, summarize
    from experiments.learned_bridge.dn_basis import LEFT_DN, RIGHT_DN

    world = GoalkeeperWorld(seed=1)
    brain = MaleCNSBrain(backend="cpu")
    vision = ShuffleVisionBridge(world.fly, brain, camera="eye_left",
                                 condition="shuffled", shuffle_mode=mode)
    decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)
    bridge, _ = build_bridge(model_path, gain=gain, cmd_smoothing=smoothing)
    ctrl = BridgeController(brain, vision, decoder, bridge=bridge)
    rows = []
    for _, group, shot in shots:
        out, _ = run_episode(world, ctrl, shot, condition="shuffled")
        rows.append(dict(group=group, result=out["result"]))
    brain.close()
    return summarize(rows)


def main():
    import argparse, json, time
    from experiments.learned_bridge.evaluate import make_shot_set, heldout_seeds
    from experiments.learned_bridge.bridge import LinearBridgeModel
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--model", type=str, default="bridge_model.npz")
    a = p.parse_args()
    OUT = ROOT / "workspace" / "outputs" / "learned_bridge"

    m = LinearBridgeModel.load(OUT / a.model)
    fm = getattr(m, "loaded_metadata", {})
    gain = fm.get("runtime_gain", 3.5); sm = fm.get("cmd_smoothing", 0.8)
    shots = make_shot_set(heldout_seeds(a.per_group, base_seed=a.base_seed))
    print(f"[shuffle-controls] {len(shots)} test shots, gain={gain} sm={sm}")
    out = {}
    t0 = time.perf_counter()
    for mode in ("fixed", "perstep"):
        s = eval_shuffle(shots, OUT / a.model, mode=mode, gain=gain, smoothing=sm)
        out[f"shuffle_{mode}"] = s
        print(f"  shuffle_{mode}: {s['save_rate']}  "
              f"L={s['by_group']['left']['save_rate']} "
              f"C={s['by_group']['center']['save_rate']} "
              f"R={s['by_group']['right']['save_rate']}")
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    out["note"] = ("shuffle_fixed preserves L/R separability (bijection) so it is "
                   "expected NOT to collapse; shuffle_perstep destroys consistent "
                   "structure and is the genuine structure-dependence control.")
    (OUT / "shuffle_controls.json").write_text(json.dumps(out, indent=2))
    print("saved shuffle_controls.json")


if __name__ == "__main__":
    main()
