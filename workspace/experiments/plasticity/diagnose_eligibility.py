"""Diagnose whether the eligibility trace is DIRECTION-DIFFERENTIATED.

The sanity test showed scalar-reward learning strengthens common-mode, not
direction. This checks, per pathway stage (source->mid vs mid->DN), whether the
accumulated eligibility on the plastic edges actually differs between left and
right shots -- and whether it is lateralized (source-side / DN-side specific).

If the source->mid eligibility is strongly direction-differentiated but the
mid->DN eligibility is common-mode, then the learning rule should credit
direction at the source->mid stage (and/or use a direction-contrast-aware
credit), not a single scalar over all edges.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld
from embodiment.vision_bridge import VisionBridge
from experiments.plasticity.engine import PlasticPathway
from experiments.plasticity.train import PathwayDecoder, make_lr_shot


def episode_eligibility(world, brain, vision, pathway, shot):
    """Run one FIXED-pose episode and return the final eligibility vector."""
    world.reset(shot)
    pathway.clear_eligibility()
    fly = world.fly; rq = fly.root_qadr; rd = fly.root_dofadr
    root = world.data.qpos[rq:rq + 7].copy()
    n = 0
    while world.result is None and n < 60:
        vision.perceive(); brain.step(20.0); pathway.accumulate()
        world.fly.set_command(0, 0, 0); world.step(0.02)
        world.data.qpos[rq:rq + 7] = root; world.data.qvel[rd:rd + 6] = 0.0
        n += 1
    return pathway.eligibility.copy()


def main():
    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=21)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left")
    pathway = PlasticPathway(brain, elig_mode="subthreshold")
    m = np.load(ROOT / "workspace/outputs/plasticity/plasticity_mask.npz", allow_pickle=True)
    stage = m["stage"]
    src_mask = stage == "source_to_mid"
    md_mask = stage == "mid_to_dn"

    rng = np.random.default_rng(21)
    def mean_elig(side, trials=5):
        accs = []
        for _ in range(trials):
            shot = make_lr_shot(world, side, rng, aim_mag=0.3, speed_range=(4.0, 5.0))
            accs.append(episode_eligibility(world, brain, vision, pathway, shot))
        return np.mean(accs, axis=0)

    eL = mean_elig("left"); eR = mean_elig("right")
    diff = eL - eR
    result = {}
    for name, mask in [("source_to_mid", src_mask), ("mid_to_dn", md_mask)]:
        el = eL[mask]; er = eR[mask]; df = diff[mask]
        # how direction-differentiated is the eligibility on this stage?
        result[name] = dict(
            n_edges=int(mask.sum()),
            mean_elig_left=round(float(el.mean()), 4),
            mean_elig_right=round(float(er.mean()), 4),
            mean_abs_diff=round(float(np.abs(df).mean()), 4),
            frac_diff_over_mean=round(float(np.abs(df).mean() /
                                            (0.5 * (el.mean() + er.mean()) + 1e-9)), 4),
            n_edges_dir_diff=int((np.abs(df) > 0.5 * np.abs(el + er).mean()).sum()),
        )
    (ROOT / "workspace/outputs/plasticity/eligibility_diag.json").write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\ninterpretation: a large frac_diff_over_mean on source_to_mid means "
          "the directional signal IS present in that stage's eligibility and the "
          "credit rule should exploit it; a near-zero value everywhere means the "
          "eligibility is common-mode and scalar reward cannot create direction.")


if __name__ == "__main__":
    main()
