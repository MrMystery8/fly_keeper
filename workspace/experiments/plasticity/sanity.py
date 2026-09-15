"""Short sanity test for the targeted plasticity rule.

Trains a small batch on the LEFT/RIGHT-ONLY easy curriculum (plasticity +
exploration ON), then probes -- with exploration OFF and the fly held fixed --
whether the target descending populations became genuinely DIRECTIONAL and
vision-dependent. Also reports weight change (bounded, in-mask) and a no-move
decoder-only closed-loop check before vs after.

Success signals to look for:
  - plastic weights change, stay in [0.1, 2.0]x baseline, no NaN
  - DN left/right response separates for left vs right shots AFTER training
    (it does not before, per the audit)
  - that separation collapses under BLIND input (vision-dependent)
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
from experiments.plasticity.train import (PathwayDecoder, run_episode, make_lr_shot,
                                          LEFT_DN, RIGHT_DN)


def dn_directional_probe(world, brain, vision, decoder, trials=6, condition="normal",
                         aim_mag=0.3, seed=100):
    """With the fly FIXED, present left vs right shots and measure the target-DN
    left/right response difference (the decoder's centered signal). Returns the
    mean centered DN signal for left vs right shots -> directional separation."""
    vision.condition = condition
    rng = np.random.default_rng(seed)
    out = {}
    for side in ("left", "right"):
        centered_vals = []
        for _ in range(trials):
            shot = make_lr_shot(world, side, rng, aim_mag=aim_mag, speed_range=(4.0, 5.0))
            world.reset(shot)
            decoder.reset()
            # hold the fly fixed; just accumulate DN readout over the approach
            fly = world.fly
            rq = fly.root_qadr; rd = fly.root_dofadr
            root = world.data.qpos[rq:rq + 7].copy()
            signal = 0.0; nwin = 0
            n = 0
            while world.result is None and n < 60:
                vision.perceive()
                brain.step(20.0)
                act = brain.read(decoder.readout_ids())
                _, diag = decoder.decode(act)
                signal += diag["centered"]; nwin += 1
                # keep fly fixed (open-loop directional probe)
                world.fly.set_command(0, 0, 0)
                world.step(0.02)
                world.data.qpos[rq:rq + 7] = root
                world.data.qvel[rd:rd + 6] = 0.0
                n += 1
            centered_vals.append(signal / max(1, nwin))
        out[side] = float(np.mean(centered_vals))
    vision.condition = "normal"
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-episodes", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--aim", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--credit", choices=["scalar", "directional"], default="scalar")
    a = p.parse_args()

    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=a.seed)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left")
    decoder = PathwayDecoder()
    pathway = PlasticPathway(brain, lr=a.lr, elig_mode="subthreshold")
    rng = np.random.default_rng(a.seed)

    # ---- BEFORE: DN directional separation (normal + blind) ----
    before_normal = dn_directional_probe(world, brain, vision, decoder,
                                         condition="normal", aim_mag=a.aim, seed=100)
    before_blind = dn_directional_probe(world, brain, vision, decoder,
                                        condition="blind", aim_mag=a.aim, seed=100)
    print("BEFORE  normal:", {k: round(v, 3) for k, v in before_normal.items()},
          " blind:", {k: round(v, 3) for k, v in before_blind.items()})

    # ---- TRAIN (plasticity + exploration ON, L/R-only easy curriculum) ----
    train_results = []
    for ep in range(a.train_episodes):
        side = "left" if ep % 2 == 0 else "right"
        shot = make_lr_shot(world, side, rng, aim_mag=a.aim, speed_range=(3.5, 4.5))
        res, _ = run_episode(world, brain, vision, decoder, pathway, shot,
                             learn=True, explore=0.9, rng=rng, credit=a.credit)
        train_results.append(res)
    from collections import Counter
    print("TRAIN results:", dict(Counter(train_results)))
    print("weight_report:", pathway.weight_report())

    # ---- AFTER: DN directional separation (normal + blind) ----
    after_normal = dn_directional_probe(world, brain, vision, decoder,
                                        condition="normal", aim_mag=a.aim, seed=100)
    after_blind = dn_directional_probe(world, brain, vision, decoder,
                                       condition="blind", aim_mag=a.aim, seed=100)
    print("AFTER   normal:", {k: round(v, 3) for k, v in after_normal.items()},
          " blind:", {k: round(v, 3) for k, v in after_blind.items()})

    def sep(d):
        return round(d["left"] - d["right"], 4)
    result = dict(
        train_episodes=a.train_episodes, lr=a.lr, aim=a.aim,
        train_save_rate=round(sum(r == "SAVE" for r in train_results) / len(train_results), 3),
        before_normal=before_normal, after_normal=after_normal,
        before_blind=before_blind, after_blind=after_blind,
        before_sep_normal=sep(before_normal), after_sep_normal=sep(after_normal),
        before_sep_blind=sep(before_blind), after_sep_blind=sep(after_blind),
        weight_report=pathway.weight_report(),
    )
    OUT = ROOT / "workspace" / "outputs" / "plasticity"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps(result, indent=2))
    print("\n=== directional separation (left_signal - right_signal) ===")
    print(f"  normal: before {result['before_sep_normal']:+.4f} -> after {result['after_sep_normal']:+.4f}")
    print(f"  blind:  before {result['before_sep_blind']:+.4f} -> after {result['after_sep_blind']:+.4f}")
    print("interpretation: want after_normal separation to grow AND stay near 0 for blind")


if __name__ == "__main__":
    main()
