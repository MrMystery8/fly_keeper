"""Train (optionally) + held-out evaluation of the targeted-plasticity controller.

Evaluation is ALWAYS plasticity OFF, exploration OFF, deterministic, on held-out
seeds distinct from training. Reports the mandatory controls. The connectome is
never modified on disk; a run uses a fresh brain and applies learning only to
the in-memory masked weights, checkpointed separately.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from experiments.plasticity.engine import PlasticPathway
from experiments.plasticity.train import (PathwayDecoder, run_episode, make_lr_shot,
                                          MAX_DECISIONS, DECISION_S)
from experiments.embodied_flykeeper.controllers import (PassiveController,
                                                        RandomController, HeuristicController)

CKPT_DIR = ROOT / "workspace" / "checkpoints"


def eval_neural(world, brain, vision, decoder, *, episodes, seed, condition="normal",
                groups=("left", "center", "right")):
    """Deterministic evaluation of the neural decoder (no plasticity, no explore)."""
    vision.condition = condition
    by = {g: [0, 0] for g in groups}
    for ep in range(episodes):
        g = str(np.random.default_rng(seed + ep).choice(groups))
        shot = world.sample_shot(g)
        world.reset(shot)
        decoder.reset()
        if condition == "static_ball":
            vision.set_static_reference(vision.render())
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            vision.perceive(); brain.step(20.0)
            act = brain.read(decoder.readout_ids())
            cmd, _ = decoder.decode(act)
            world.fly.set_command(0.0, 0.0, cmd["gait_on"], lateral=cmd["lateral"])
            world.step(DECISION_S)
            n += 1
        res = world.result or "GOAL"
        by[g][0] += int(res == "SAVE"); by[g][1] += 1
    vision.condition = "normal"
    tot_s = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    return dict(save_rate=round(tot_s / tot_n, 3),
                by_group={g: round(v[0] / v[1], 3) if v[1] else None for g, v in by.items()})


def eval_simple(controller_kind, world, *, episodes, seed, groups=("left", "center", "right")):
    ctrl = {"passive": PassiveController(), "random": RandomController(seed),
            "heuristic": HeuristicController(GOAL_LINE_X)}[controller_kind]
    by = {g: [0, 0] for g in groups}
    for ep in range(episodes):
        g = str(np.random.default_rng(seed + ep).choice(groups))
        shot = world.sample_shot(g)
        world.reset(shot); ctrl.reset()
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            cmd, _ = ctrl.act(world)
            world.fly.set_command(cmd["forward"], cmd.get("turn", 0), cmd["gait_on"],
                                  lateral=cmd.get("lateral", 0))
            world.step(DECISION_S)
            n += 1
        res = world.result or "GOAL"
        by[g][0] += int(res == "SAVE"); by[g][1] += 1
    tot_s = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    return dict(save_rate=round(tot_s / tot_n, 3),
                by_group={g: round(v[0] / v[1], 3) if v[1] else None for g, v in by.items()})


def train(brain, world, vision, decoder, pathway, *, episodes, lr, aim, credit,
          seed, explore=0.9):
    rng = np.random.default_rng(seed)
    res = []
    for ep in range(episodes):
        side = "left" if ep % 2 == 0 else "right"
        shot = make_lr_shot(world, side, rng, aim_mag=aim, speed_range=(3.5, 4.5))
        r, _ = run_episode(world, brain, vision, decoder, pathway, shot,
                           learn=True, explore=explore, rng=rng, credit=credit)
        res.append(r)
    from collections import Counter
    return dict(Counter(res))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-episodes", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--aim", type=float, default=0.25)
    p.add_argument("--credit", choices=["scalar", "directional"], default="directional")
    p.add_argument("--train-seed", type=int, default=101)
    p.add_argument("--eval-seed", type=int, default=9000)  # held-out, disjoint
    p.add_argument("--run-name", type=str, default="run_directional")
    a = p.parse_args()

    from adapters.brain import MaleCNSBrain
    world = GoalkeeperWorld(seed=a.train_seed)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera="eye_left")
    decoder = PathwayDecoder()
    pathway = PlasticPathway(brain, lr=a.lr, elig_mode="subthreshold")

    out = {"config": vars(a)}

    # baselines + fixed brain (pre-training) on held-out full distribution
    out["passive"] = eval_simple("passive", world, episodes=a.eval_episodes, seed=a.eval_seed)
    out["random"] = eval_simple("random", world, episodes=a.eval_episodes, seed=a.eval_seed)
    out["heuristic"] = eval_simple("heuristic", world, episodes=a.eval_episodes, seed=a.eval_seed)
    out["fixed_malecns"] = eval_neural(world, brain, vision, decoder,
                                       episodes=a.eval_episodes, seed=a.eval_seed)
    print("baselines:", json.dumps({k: out[k]["save_rate"] for k in
          ["passive", "random", "heuristic", "fixed_malecns"]}))

    # train
    t0 = time.perf_counter()
    out["train_results"] = train(brain, world, vision, decoder, pathway,
                                 episodes=a.train_episodes, lr=a.lr, aim=a.aim,
                                 credit=a.credit, seed=a.train_seed)
    out["train_wall_s"] = round(time.perf_counter() - t0, 1)
    out["weight_report"] = pathway.weight_report()
    print("trained:", out["train_results"], out["weight_report"])

    # save checkpoint (masked weights only)
    ckpt = CKPT_DIR / a.run_name / "learned.npz"
    pathway.save_checkpoint(ckpt, metadata={"config": vars(a),
                                            "weight_report": pathway.weight_report()})
    out["checkpoint"] = str(ckpt)

    # evaluate trained decoder (plasticity OFF) + controls on held-out seeds
    for cond in ("normal", "blind", "mirrored", "shuffled"):
        out[f"trained_{cond}"] = eval_neural(world, brain, vision, decoder,
                                             episodes=a.eval_episodes,
                                             seed=a.eval_seed, condition=cond)
        print(f"trained/{cond}:", out[f"trained_{cond}"])

    # frozen pre-training control: reset masked weights to baseline, re-eval
    pathway.reset_to_baseline()
    out["frozen_baseline_recheck"] = eval_neural(world, brain, vision, decoder,
                                                 episodes=a.eval_episodes, seed=a.eval_seed)
    print("frozen baseline recheck:", out["frozen_baseline_recheck"])

    OUT = ROOT / "workspace" / "outputs" / "plasticity"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"evaluate_{a.run_name}.json").write_text(json.dumps(out, indent=2))
    print("\nsaved evaluate_" + a.run_name + ".json")


if __name__ == "__main__":
    main()
