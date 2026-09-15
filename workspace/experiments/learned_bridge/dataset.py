"""Episode-level dataset generation for the learned bridge.

For each penalty episode we run the fixed MaleCNS closed loop with the BRIDGE
OFF (Natural MaleCNS -- the fly barely moves, so the retinal stream is the
natural approaching-shot view), and at every 20 ms decision t we record:

  * per-window spike counts of ALL selected optic-lobe neurons (the 207-neuron
    candidate pool) for that 20 ms window -- stored RAW per window so any
    temporal-window construction, feature-count subset, and alignment offset can
    be built offline WITHOUT re-simulating the expensive brain.
  * the heuristic TEACHER's continuous lateral command u_teacher in [-1,1] at t,
    computed from the TRUE ball state (privileged -- labels only).
  * true ball/fly state at t (for alignment + split bookkeeping; NEVER a bridge
    input feature).

The teacher label is a continuous signed command:
    -1 = strong left, 0 = neutral, +1 = strong right   (DECODER convention:
    the DescendingMotorDecoder / DNMotorBasis use u<0 = strafe left). The
    HeuristicController's `lateral` field already uses lateral>0 = strafe left,
    so u_teacher = -lateral to match the decoder-side convention used by the
    bridge/DN basis.

Splitting is by COMPLETE EPISODE / shot seed (Section 21): each episode's
windows go entirely to train, val, or test. We record the seed, group, speed,
and aim of every episode so splits are reproducible and stratifiable.

Bridge inputs at inference use ONLY the recorded neural spike counts. Ball/fly
state is stored for labels and analysis and is dropped from the feature matrix.
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

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X, GOAL_HALF_WIDTH
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.embodied_flykeeper.controllers import HeuristicController
from experiments.learned_bridge.dn_basis import LEFT_DN, RIGHT_DN

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
DECISION_MS = 20.0
DECISION_S = DECISION_MS / 1000.0
MAX_DECISIONS = 75


def teacher_command(world, turn_gain=2.5):
    """Continuous teacher label u in [-1,1] from TRUE ball state (labels only).

    Reproduces HeuristicController's predicted goal-line crossing, then maps the
    lateral error to a signed command in the DECODER convention (u<0 = strafe
    left toward +y). HeuristicController uses lateral>0 = strafe left, so we
    negate to get the decoder-side u.
    """
    ball = world._observe_ball()
    bx, by, _ = ball["pos"]
    vx, vy, _ = ball["vel"]
    if vx < -1e-3:
        t = (GOAL_LINE_X - bx) / vx
        y_cross = by + vy * t
    else:
        y_cross = by
    fly_y = world.fly.position[1]
    err = y_cross - fly_y                      # >0 => target is to the left (+y)
    lateral = float(np.clip(turn_gain * err, -1, 1))   # >0 = strafe left
    u = -lateral                               # decoder convention: u<0 = left
    return u, dict(y_cross=float(y_cross), fly_y=float(fly_y), err=float(err))


def generate(seeds, *, camera="eye_left", n_windows_store=None, verbose=True):
    """Run bridge-OFF episodes and collect per-window selected-neuron counts +
    teacher labels. `seeds` is a list of (episode_seed, group) tuples.

    Returns a dict of arrays ready to save. Feature windows are stored raw:
    counts[episode] is (n_steps, n_selected) int16 (spikes in THAT 20 ms window).
    """
    from adapters.brain import MaleCNSBrain
    man = np.load(OUT / "visual_manifest.npz")
    sel_graph = man["graph_index"].astype(np.int64)     # 207 selected neurons
    sel_body = man["body_ids"].astype(np.int64)
    n_sel = len(sel_graph)

    world = GoalkeeperWorld(seed=1)
    brain = MaleCNSBrain(backend="cpu")
    vision = VisionBridge(world.fly, brain, camera=camera, condition="normal")
    decoder = DescendingMotorDecoder(left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
                                     turn_gain=2.5, forward_bias=0.0, smoothing=0.5)

    episodes = []
    t0 = time.perf_counter()
    for ei, (seed, group) in enumerate(seeds):
        # deterministic shot from this episode's seed
        w2 = GoalkeeperWorld(seed=seed)
        shot = w2.sample_shot(group)
        world.reset(shot)
        decoder.reset()
        counts_steps = []          # per-step (n_sel,) window spike counts
        labels = []                # teacher u per step
        ball_y = []; fly_y = []; ball_x = []
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            # Natural MaleCNS closed loop (BRIDGE OFF): vision -> step -> readDN -> decode
            vision.perceive()
            brain.step(DECISION_MS)
            win_counts = brain._brain.counts[sel_graph].astype(np.int16)
            counts_steps.append(win_counts)
            # teacher label from TRUE state at this decision time
            u, tdiag = teacher_command(world)
            labels.append(u)
            b = world._observe_ball()
            ball_x.append(float(b["pos"][0])); ball_y.append(float(b["pos"][1]))
            fly_y.append(float(world.fly.position[1]))
            # advance body under the natural decoder command
            act = brain.read(decoder.readout_ids())
            cmd, _ = decoder.decode(act)
            world.fly.set_command(cmd["forward"], 0.0, cmd["gait_on"],
                                  lateral=cmd["lateral"])
            world.step(DECISION_S)
            n += 1
        result = world.result or "GOAL"
        episodes.append(dict(
            seed=seed, group=group, result=result, n_steps=n,
            aim=float(shot.y_target), speed=float(shot.speed),
            counts=np.asarray(counts_steps, dtype=np.int16),   # (n, n_sel)
            u_teacher=np.asarray(labels, dtype=np.float32),     # (n,)
            ball_x=np.asarray(ball_x, dtype=np.float32),
            ball_y=np.asarray(ball_y, dtype=np.float32),
            fly_y=np.asarray(fly_y, dtype=np.float32),
        ))
        if verbose and (ei + 1) % 10 == 0:
            dt = time.perf_counter() - t0
            print(f"  {ei+1}/{len(seeds)} episodes  ({dt:.0f}s, "
                  f"{dt/(ei+1):.1f}s/ep)")
    brain.close()

    return dict(sel_graph=sel_graph, sel_body=sel_body, n_sel=n_sel,
                episodes=episodes, decision_ms=DECISION_MS)


def build_seed_list(n_per_group, groups=("left", "center", "right"),
                    base_seed=1000):
    """Deterministic (seed, group) list, stratified by group."""
    seeds = []
    s = base_seed
    for g in groups:
        for _ in range(n_per_group):
            seeds.append((s, g)); s += 1
    return seeds


def save_dataset(data, path):
    """Save as a compressed npz with a ragged episode layout (object arrays)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    eps = data["episodes"]
    np.savez_compressed(
        path,
        sel_graph=data["sel_graph"], sel_body=data["sel_body"],
        n_sel=data["n_sel"], decision_ms=data["decision_ms"],
        seeds=np.array([e["seed"] for e in eps], dtype=np.int64),
        groups=np.array([e["group"] for e in eps]),
        results=np.array([e["result"] for e in eps]),
        n_steps=np.array([e["n_steps"] for e in eps], dtype=np.int32),
        aim=np.array([e["aim"] for e in eps], dtype=np.float32),
        speed=np.array([e["speed"] for e in eps], dtype=np.float32),
        counts=np.array([e["counts"] for e in eps], dtype=object),
        u_teacher=np.array([e["u_teacher"] for e in eps], dtype=object),
        ball_x=np.array([e["ball_x"] for e in eps], dtype=object),
        ball_y=np.array([e["ball_y"] for e in eps], dtype=object),
        fly_y=np.array([e["fly_y"] for e in eps], dtype=object),
    )
    return str(path)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-group", type=int, default=20,
                   help="episodes per group (left/center/right)")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--out", type=str, default="dataset_main.npz")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    seeds = build_seed_list(a.per_group, base_seed=a.base_seed)
    print(f"[dataset] generating {len(seeds)} episodes "
          f"({a.per_group}/group) bridge-OFF ...")
    data = generate(seeds)
    path = save_dataset(data, OUT / a.out)

    # quick summary
    eps = data["episodes"]
    tot_steps = sum(e["n_steps"] for e in eps)
    us = np.concatenate([e["u_teacher"] for e in eps])
    summary = dict(
        n_episodes=len(eps), total_steps=int(tot_steps),
        n_selected_neurons=int(data["n_sel"]),
        steps_per_episode_mean=round(tot_steps / len(eps), 1),
        teacher_u=dict(min=round(float(us.min()), 3), max=round(float(us.max()), 3),
                       mean=round(float(us.mean()), 3),
                       frac_left=round(float((us < -0.1).mean()), 3),
                       frac_neutral=round(float((np.abs(us) <= 0.1).mean()), 3),
                       frac_right=round(float((us > 0.1).mean()), 3)),
        by_group={g: int(sum(e["group"] == g for e in eps)) for g in ("left", "center", "right")},
        path=path,
    )
    (OUT / (Path(a.out).stem + "_summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
