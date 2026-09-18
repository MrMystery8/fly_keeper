"""Shot-validity oracle (Phase 1).

HARD PREREQUISITE for RL: every sampled training/evaluation penalty must be a
GENUINE goal threat -- i.e. it would score against a passive/absent keeper.
Otherwise RL can learn to exploit naturally-missed shots (a high ball that hits
the crossbar, a wide ball, a ball that stalls) and receive SAVE reward for an
outcome that would have happened with NO keeper.

This module simulates each shot with a GENUINELY ABSENT keeper:
  * the fly body is teleported far off the pitch (cannot physically block/deflect)
  * the non-contact reach interception is disabled (`world.disable_keeper()`)
  * the ball's freshly-launched qpos/qvel are preserved across the teleport
and then classifies the ball's trajectory geometrically against the goal frame:

    CLEAN_GOAL  - ball centre crosses x<=GOAL_LINE_X inside the posts & below the
                  bar, with its EDGE clearing the frame (radius margin) -> valid
    MISS_WIDE   - crosses the plane but |y| outside the posts
    MISS_HIGH   - crosses the plane but above the crossbar
    POST        - ends up laterally against a post without crossing
    CROSSBAR    - ends up vertically against the crossbar without crossing
    STALL       - loses its energy in front of the goal without crossing
    OTHER       - anything else (should be empty for accepted cells)

Only CLEAN_GOAL is accepted for normal keeper training/evaluation.

The oracle uses privileged simulator state (it IS the environment); this is legal
because it only labels/filters shots -- it never feeds the controller.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from embodiment.mujoco_world import (GOAL_LINE_X, GOAL_HALF_WIDTH, GOAL_HEIGHT,
                                     BALL_RADIUS, POST_RADIUS)
from experiments.arcade_demo.arcade_world import (ArcadeGoalkeeperWorld,
                                                  INNER_HALF_WIDTH, INNER_TOP)
from experiments.arcade_demo.arcade_runtime import DECISION_S, MAX_DECISIONS

CLEAN_GOAL = "CLEAN_GOAL"
STALL_SPEED = 1.2           # matches world stall threshold


def simulate_absent_keeper(world, shot, max_steps=MAX_DECISIONS):
    """Run `shot` with a genuinely absent keeper; return (result, trajectory).

    trajectory is a list of (x, y, z, vx, vy, vz) ball samples.
    """
    world.reset(shot)
    qadr, dofadr = world._ball_qadr, world._ball_dofadr
    ball_q = world.data.qpos[qadr:qadr + 7].copy()
    ball_v = world.data.qvel[dofadr:dofadr + 6].copy()
    # teleport the fly far off the pitch so it can never physically touch the ball
    world.fly.set_pose(xy=(0.0, 60.0), yaw=0.0)
    world.data.qpos[qadr:qadr + 7] = ball_q
    world.data.qvel[dofadr:dofadr + 6] = ball_v
    world.disable_keeper()                       # no reach; flat-loft still holds
    mujoco.mj_forward(world.model, world.data)
    traj = []
    n = 0
    while world.result is None and n < max_steps:
        world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
        world.step(DECISION_S)
        b = world._observe_ball()
        traj.append((float(b["pos"][0]), float(b["pos"][1]), float(b["pos"][2]),
                     float(b["vel"][0]), float(b["vel"][1]), float(b["vel"][2])))
        n += 1
    return world.result, traj


def classify_trajectory(traj):
    """Geometric classification of an absent-keeper ball trajectory.

    CLEAN_GOAL requires the ball CENTRE to cross the goal-line plane inside the
    EFFECTIVE inner volume (so the ball edge clears the posts/crossbar), i.e.
    |y| <= INNER_HALF_WIDTH and z <= INNER_TOP. A centre that crosses inside the
    RAW frame but outside the inner volume grazes the frame -> POST/CROSSBAR.
    """
    if not traj:
        return "OTHER"
    for (x, y, z, vx, vy, vz) in traj:
        if x <= GOAL_LINE_X:
            if abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT:
                # crossed inside the raw frame -- clean only if edge clears it
                if abs(y) <= INNER_HALF_WIDTH and z <= INNER_TOP:
                    return CLEAN_GOAL
                return "CROSSBAR" if z > INNER_TOP else "POST"
            return "MISS_HIGH" if z > GOAL_HEIGHT else "MISS_WIDE"
    xf, yf, zf, vxf, vyf, vzf = traj[-1]
    if abs(abs(yf) - GOAL_HALF_WIDTH) <= (POST_RADIUS + BALL_RADIUS):
        return "POST"
    if abs(zf - GOAL_HEIGHT) <= (POST_RADIUS + BALL_RADIUS):
        return "CROSSBAR"
    if float(np.hypot(np.hypot(vxf, vyf), vzf)) < STALL_SPEED:
        return "STALL"
    return "OTHER"


def classify_shot(seed, group, height_frac):
    """Build a world for `seed`, sample the cell shot, classify it (absent keeper)."""
    w = ArcadeGoalkeeperWorld(seed=seed)
    shot = w.sample_shot(group, height_frac=height_frac)
    result, traj = simulate_absent_keeper(w, shot)
    cls = classify_trajectory(traj)
    return cls, shot, traj


def is_valid_shot(seed, group, height_frac):
    """True iff this exact (seed, group, height_frac) shot is a CLEAN_GOAL."""
    cls, _, _ = classify_shot(seed, group, height_frac)
    return cls == CLEAN_GOAL


def accept_cells(cells):
    """Filter a list of (seed, group, hname, hf) cells to CLEAN_GOAL-only.

    Returns (accepted, rejected) where rejected carries the reject label.
    """
    accepted, rejected = [], []
    for seed, g, hname, hf in cells:
        cls, _, _ = classify_shot(seed, g, hf)
        if cls == CLEAN_GOAL:
            accepted.append((seed, g, hname, hf))
        else:
            rejected.append((seed, g, hname, hf, cls))
    return accepted, rejected


def resample_valid_cell(group, hname, hf, start_seed, max_tries=200):
    """Find the first seed >= start_seed whose shot for this cell is CLEAN_GOAL."""
    s = start_seed
    for _ in range(max_tries):
        if is_valid_shot(s, group, hf):
            return s
        s += 1
    raise RuntimeError(f"no valid shot for {group}/{hname} in {max_tries} tries "
                       f"from seed {start_seed}")


GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 1.0))


def classify_random_shot(seed):
    """Sample a continuously-randomized shot for `seed`; classify (absent keeper)."""
    w = ArcadeGoalkeeperWorld(seed=seed)
    shot = w.sample_shot_random()
    result, traj = simulate_absent_keeper(w, shot)
    return classify_trajectory(traj), shot


def randomized_valid_shots(n, base_seed, max_tries_mult=6):
    """Return n (seed, shot) pairs whose randomized shots are all CLEAN_GOAL.

    Broadens the training distribution beyond the fixed 3x3 grid with continuous
    lateral/height/speed jitter; each accepted shot still scores against a
    passive/absent keeper (hard prerequisite preserved).
    """
    out = []
    seed = base_seed
    tries = 0
    while len(out) < n and tries < n * max_tries_mult:
        cls, shot = classify_random_shot(seed)
        if cls == CLEAN_GOAL:
            out.append((seed, shot))
        seed += 1
        tries += 1
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} valid randomized shots from "
                           f"{base_seed} in {tries} tries")
    return out


# Stratified continuous sampling: LEFT/CENTER/RIGHT x LOW/MID/HIGH cells with
# continuous jitter INSIDE each stratum. Guarantees balanced coverage (esp.
# real LOW shots, which unrestricted uniform-height sampling under-represents),
# while every accepted shot still passes CLEAN_GOAL. Height bands are given in
# height_frac (0..1): LOW ~ rolling, HIGH ~ upper interior with crossbar margin.
STRATA_GROUPS = ("left", "center", "right")
STRATA_HEIGHTS = (("low", (0.0, 0.20)), ("mid", (0.40, 0.60)),
                  ("high", (0.80, 1.00)))
# lateral bands (y_target, cm) inside the interior; center kept genuinely central
STRATA_LAT = {"left": (0.45, 1.00), "center": (-0.15, 0.15), "right": (-1.00, -0.45)}


def stratified_valid_shots(per_cell, base_seed, max_tries_mult=40):
    """Balanced LEFT/CENTER/RIGHT x LOW/MID/HIGH CLEAN_GOAL shots w/ continuous jitter.

    For each of the 9 strata, draw continuous (y_target, height_frac, speed)
    inside the stratum's ranges until `per_cell` shots pass CLEAN_GOAL. Uses a
    world's RNG per candidate for the ball jitter; the ShotSpec is built directly
    from the drawn continuous values (bypassing the discrete grid sampler).
    Returns a flat list of (seed, shot) with balanced strata; the shot.group /
    height already reflect the drawn stratum.
    """
    from embodiment.mujoco_world import ShotSpec
    from experiments.arcade_demo.arcade_world import height_for_frac
    out = []
    seed = base_seed
    rng = np.random.default_rng(base_seed)
    for g in STRATA_GROUPS:
        lo_y, hi_y = STRATA_LAT[g]
        for hname, (lo_h, hi_h) in STRATA_HEIGHTS:
            got = 0; tries = 0
            while got < per_cell and tries < per_cell * max_tries_mult:
                y = float(rng.uniform(lo_y, hi_y))
                hf = float(rng.uniform(lo_h, hi_h))
                spd = float(rng.uniform(4.5, 6.0))
                shot = ShotSpec(g, y, spd, float(height_for_frac(hf)))
                w = ArcadeGoalkeeperWorld(seed=seed)
                _, traj = simulate_absent_keeper(w, shot)
                if classify_trajectory(traj) == CLEAN_GOAL:
                    out.append((seed, shot)); got += 1
                seed += 1; tries += 1
            if got < per_cell:
                raise RuntimeError(f"{g}/{hname}: only {got}/{per_cell} valid "
                                   f"shots in {tries} tries")
    return out


def validation_table(per_cell=8, base_seed=90000):
    """Classify every nominal cell shot vs an absent keeper; return a report."""
    rows, seed = [], base_seed
    cell = defaultdict(lambda: defaultdict(int))
    heights_z = defaultdict(list)
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            for _ in range(per_cell):
                cls, shot, traj = classify_shot(seed, g, hf)
                cell[(g, hname)][cls] += 1
                # record the crossing height for a clean goal
                if cls == CLEAN_GOAL:
                    for (x, y, z, *_ ) in traj:
                        if x <= GOAL_LINE_X:
                            heights_z[hname].append(z); break
                rows.append((g, hname, seed, round(shot.y_target, 3),
                             round(shot.height, 3), cls))
                seed += 1
    total = len(rows)
    clean = sum(1 for r in rows if r[-1] == CLEAN_GOAL)
    return dict(
        per_cell=per_cell, base_seed=base_seed, total=total,
        clean_goal=clean, clean_rate=round(clean / total, 3),
        by_cell={f"{g}/{h}": dict(cnts) for (g, h), cnts in sorted(cell.items())},
        crossing_z={h: dict(mean=round(float(np.mean(v)), 3),
                            min=round(float(np.min(v)), 3),
                            max=round(float(np.max(v)), 3), n=len(v))
                    for h, v in sorted(heights_z.items()) if v},
        rows=rows,
    )


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=90000)
    a = p.parse_args()
    rep = validation_table(per_cell=a.per_cell, base_seed=a.base_seed)
    print("=== SHOT-VALIDITY VALIDATION TABLE (passive/absent keeper) ===")
    print(f"goal: inner |y|<={INNER_HALF_WIDTH:.3f}  z<={INNER_TOP:.3f}  "
          f"(raw hw={GOAL_HALF_WIDTH} h={GOAL_HEIGHT}, ball_r={BALL_RADIUS}, "
          f"post_r={POST_RADIUS})")
    print(f"per-cell={rep['per_cell']}  total={rep['total']}  "
          f"CLEAN_GOAL={rep['clean_goal']}/{rep['total']} "
          f"({rep['clean_rate']:.1%})\n")
    for cell, cnts in rep["by_cell"].items():
        summary = " ".join(f"{k}:{v}" for k, v in sorted(cnts.items()))
        flag = "" if set(cnts) == {CLEAN_GOAL} else "   <-- NOT ALL CLEAN"
        print(f"  {cell:14s} {summary}{flag}")
    print("\ncrossing height by band (clean goals):")
    for h, s in rep["crossing_z"].items():
        print(f"  {h:5s} mean_z={s['mean']}  range=[{s['min']},{s['max']}]  n={s['n']}")
    out = ROOT / "workspace" / "outputs" / "arcade_demo"
    out.mkdir(parents=True, exist_ok=True)
    rep_json = {k: v for k, v in rep.items() if k != "rows"}
    (out / "shot_validity_table.json").write_text(json.dumps(rep_json, indent=2))
    print(f"\nsaved shot_validity_table.json")
    ok = all(set(c) == {CLEAN_GOAL} for c in rep["by_cell"].values())
    print("\nRESULT:", "ALL NOMINAL CELLS ARE CLEAN GOALS ✓" if ok
          else "SOME CELLS NOT CLEAN — recalibration needed ✗")


if __name__ == "__main__":
    main()
