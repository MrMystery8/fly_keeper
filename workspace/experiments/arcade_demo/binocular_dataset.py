"""Generate the binocular Arcade dataset (candidate-pool features, 2-axis teacher).

Same corrected pipeline as arcade_dataset.py (collect only while shot_live,
per-episode manifest, integrity assertions, same shots/seeds/teacher), with TWO
differences, both deliberate:

  1. VISION: BinocularVisionBridge(condition="both") -- left-eye receptors sample
     eye_left, right-eye receptors sample eye_right (true two-eye input).
  2. FEATURES: records the 600-neuron BILATERAL candidate pool (binocular_pool.npz)
     rather than the frozen 207. Final feature selection happens later, from
     TRAINING DATA ONLY (binocular_train.py).

Everything else (shot grid, seeds, 4x20ms windows, teacher labels, split-by-seed)
matches the v2 dataset so one-eye-v2 vs binocular is a clean comparison. Metal
backend so features match the arcade runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S, MAX_DECISIONS
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
# Store the longest causal history once; analyses may use its newest 1/2/4/6/8
# bins without recollecting data.  The ring buffer is reset for every episode.
N_WINDOWS = 8
GROUPS = ("left", "center", "right")
# Named, balanced Arcade training cells.  MID is deliberately not inferred from
# a random height; it is an explicit part of every complete-episode split.
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))


def height_class(height_frac):
    if height_frac < 0.25:
        return "low"
    if height_frac < 0.7:
        return "mid"
    return "high"


def load_pool():
    m = np.load(OUT / "binocular_pool.npz", allow_pickle=False)
    return (m["body_ids"].astype(np.int64), m["somaSide"].astype(str),
            m["type"].astype(str))


def generate(per_cell=10, base_seed=1000, backend="metal",
             out_name="binocular_balanced_dataset.npz", condition="both",
             n_windows=N_WINDOWS, retina_map="viewport"):
    body_ids, sides, types = load_pool()
    body_list = [int(x) for x in body_ids]
    brain = MaleCNSBrain(backend=backend)

    feats, lat, vert = [], [], []
    meta_rows, manifest = [], []
    episodes = 0
    seed = base_seed
    t0 = time.perf_counter()
    for group in GROUPS:
        for hname, hf in HEIGHTS:
            for _ in range(per_cell):
                world = ArcadeGoalkeeperWorld(seed=seed)
                vision = BinocularVisionBridge(world.fly, brain,
                                               condition=condition,
                                               retina_map=retina_map)
                shot = world.sample_shot(group, height_frac=hf)
                world.reset(shot); brain.reset()
                # Stable teacher destination labels are properties of the shot,
                # not a later, possibly post-impact extrapolation of a ball
                # that is no longer on its goalward flight.  They remain labels
                # only; no bridge ever receives them at inference.
                shot_y_cross = float(shot.y_target)
                shot_z_cross = float(shot.height)
                # No temporal state is permitted to cross an episode boundary.
                buf = [np.zeros(len(body_list), np.float32) for _ in range(n_windows)]
                n = 0; ep_start = len(feats); last = -1
                while world.shot_live and n < MAX_DECISIONS:
                    vision.perceive(); brain.step(DECISION_MS)
                    rd = brain.read(body_list)
                    counts = np.array([rd[i]["spikes"] for i in body_list], np.float32)
                    buf.append(counts); buf.pop(0)
                    x = np.concatenate(buf[::-1]).astype(np.float32)
                    (u_lat, u_vert), intercept = teacher_command(world)
                    feats.append(x); lat.append(u_lat); vert.append(u_vert)
                    # The interception target is a LABEL only.  It is retained
                    # to compare "where" vs instantaneous-control semantics;
                    # it is never available to bridge inference.
                    meta_rows.append((seed, group, hname, hf, n,
                                      shot_y_cross, shot_z_cross)); last = n
                    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
                    world.step(DECISION_S); n += 1
                diag = world.outcome_diagnostics()
                ts = diag.get("terminal_step")
                manifest.append(dict(
                    episode_id=episodes, shot_seed=int(seed), shot_class=group,
                    height_class=hname,
                    height_frac=float(hf), episode_length=int(len(feats) - ep_start),
                    terminal_result=world.result,
                    terminal_step=(int(ts) if ts is not None else None),
                    last_step_collected=int(last),
                    reached_max=bool(n >= MAX_DECISIONS),
                    eye_configuration=condition,
                    retina_map=retina_map,
                    n_temporal_bins=int(n_windows),
                    sample_start=int(ep_start), sample_end_exclusive=int(len(feats))))
                episodes += 1; seed += 1
                if episodes % 5 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {episodes} eps, {len(feats)} steps, {el:.0f}s "
                          f"({el/episodes:.1f}s/ep)", flush=True)
    brain.close()

    # integrity: no padding / no post-terminal
    bad = [m for m in manifest
           if m["terminal_result"] is None or m["reached_max"]
           or (m["terminal_step"] is not None and m["last_step_collected"] > m["terminal_step"])]
    if bad:
        raise AssertionError(f"binocular dataset integrity FAILED: {bad[:3]}")
    expected = {(g, h): per_cell for g in GROUPS for h, _ in HEIGHTS}
    observed = Counter((m["shot_class"], m["height_class"]) for m in manifest)
    if dict(observed) != expected:
        raise AssertionError(f"3x3 balance FAILED: expected={expected}, observed={dict(observed)}")
    if any(m["episode_length"] <= 0 or m["sample_end_exclusive"] - m["sample_start"] != m["episode_length"]
           for m in manifest):
        raise AssertionError("episode manifest/sample boundaries FAILED")

    feats = np.asarray(feats, np.float32)
    lat = np.asarray(lat, np.float32); vert = np.asarray(vert, np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / out_name, features=feats, u_lat=lat, u_vert=vert,
             pool_body_ids=body_ids, pool_side=sides, pool_type=types,
             n_windows=np.array([n_windows], dtype=np.int32),
             seeds=np.array([m[0] for m in meta_rows]),
             groups=np.array([m[1] for m in meta_rows]),
             height_classes=np.array([m[2] for m in meta_rows]),
             heights=np.array([m[3] for m in meta_rows]),
             steps=np.array([m[4] for m in meta_rows]),
             teacher_y_cross=np.array([m[5] for m in meta_rows], dtype=np.float32),
             teacher_z_cross=np.array([m[6] for m in meta_rows], dtype=np.float32),
             condition=np.array([condition]),
             retina_map=np.array([retina_map]),
             dataset_contract=np.array(["balanced_3x3_complete_episode_v1"]))
    lens = np.array([m["episode_length"] for m in manifest])
    summary = dict(episodes=episodes, steps=int(feats.shape[0]),
                   pool_size=int(len(body_list)), n_windows=int(n_windows),
                   feature_dim=int(feats.shape[1]), backend=backend,
                   vision=condition, retina_map=retina_map,
                   cells={f"{g}/{h}": int(per_cell)
                                             for g in GROUPS for h, _ in HEIGHTS},
                   episode_length=dict(min=int(lens.min()),
                                       median=float(np.median(lens)),
                                       mean=round(float(lens.mean()), 2),
                                       max=int(lens.max())),
                   terminal_results=dict(Counter(m["terminal_result"] for m in manifest)),
                   integrity_ok=True,
                   wall_seconds=round(time.perf_counter() - t0, 1))
    (OUT / out_name.replace(".npz", "_summary.json")).write_text(json.dumps(summary, indent=2))
    (OUT / out_name.replace(".npz", "_manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--backend", default="metal")
    p.add_argument("--condition", default="both")
    p.add_argument("--out", default="binocular_balanced_dataset.npz")
    p.add_argument("--n-windows", type=int, default=N_WINDOWS)
    p.add_argument("--retina-map", default="viewport",
                   choices=["viewport", "fullframe"])
    a = p.parse_args()
    generate(per_cell=a.per_cell, base_seed=a.base_seed, backend=a.backend,
             out_name=a.out, condition=a.condition, n_windows=a.n_windows,
             retina_map=a.retina_map)


if __name__ == "__main__":
    main()
