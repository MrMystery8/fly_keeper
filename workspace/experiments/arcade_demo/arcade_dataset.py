"""Generate the arcade closed-loop dataset (2-axis teacher) on the Metal backend.

Runs the arcade world with the bridge OFF (natural body drifts little, so the
retinal stream is the natural approaching shot, exactly as the frozen dataset was
collected). Per 20 ms decision it records:

  * raw per-window spike counts of the 207 selected optic-lobe neurons (the
    SAME neurons/order as the frozen manifest), last 4 windows;
  * the 2-axis TEACHER command (u_lateral, u_vertical) from privileged ball
    state (labels only);
  * true ball/fly state (for alignment/splits only, never a feature).

Shots span low and lofted, left/center/right, so the vertical channel has signal
to learn. Uses the Metal backend so features match the arcade runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from embodiment.vision_bridge import VisionBridge
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S, MAX_DECISIONS

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
N_WINDOWS = 4
GROUPS = ("left", "center", "right")
HEIGHTS = (0.0, 0.85)          # low + lofted


def selected_ids(n_neurons=207):
    """Return (graph_index, body_ids) for the frozen 207-neuron selection.

    On the Metal backend spike counts are GPU-resident (brain._brain.counts is
    NOT populated), so features must be read via brain.read(body_ids). The
    manifest body_ids map exactly to graph_index, so both backends agree.
    """
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    gi = man["graph_index"][:n_neurons].astype(np.int64)
    body = man["body_ids"][:n_neurons].astype(np.int64)
    return gi, body


def generate(per_cell=8, base_seed=1000, backend="metal", n_neurons=207,
             out_name="arcade_dataset.npz"):
    gi, body_ids = selected_ids(n_neurons)
    body_list = [int(x) for x in body_ids]
    brain = MaleCNSBrain(backend=backend)

    feats, lat_labels, vert_labels = [], [], []
    meta_rows = []
    episodes = 0
    seed = base_seed
    t0 = time.perf_counter()
    for group in GROUPS:
        for hf in HEIGHTS:
            for _ in range(per_cell):
                world = ArcadeGoalkeeperWorld(seed=seed)
                vision = VisionBridge(world.fly, brain, camera="eye_left",
                                      condition="normal")
                shot = world.sample_shot(group, height_frac=hf)
                world.reset(shot)
                brain.reset()
                # ring buffer of last N_WINDOWS spike-count vectors
                buf = [np.zeros(len(gi), dtype=np.float32)
                       for _ in range(N_WINDOWS)]
                n = 0
                while world.result is None and n < MAX_DECISIONS:
                    vision.perceive()
                    brain.step(DECISION_MS)
                    # read spikes by BODY ID (works on CPU and Metal)
                    rd = brain.read(body_list)
                    counts = np.array([rd[i]["spikes"] for i in body_list],
                                      dtype=np.float32)
                    buf.append(counts); buf.pop(0)
                    # newest-first temporal feature vector (matches frozen order)
                    x = np.concatenate(buf[::-1]).astype(np.float32)
                    (u_lat, u_vert), _ = teacher_command(world)
                    feats.append(x)
                    lat_labels.append(u_lat)
                    vert_labels.append(u_vert)
                    meta_rows.append((seed, group, hf, n))
                    # bridge OFF: body barely moves; just advance physics
                    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
                    world.step(DECISION_S)
                    n += 1
                episodes += 1
                seed += 1
                if episodes % 5 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {episodes} episodes, {len(feats)} steps, "
                          f"{el:.0f}s ({el/episodes:.1f}s/ep)", flush=True)
    brain.close()

    feats = np.asarray(feats, dtype=np.float32)
    lat = np.asarray(lat_labels, dtype=np.float32)
    vert = np.asarray(vert_labels, dtype=np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / out_name, features=feats, u_lat=lat, u_vert=vert,
             graph_index=gi, n_windows=N_WINDOWS,
             seeds=np.array([m[0] for m in meta_rows]),
             groups=np.array([m[1] for m in meta_rows]),
             heights=np.array([m[2] for m in meta_rows]),
             steps=np.array([m[3] for m in meta_rows]))
    summary = dict(episodes=episodes, steps=int(feats.shape[0]),
                   feature_dim=int(feats.shape[1]), n_neurons=int(n_neurons),
                   n_windows=N_WINDOWS, backend=backend,
                   lat_frac_left=float((lat < -0.05).mean()),
                   lat_frac_right=float((lat > 0.05).mean()),
                   vert_frac_active=float((vert > 0.05).mean()),
                   wall_seconds=round(time.perf_counter() - t0, 1))
    (OUT / out_name.replace(".npz", "_summary.json")).write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=8,
                   help="episodes per (group x height) cell; 3 groups x 2 heights")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--backend", default="metal")
    p.add_argument("--out", default="arcade_dataset.npz")
    a = p.parse_args()
    generate(per_cell=a.per_cell, base_seed=a.base_seed, backend=a.backend,
             out_name=a.out)


if __name__ == "__main__":
    main()
