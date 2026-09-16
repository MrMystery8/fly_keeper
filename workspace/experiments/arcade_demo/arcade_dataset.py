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

ARCADE BRIDGE V2 (stale-dataset fix)
------------------------------------
The v1 dataset was contaminated: ~1/3 of episodes padded to MAX_DECISIONS=90
because lofted shots deflected off the crossbar and never reached a terminal
state, so ~40+ post-resolution "dead-ball" frames per stale episode were paired
with saturated teacher labels. This generator now:

  * collects ONLY while `world.shot_live` (the shot has not resolved), so
    collection stops exactly at the terminal step -- never padded;
  * records a per-episode MANIFEST (seed, class, height, length, terminal
    result, terminal step);
  * ASSERTS no feature window is collected after terminal resolution and that
    each episode's last collected sample index <= terminal step;
  * reports the natural (variable) episode-length distribution.

Only the DATA is corrected. The visual architecture is unchanged: single
`eye_left` camera, the same 207 optic-lobe neurons in the same order, and the
same 4 x 20 ms causal windows as the frozen/v1 bridge, so v2 isolates the
stale-dataset fix.
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
             out_name="arcade_dataset_v2.npz"):
    gi, body_ids = selected_ids(n_neurons)
    body_list = [int(x) for x in body_ids]
    brain = MaleCNSBrain(backend=backend)

    feats, lat_labels, vert_labels = [], [], []
    meta_rows = []                 # (seed, group, hf, step_in_episode)
    manifest = []                  # per-episode integrity record
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
                ep_start = len(feats)
                last_step_collected = -1
                # collect ONLY while the shot is live (not yet resolved). A hard
                # MAX_DECISIONS guard remains as a safety net; with the corrected
                # world every shot resolves well before it.
                while world.shot_live and n < MAX_DECISIONS:
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
                    last_step_collected = n
                    # bridge OFF: body barely moves; just advance physics. The
                    # step may resolve the shot, ending the loop -> no frame is
                    # collected after resolution.
                    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
                    world.step(DECISION_S)
                    n += 1
                diag = world.outcome_diagnostics()
                term_step = diag.get("terminal_step")
                ep_len = len(feats) - ep_start
                manifest.append(dict(
                    episode_id=episodes, shot_seed=int(seed), shot_class=group,
                    height_class=("low" if hf < 0.34 else "mid"
                                  if hf < 0.67 else "high"),
                    height_frac=float(hf), episode_length=int(ep_len),
                    terminal_result=world.result,
                    terminal_step=(int(term_step) if term_step is not None
                                   else None),
                    last_step_collected=int(last_step_collected),
                    keeper_contact=bool(diag.get("keeper_contact", False)),
                    reached_max=bool(n >= MAX_DECISIONS),
                ))
                episodes += 1
                seed += 1
                if episodes % 5 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {episodes} episodes, {len(feats)} steps, "
                          f"{el:.0f}s ({el/episodes:.1f}s/ep)", flush=True)
    brain.close()

    # ------------------------------------------------------------- integrity
    _assert_integrity(manifest)

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

    lens = np.array([m["episode_length"] for m in manifest])
    n_reached_max = int(sum(m["reached_max"] for m in manifest))
    n_no_result = int(sum(m["terminal_result"] is None for m in manifest))
    results = dict(Counter(m["terminal_result"] for m in manifest))
    summary = dict(
        episodes=episodes, steps=int(feats.shape[0]),
        feature_dim=int(feats.shape[1]), n_neurons=int(n_neurons),
        n_windows=N_WINDOWS, backend=backend, camera="eye_left",
        lat_frac_left=float((lat < -0.05).mean()),
        lat_frac_right=float((lat > 0.05).mean()),
        vert_frac_active=float((vert > 0.05).mean()),
        episode_length=dict(min=int(lens.min()), median=float(np.median(lens)),
                            mean=round(float(lens.mean()), 2),
                            max=int(lens.max())),
        n_episodes_reached_max=n_reached_max,
        n_episodes_no_result=n_no_result,
        terminal_results=results,
        integrity_ok=True,
        wall_seconds=round(time.perf_counter() - t0, 1))
    (OUT / out_name.replace(".npz", "_summary.json")).write_text(
        json.dumps(summary, indent=2))
    (OUT / out_name.replace(".npz", "_manifest.json")).write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _assert_integrity(manifest):
    """Fail loudly if any episode padded past resolution or left frames after
    the terminal step. These are the exact defects that contaminated v1."""
    problems = []
    for m in manifest:
        term = m["terminal_step"]
        last = m["last_step_collected"]
        # 1. every episode must have resolved (no null padded-out episodes)
        if m["terminal_result"] is None:
            problems.append(f"ep {m['episode_id']} seed {m['shot_seed']} "
                            f"never resolved (padded to {m['episode_length']})")
            continue
        # 2. no feature window may be collected AFTER the terminal step. Because
        #    we collect the frame and THEN step (which may resolve the shot),
        #    the last collected step equals terminal_step - 1 in the normal case;
        #    it must never exceed the terminal step.
        if term is not None and last > term:
            problems.append(f"ep {m['episode_id']} seed {m['shot_seed']} "
                            f"collected step {last} > terminal {term}")
        # 3. no episode should have hit the safety cap (that means it padded)
        if m["reached_max"]:
            problems.append(f"ep {m['episode_id']} seed {m['shot_seed']} "
                            f"hit MAX_DECISIONS safety cap")
    if problems:
        raise AssertionError("dataset integrity FAILED:\n  " +
                             "\n  ".join(problems))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=8,
                   help="episodes per (group x height) cell; 3 groups x 2 heights")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--backend", default="metal")
    p.add_argument("--out", default="arcade_dataset_v2.npz")
    a = p.parse_args()
    generate(per_cell=a.per_cell, base_seed=a.base_seed, backend=a.backend,
             out_name=a.out)


if __name__ == "__main__":
    main()
