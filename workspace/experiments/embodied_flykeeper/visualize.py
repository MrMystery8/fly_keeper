"""Visual demo of the embodied MaleCNS fly goalkeeper.

Renders an mp4 with a MAIN third-person view of the articulated fly in the goal
plus debug panels:
  - the fly's own eye-camera view (what reaches the retina),
  - the R1-R6 retinal luminance drive,
  - the left/right descending-neuron readout and the decoded command,
  - the current save/goal status.

Usage:
    python -m workspace.experiments.embodied_flykeeper.visualize \
        --controller malecns --episodes 5 --out workspace/outputs/embodied_flykeeper/demo.mp4

MuJoCo's own interactive viewer is also fine for a quick look; this script is
the reproducible, headless-friendly recording path.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.embodied_flykeeper.controllers import build_controller

MAIN_W, MAIN_H = 640, 480
EYE_W, EYE_H = 200, 150
DECISION_S = 0.02
MAX_DECISIONS = 75


def _text(img, row, text, color=(255, 255, 255)):
    """Tiny 5x7 bitmap-free text: draw as colored blocks is overkill; instead we
    just tint a status bar. Kept dependency-free by rendering simple bars."""
    return img  # panels below carry the quantitative info as bars/plots


def _bar_panel(width, height, left, right, command, result):
    """A simple numpy panel showing L/R readout bars and command arrow."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 24)
    total = max(1.0, left + right)
    lh = int((height - 40) * min(1.0, left / max(1.0, total)))
    rh = int((height - 40) * min(1.0, right / max(1.0, total)))
    # left readout bar (cyan) on the left, right readout bar (magenta) on right.
    panel[height - 20 - lh:height - 20, 20:width // 2 - 10] = (60, 200, 220)
    panel[height - 20 - rh:height - 20, width // 2 + 10:width - 20] = (220, 80, 200)
    # command indicator strip at top: green=LEFT-ish, red=RIGHT-ish.
    move = (command or {}).get("move", "STAY")
    color = {"LEFT": (80, 220, 80), "RIGHT": (220, 80, 80), "STAY": (150, 150, 150)}[move]
    panel[6:26, :] = color
    # result tint at bottom.
    if result == "SAVE":
        panel[height - 8:, :] = (60, 220, 60)
    elif result == "GOAL":
        panel[height - 8:, :] = (220, 60, 60)
    return panel


def _retina_panel(width, height, luminance, uv):
    """Scatter the R1-R6 luminance onto a small image at receptor UV positions."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (10, 10, 14)
    xs = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    ys = np.clip((uv[:, 1] * (height - 1)).astype(int), 0, height - 1)
    v = np.clip(luminance * 255, 0, 255).astype(np.uint8)
    panel[ys, xs] = np.stack([v, v, v], axis=1)
    return panel


def _compose(main, eye, retina, bars):
    """Stack the eye/retina/bars column to the right of the main view."""
    h = main.shape[0]
    col_w = max(eye.shape[1], retina.shape[1], bars.shape[1])
    col = np.zeros((h, col_w, 3), dtype=np.uint8)
    col[:] = (12, 12, 16)
    y = 0
    for panel in (eye, retina, bars):
        ph, pw = panel.shape[:2]
        col[y:y + ph, 0:pw] = panel
        y += ph + 4
    return np.concatenate([main, col], axis=1)


def run(controller_name, episodes, out_path, seed=7, condition="normal"):
    world = GoalkeeperWorld(seed=seed)
    brain = vision = decoder = None
    if controller_name == "malecns":
        from adapters.brain import MaleCNSBrain
        brain = MaleCNSBrain(backend="cpu")
        vision = VisionBridge(world.fly, brain, camera="eye_left", condition=condition)
        decoder = DescendingMotorDecoder(left_ids=[10162], right_ids=[10059],
                                         turn_gain=2.5, smoothing=0.5)
    controller = build_controller(controller_name, brain=brain,
                                  vision_bridge=vision, decoder=decoder,
                                  goal_line_x=GOAL_LINE_X, seed=seed)

    main_r = mujoco.Renderer(world.model, height=MAIN_H, width=MAIN_W)
    eye_r = mujoco.Renderer(world.model, height=EYE_H, width=EYE_W)
    # A separate eye renderer for the panel even for non-malecns controllers.
    uv = brain._brain.uv if brain else np.load(
        ROOT / "upstream/doomfly/outputs/doom/malecns_v1/graph.npz")["uv"]

    import imageio
    writer = imageio.get_writer(str(out_path), fps=30, macro_block_size=None)
    for ep in range(episodes):
        shot = world.sample_shot(str(np.random.default_rng(seed + ep).choice(
            ["left", "center", "right"])))
        world.reset(shot)
        controller.reset()
        if vision is not None and condition == "static_ball":
            vision.set_static_reference(vision.render())
        n = 0
        last_lum = np.zeros(len(uv), dtype=np.float32)
        while world.result is None and n < MAX_DECISIONS:
            command, diag = controller.act(world)
            if "retinal_mean" in diag and vision is not None:
                # capture the eye view the controller used for the panel
                pass
            world.fly.set_command(command["forward"], command.get("turn", 0.0),
                                  command["gait_on"], lateral=command.get("lateral", 0.0))
            world.step(DECISION_S)
            # panels
            main_r.update_scene(world.data, camera="back")
            main = main_r.render()
            eye_r.update_scene(world.data, camera="eye_left")
            eye = eye_r.render()
            from doom.game import retinal_samples
            last_lum = retinal_samples(eye, uv)
            retina = _retina_panel(EYE_W, EYE_H, last_lum, uv)
            bars = _bar_panel(EYE_W, MAIN_H - 2 * EYE_H - 8,
                              float(diag.get("left_spikes", 0) or 0),
                              float(diag.get("right_spikes", 0) or 0),
                              command, world.result)
            writer.append_data(_compose(main, eye, retina, bars))
            n += 1
        # hold the final frame briefly to show the outcome
        for _ in range(15):
            main_r.update_scene(world.data, camera="back")
            main = main_r.render()
            eye_r.update_scene(world.data, camera="eye_left")
            eye = eye_r.render()
            retina = _retina_panel(EYE_W, EYE_H, last_lum, uv)
            bars = _bar_panel(EYE_W, MAIN_H - 2 * EYE_H - 8, 0, 0,
                              {"move": "STAY"}, world.result)
            writer.append_data(_compose(main, eye, retina, bars))
        print(f"episode {ep}: {shot.group:6s} -> {world.result}")
    writer.close()
    print(f"saved demo to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["malecns", "random", "heuristic"], default="malecns")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--condition", default="normal")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "workspace/outputs/embodied_flykeeper/demo.mp4"))
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run(args.controller, args.episodes, args.out, args.seed, args.condition)


if __name__ == "__main__":
    main()
