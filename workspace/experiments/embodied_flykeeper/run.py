"""Headless / visual runner for the embodied MaleCNS fly goalkeeper.

Examples
--------
    # closed-loop neural controller, 50 headless episodes
    python -m workspace.experiments.embodied_flykeeper.run \
        --controller malecns --episodes 50 --headless

    # baselines
    python -m workspace.experiments.embodied_flykeeper.run --controller random    --episodes 50 --headless
    python -m workspace.experiments.embodied_flykeeper.run --controller heuristic --episodes 50 --headless

    # a sensory control (behavior should NOT depend on a blinded eye for MaleCNS)
    python -m workspace.experiments.embodied_flykeeper.run \
        --controller malecns --episodes 50 --headless --condition blind

Timing (see report): physics dt = 0.1 ms (flybody native); control/vision and
the MaleCNS decision window are both 20 ms (50 Hz); episodes run until the ball
resolves. The neural core integrates 200 x 0.1 ms steps per 20 ms decision.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.embodied_flykeeper.controllers import build_controller

GROUPS = ("left", "center", "right")
CONDITIONS = ("normal", "blind", "mirrored", "static_ball")
DECISION_MS = 20.0
DECISION_S = DECISION_MS / 1000.0
MAX_DECISIONS = 75  # hard cap per episode (~1.5 s sim)


def run(controller_name, episodes, seed, *, condition="normal", group="all",
        backend="cpu", decoder_ids=None, telemetry=None, record=None):
    world = GoalkeeperWorld(seed=seed)
    brain = vision = decoder = None
    if controller_name == "malecns":
        from adapters.brain import MaleCNSBrain
        brain = MaleCNSBrain(backend=backend)
        vision = VisionBridge(world.fly, brain, camera="eye_left", condition=condition)
        decoder_kind = (decoder_ids or {}).get("kind", "dnp20")
        if decoder_kind == "leaky":
            from embodiment.motor_decoder import LeakyIntegratorDecoder
            decoder = LeakyIntegratorDecoder()
        else:
            left = (decoder_ids or {}).get("left", [10162])
            right = (decoder_ids or {}).get("right", [10059])
            decoder = DescendingMotorDecoder(left_ids=left, right_ids=right,
                                             turn_gain=2.5, forward_bias=0.25,
                                             smoothing=0.5)
    controller = build_controller(controller_name, brain=brain,
                                  vision_bridge=vision, decoder=decoder,
                                  goal_line_x=GOAL_LINE_X, seed=seed)

    groups = GROUPS if group == "all" else (group,)
    rows, decisions = [], []
    recorder = _Recorder(world, record) if record else None
    start = time.perf_counter()
    brain_ms0 = brain._brain.sim_ms if brain else 0.0

    for ep in range(episodes):
        shot = world.sample_shot(str(np.random.default_rng(seed + ep).choice(groups)))
        world.reset(shot)
        controller.reset()
        # static-ball control needs a reference frame with the ball at origin.
        if vision is not None and condition == "static_ball":
            vision.set_static_reference(vision.render())
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            command, diag = controller.act(world)
            world.fly.set_command(command["forward"], command.get("turn", 0.0),
                                  command["gait_on"], lateral=command.get("lateral", 0.0))
            world.step(DECISION_S)
            ball = world._observe_ball()
            fly_pos = world.fly.position
            decisions.append({
                "episode": ep, "step": n, "group": shot.group,
                "sim_time_s": round(n * DECISION_S, 4),
                "fly_x": round(float(fly_pos[0]), 4), "fly_y": round(float(fly_pos[1]), 4),
                "fly_heading": round(float(world.fly.heading), 4),
                "fly_vy": round(float(world.fly.velocity[1]), 4),
                "ball_x": round(float(ball["pos"][0]), 4), "ball_y": round(float(ball["pos"][1]), 4),
                "ball_vx": round(float(ball["vel"][0]), 3), "ball_vy": round(float(ball["vel"][1]), 3),
                "move": command.get("move"),
                "lateral": round(float(command.get("lateral", 0.0)), 3),
                "forward": round(float(command["forward"]), 3),
                "left_spikes": diag.get("left_spikes"), "right_spikes": diag.get("right_spikes"),
                "asymmetry": diag.get("asymmetry"), "retinal_mean": diag.get("retinal_mean"),
                "condition": condition, "contact": world.contact,
            })
            if recorder:
                recorder.capture()
            if telemetry:
                telemetry.publish({
                    "episode": ep, "game": {"ball_x": float(ball["pos"][0]),
                    "ball_y": float(ball["pos"][1]), "fly_x": float(fly_pos[0]),
                    "fly_y": float(fly_pos[1]), "action": command.get("move"),
                    "result": world.result}, "readout": {
                    "left": diag.get("left_spikes"), "right": diag.get("right_spikes")}})
            n += 1
        result = world.result or "GOAL"  # timed out without a save = conceded
        rows.append({"episode": ep, "group": shot.group, "result": result,
                     "decisions": n, "final_fly_y": round(float(world.fly.position[1]), 4),
                     "contact": world.contact})

    if recorder:
        recorder.save()
    saves = sum(r["result"] == "SAVE" for r in rows)
    summary = {
        "controller": controller_name, "backend": backend, "condition": condition,
        "group": group, "episodes": episodes, "seed": seed,
        "saves": saves, "goals": episodes - saves, "save_rate": round(saves / episodes, 4),
        "by_group": {g: _group_rate(rows, g) for g in GROUPS},
        "wall_seconds": round(time.perf_counter() - start, 2),
        "brain_ms": round((brain._brain.sim_ms - brain_ms0) if brain else 0.0, 1),
    }
    return rows, decisions, summary


def _group_rate(rows, g):
    sub = [r for r in rows if r["group"] == g]
    if not sub:
        return None
    return {"n": len(sub), "saves": sum(r["result"] == "SAVE" for r in sub),
            "save_rate": round(sum(r["result"] == "SAVE" for r in sub) / len(sub), 4)}


class _Recorder:
    """Optional offscreen video recorder for the visual demo."""

    def __init__(self, world, path, camera="back", fps=30):
        import mujoco
        self.world = world
        self.path = Path(path)
        self.camera = camera
        self.fps = fps
        self.renderer = mujoco.Renderer(world.model, height=480, width=640)
        self.frames = []

    def capture(self):
        self.renderer.update_scene(self.world.data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())

    def save(self):
        try:
            import imageio
            self.path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self.path, self.frames, fps=self.fps)
        except Exception as exc:  # pragma: no cover - optional dependency
            np.save(self.path.with_suffix(".npy"), np.asarray(self.frames))
            print(f"[recorder] imageio unavailable ({exc}); saved raw frames to "
                  f"{self.path.with_suffix('.npy')}")


def main():
    p = argparse.ArgumentParser(description="Embodied MaleCNS fly goalkeeper")
    p.add_argument("--controller", choices=["malecns", "random", "heuristic", "passive"], default="malecns")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--condition", choices=CONDITIONS, default="normal")
    p.add_argument("--group", choices=["all", *GROUPS], default="all")
    p.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    p.add_argument("--decoder", choices=["dnp20", "leaky"], default="dnp20",
                   help="dnp20 = original 20ms total-count; leaky = timing-aware integrator")
    p.add_argument("--record", type=str, default=None, help="path to save an mp4 demo")
    p.add_argument("--visualizer", action="store_true")
    args = p.parse_args()

    telemetry = None
    if args.visualizer:
        from telemetry import TelemetryServer
        telemetry = TelemetryServer()
        telemetry.start()

    rows, decisions, summary = run(
        args.controller, args.episodes, args.seed, condition=args.condition,
        group=args.group, backend=args.backend, telemetry=telemetry,
        record=args.record, decoder_ids={"kind": args.decoder})

    out = ROOT / "workspace" / "outputs" / "embodied_flykeeper" / uuid.uuid4().hex
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, data in [("episodes.csv", rows), ("decisions.csv", decisions)]:
        if data:
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print(json.dumps({"output": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()
