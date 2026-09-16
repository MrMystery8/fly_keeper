"""Decoupled simulation session: scientific loop in a worker thread.

Section 18/19: the full MaleCNS closed loop runs SLOWER than real time on CPU
(~0.1 s per 20 ms neural decision). We must NOT alter the scientific timing to
make rendering smooth. Instead we run the exact ``GoalkeeperEngine`` in a
background worker thread and let the UI thread read the latest thread-safe
snapshot at its own display refresh rate. Rendering/UI never touches simulation
timing; the neural decision cadence, MuJoCo timestep, retina timing, and bridge
windows are all unchanged.

The worker also renders the scene camera and the fly-eye camera FROM the sim
thread (MuJoCo renderers are not thread-safe to share), publishing the RGB
frames into the snapshot. The UI just blits whatever is current.

Public surface:
    SimSession(mode, vision, artifact)     construct (does NOT start a shot)
    session.start_shot(aim, power)         queue a human penalty (worker runs it)
    session.snapshot()                     thread-safe copy of current state
    session.stop()                         shut the worker down and free the brain
"""
from __future__ import annotations

import copy
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from doom.game import retinal_samples

from experiments.interactive_demo.modes import (GoalkeeperEngine, StepDiag,
                                                MAX_DECISIONS)
from experiments.interactive_demo.shots import shot_from_human, aim_power_from_shot

SCENE_CAMERA = "back"          # third-person player view (Section 15)
SCENE_W, SCENE_H = 640, 480
EYE_W, EYE_H = 160, 96         # matches the retinal render resolution


@dataclass
class Snapshot:
    """Immutable-ish copy of the sim state for the UI (produced by the worker)."""
    mode: str
    vision: str
    phase: str = "idle"        # idle | running | done
    decision: int = 0
    max_decisions: int = MAX_DECISIONS
    result: Optional[str] = None
    bridge_on: bool = False
    is_neural: bool = False
    is_oracle: bool = False
    # live neural / behavioural read-outs (debug panel)
    bridge_u: float = 0.0
    left_dn_spikes: float = 0.0
    right_dn_spikes: float = 0.0
    dn_asymmetry: float = 0.0
    inj_total_mv: float = 0.0
    retinal_mean: float = 0.0
    brain_spikes: int = 0
    move: str = "STAY"
    lateral: float = 0.0
    fly_y: float = 0.0
    ball_x: float = 0.0
    ball_y: float = 0.0
    # last human shot (for the result card)
    shot_aim: float = 0.0
    shot_power: float = 0.0
    shot_group: str = ""
    # MuJoCo pose snapshot (qpos copy) for MAIN-THREAD rendering. On macOS the
    # GL context is bound to the main thread, so the worker must NOT touch a
    # Renderer; it only publishes state and the UI thread renders from it.
    qpos: Optional[np.ndarray] = None
    # rendered frames (RGB uint8), filled in on the MAIN thread by SceneRenderer.
    scene_rgb: Optional[np.ndarray] = None
    eye_rgb: Optional[np.ndarray] = None      # what the fly's eye samples (post-condition)
    retina_lum: Optional[np.ndarray] = None   # R1-R6 luminance actually driven
    # short history for plots (bridge u and DN drives)
    u_history: list = field(default_factory=list)
    l_history: list = field(default_factory=list)
    r_history: list = field(default_factory=list)
    # wall-clock pace info (purely informational)
    sim_realtime_ratio: float = 0.0


class SimSession:
    """Runs GoalkeeperEngine shots + ALL MuJoCo rendering in ONE worker thread.

    macOS pins a MuJoCo GL context to the thread that created it, and creating a
    second GL context on a different thread deadlocks the native kernel. The
    scientific chain ITSELF renders (VisionBridge feeds the retina), so that
    rendering must live in the worker; therefore the display frames are rendered
    in the SAME worker thread. The main thread only runs pygame/SDL, which is a
    separate windowing path and does not create a MuJoCo GL context.
    """

    def __init__(self, mode: str, *, vision: str = "normal", artifact=None,
                 seed: int = 1):
        self.mode = mode
        self.vision = vision
        self._artifact = artifact
        self._seed = seed

        # The engine (and, for neural modes, its VisionBridge GL context) is
        # built INSIDE the worker thread so EVERY MuJoCo GL context this session
        # uses lives on one thread. Constructing it on the main thread would put
        # the scientific VisionBridge renderer on a different thread from the
        # worker's step + display renderers, which deadlocks on macOS.
        self.engine = None
        self._uv = None
        self._scene_r = None
        self._eye_r = None
        self._pending_bridge_enabled = None    # deferred toggle before engine ready

        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._snap = Snapshot(mode=mode, vision=vision)
        self._cmd_queue = []          # pending (aim, power) shots
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._init_error = None
        self._worker = threading.Thread(target=self._run, name="flykeeper-sim",
                                        daemon=True)
        self._worker.start()
        # Wait for the worker to build the engine so callers see accurate flags.
        # If the mode is unavailable (e.g. plasticity) the worker sets an error.
        if not self._ready.wait(timeout=60):
            raise RuntimeError("sim worker failed to initialize within 60 s")
        if self._init_error is not None:
            raise self._init_error

    # --------------------------------------------------------------- controls
    def start_shot(self, aim: float, power: float):
        """Queue a human penalty; the worker picks it up and runs it."""
        with self._lock:
            self._cmd_queue.append(("shot", float(aim), float(power)))
        self._wake.set()

    def start_shot_spec(self, shot):
        """Queue a pre-built ShotSpec (used by the science-demo runner)."""
        aim, power = aim_power_from_shot(shot)
        with self._lock:
            self._cmd_queue.append(("shot_spec", shot, (aim, power)))
        self._wake.set()

    def set_bridge_enabled(self, enabled: bool):
        """Toggle bridge ON/OFF between shots (learned mode)."""
        if self.engine is None:
            self._pending_bridge_enabled = bool(enabled)
            return
        self.engine.set_bridge_enabled(enabled)
        with self._lock:
            self._snap.bridge_on = self.engine.bridge_enabled

    def snapshot(self) -> Snapshot:
        with self._lock:
            return copy.copy(self._snap)

    def is_busy(self) -> bool:
        with self._lock:
            if self._snap.phase == "error":
                return False
            return self._snap.phase == "running" or bool(self._cmd_queue)

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=10.0)
        # engine is owned by the worker; close from here only if the worker is
        # gone (its brain has its own resources; close is idempotent-ish).
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- worker
    def _make_demo_camera(self):
        """A presentation-only free camera framing the whole penalty.

        Positioned above and behind the goal line looking down the pitch toward
        the incoming ball, so the player sees ball + goal + fly goalkeeper + the
        ball trajectory in one view (Section 15). This is PURELY a rendering
        camera; it does not touch the fly-eye cameras or any scientific state.
        """
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # look at a point just in front of the goal, on the goal line
        cam.lookat[:] = (0.9, 0.0, 0.35)
        cam.distance = 5.2
        cam.azimuth = 180.0     # look toward +x (down the pitch at the ball)
        cam.elevation = -18.0   # slightly above
        return cam

    def _ensure_renderers(self):
        if self._scene_r is None:
            self._scene_r = mujoco.Renderer(self.engine.world.model,
                                            height=SCENE_H, width=SCENE_W)
            self._eye_r = mujoco.Renderer(self.engine.world.model,
                                          height=EYE_H, width=EYE_W)
            self._demo_cam = self._make_demo_camera()

    def _render_into(self, snap: Snapshot):
        """Render scene + fly-eye (with the current vision condition applied).

        Runs in the worker thread. The fly-eye frame has the SAME condition
        transform the scientific path applies (blind/mirrored/shuffled), so the
        FLY VISION panel shows the true retinal input. This is a visualization
        render; the scientific luminance is driven inside the engine's own
        vision.perceive during decision_step.
        """
        world = self.engine.world
        self._scene_r.update_scene(world.data, camera=self._demo_cam)
        snap.scene_rgb = self._scene_r.render().copy()
        self._eye_r.update_scene(world.data, camera="eye_left")
        raw_eye = self._eye_r.render()
        eye_used = raw_eye
        if self.engine.vision is not None:
            eye_used = self.engine.vision._apply_condition(raw_eye)
        snap.eye_rgb = np.ascontiguousarray(eye_used).copy()
        try:
            snap.retina_lum = np.asarray(retinal_samples(eye_used, self._uv),
                                         dtype=np.float32).copy()
        except Exception:
            snap.retina_lum = None

    def _build_engine(self):
        """Construct the engine (and its GL contexts) ON the worker thread."""
        self.engine = GoalkeeperEngine(self.mode, vision=self.vision,
                                       artifact=self._artifact, seed=self._seed)
        if self.engine.brain is not None:
            self._uv = np.asarray(self.engine.brain._brain.uv)
        else:
            g = np.load(ROOT / "upstream/doomfly/outputs/doom/malecns_v1/graph.npz")
            self._uv = np.asarray(g["uv"])
        if self._pending_bridge_enabled is not None:
            self.engine.set_bridge_enabled(self._pending_bridge_enabled)
        with self._lock:
            self._snap.is_neural = self.engine.is_neural
            self._snap.is_oracle = self.engine.is_oracle
            self._snap.bridge_on = self.engine.bridge_enabled

    def _run(self):
        try:
            self._build_engine()
        except Exception as exc:
            self._init_error = exc
            with self._lock:
                self._snap.phase = "error"
                self._snap.result = f"INIT ERROR: {exc}"
            self._ready.set()
            return
        self._ready.set()
        try:
            self._run_inner()
        except Exception as exc:  # surface worker crashes instead of hanging
            import traceback
            traceback.print_exc()
            with self._lock:
                s = copy.copy(self._snap)
                s.phase = "error"
                s.result = f"SIM ERROR: {exc}"
                self._snap = s

    def _run_inner(self):
        self._ensure_renderers()
        # publish an initial idle frame (fly standing, no shot yet)
        snap = copy.copy(self._snap)
        snap.phase = "idle"
        self._render_into(snap)
        with self._lock:
            self._snap = snap

        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            job = None
            with self._lock:
                if self._cmd_queue:
                    job = self._cmd_queue.pop(0)
            if job is None:
                continue
            if job[0] == "shot":
                _, aim, power = job
                shot = shot_from_human(aim, power)
            else:
                _, shot, (aim, power) = job
            self._run_one_shot(shot, aim, power)

    def _run_one_shot(self, shot, aim, power):
        engine = self.engine
        engine.new_shot(shot)
        # reset per-shot snapshot fields
        with self._lock:
            s = self._snap
            s.phase = "running"
            s.result = None
            s.decision = 0
            s.shot_aim, s.shot_power, s.shot_group = aim, power, shot.group
            s.u_history, s.l_history, s.r_history = [], [], []
            self._snap = copy.copy(s)

        t0 = time.perf_counter()
        while not engine.episode_done() and not self._stop.is_set():
            diag = engine.decision_step()      # EXACT scientific 20 ms decision
            # render current physical state from THIS (sim) thread
            snap = copy.copy(self._snap)
            snap.phase = "running"
            snap.decision = diag.decision + 1
            snap.result = diag.result
            snap.bridge_on = diag.bridge_on
            snap.bridge_u = diag.bridge_u
            snap.left_dn_spikes = diag.left_dn_spikes
            snap.right_dn_spikes = diag.right_dn_spikes
            snap.dn_asymmetry = diag.dn_asymmetry
            snap.inj_total_mv = diag.inj_total_mv
            snap.retinal_mean = diag.retinal_mean
            snap.brain_spikes = diag.brain_spikes
            snap.move = diag.move
            snap.lateral = diag.lateral
            snap.fly_y = diag.fly_y
            snap.ball_x = diag.ball_x
            snap.ball_y = diag.ball_y
            snap.u_history = (snap.u_history + [diag.bridge_u])[-MAX_DECISIONS:]
            snap.l_history = (snap.l_history + [diag.left_dn_spikes])[-MAX_DECISIONS:]
            snap.r_history = (snap.r_history + [diag.right_dn_spikes])[-MAX_DECISIONS:]
            sim_s = snap.decision * 0.02
            wall = time.perf_counter() - t0
            snap.sim_realtime_ratio = round(sim_s / wall, 3) if wall > 0 else 0.0
            self._render_into(snap)           # render in the worker (macOS GL)
            with self._lock:
                self._snap = snap

        result = engine.finalize_result()
        s = copy.copy(self._snap)
        s.phase = "done"
        s.result = result
        self._render_into(s)
        with self._lock:
            self._snap = s
