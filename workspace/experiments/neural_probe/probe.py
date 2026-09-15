"""Present controlled visual stimuli to MaleCNS with a FIXED body pose and
record sparse per-neuron spike responses.

Design
------
- The fly is placed at the goal-line origin facing +x and never moves (open
  loop). The ball is placed at a controlled *static* position for each stimulus
  so the retinal image is well-defined and repeatable.
- Each trial: reset the brain, present the stimulus for a warm-up + a sequence
  of short recording windows, and store per-window spike counts for every
  neuron that fired (sparse), preserving body IDs.
- Conditions include left / center / right / strong-left / strong-right /
  mirrored-left / mirrored-right / static / blind / shuffled, plus a baseline
  (dark) for baseline subtraction.

Storage: for each (condition, trial) we keep, per recording window, arrays of
(neuron_index, spike_count). Windows tile the presentation so we can later
analyse any temporal window (5/10/20/50/100 ms) by summing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from doom.game import retinal_samples
from embodiment.mujoco_world import GoalkeeperWorld, GOAL_HALF_WIDTH, BALL_RADIUS

# Stimulus conditions. Each maps to a static ball lateral offset (y, cm) and a
# visual transform. Positive y is the fly's LEFT (it faces +x).
STIMULI = {
    "left":         dict(y=+0.6, transform="none"),
    "center":       dict(y=0.0,  transform="none"),
    "right":        dict(y=-0.6, transform="none"),
    "strong_left":  dict(y=+1.1, transform="none"),
    "strong_right": dict(y=-1.1, transform="none"),
    "mirrored_left":  dict(y=+0.6, transform="mirror"),   # left ball, image flipped
    "mirrored_right": dict(y=-0.6, transform="mirror"),   # right ball, image flipped
    "static":       dict(y=0.0,  transform="none"),       # same as center; ball frozen
    "blind":        dict(y=0.0,  transform="blind"),
    "shuffled":     dict(y=0.0,  transform="shuffle"),
}

BALL_X = 2.0  # fixed static ball distance (cm) in front of the fly


class NeuralProbe:
    def __init__(self, seed=7, camera="eye_left"):
        self.world = GoalkeeperWorld(seed=seed)
        self.model = self.world.model
        self.data = self.world.data
        self.camera = camera
        self.renderer = mujoco.Renderer(self.model, height=96, width=160)
        # Build the brain lazily (heavy) but keep one instance for all trials.
        from adapters.brain import MaleCNSBrain
        self.brain = MaleCNSBrain(backend="cpu")
        self._b = self.brain._brain
        self.retina_ids = [int(self._b.ids[i]) for i in self._b.retina]
        self.uv = self._b.uv
        self.ids = np.asarray(self._b.ids)          # body IDs by graph index
        self.n = self._b.n
        self._shuffle_perm = np.random.default_rng(1234).permutation(len(self.uv))

    # ------------------------------------------------------------- stimulus
    def _place_ball(self, y):
        """Freeze the fly at origin facing +x and put a static ball at (BALL_X, y)."""
        self.world.fly.set_pose(xy=(0.0, 0.0), yaw=0.0)
        q = self.data.qpos
        qadr = self.world._ball_qadr
        q[qadr + 0] = BALL_X
        q[qadr + 1] = float(np.clip(y, -GOAL_HALF_WIDTH + 0.2, GOAL_HALF_WIDTH - 0.2))
        q[qadr + 2] = BALL_RADIUS
        q[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.world._ball_dofadr:self.world._ball_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _render_luminance(self, transform):
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render()
        if transform == "blind":
            rgb = np.zeros_like(rgb)
        elif transform == "mirror":
            rgb = rgb[:, ::-1].copy()
        lum = retinal_samples(rgb, self.uv)
        if transform == "shuffle":
            lum = lum[self._shuffle_perm]
        return lum

    # -------------------------------------------------------------- record
    def run_trial(self, condition, window_ms=5.0, n_windows=24, warmup_ms=20.0):
        """Present one stimulus and record sparse per-window spike counts.

        Returns dict with 'windows': list of (idx_array, count_array), and the
        retinal luminance used. The body does not move; the ball is static, so
        the same luminance is re-presented every window (steady stimulus).
        """
        spec = STIMULI[condition]
        self._place_ball(spec["y"])
        lum = self._render_luminance(spec["transform"])
        self.brain.reset()
        # reset() rebuilds the NativeBrain; re-bind the cached handle so we read
        # the CURRENT runtime's spike-count buffer, not the stale one.
        self._b = self.brain._brain
        # Warm-up: let transients settle before recording.
        n_warm = int(round(warmup_ms / window_ms))
        for _ in range(n_warm):
            self.brain.stimulate_retinal_luminance(self.retina_ids, lum)
            self.brain.step(window_ms)
        windows = []
        for _ in range(n_windows):
            self.brain.stimulate_retinal_luminance(self.retina_ids, lum)
            self.brain.step(window_ms)
            counts = self._b.counts
            idx = np.flatnonzero(counts)
            windows.append((idx.astype(np.int32), counts[idx].astype(np.int32)))
        return {"condition": condition, "luminance": lum.astype(np.float32),
                "windows": windows, "window_ms": window_ms}

    # -------------------------------------------------- dynamic (moving) shots
    def _dynamic_luminance_sequence(self, group, speed, transform, n_frames,
                                    frame_dt_s):
        """Replay the retinal sequence the embodied fly would see for a moving
        shot: the ball approaches from x=SHOT_X toward the goal while the fly is
        held fixed at the origin facing +x. Returns a list of luminance vectors,
        one per control frame (the real closed-loop visual input, open loop)."""
        from embodiment.mujoco_world import SHOT_X, GOAL_LINE_X, BALL_RADIUS as BR
        self.world.fly.set_pose(xy=(0.0, 0.0), yaw=0.0)
        qadr = self.world._ball_qadr
        dofadr = self.world._ball_dofadr
        aim = {"left": 0.6, "center": 0.0, "right": -0.6,
               "strong_left": 1.1, "strong_right": -1.1}.get(group, 0.0)
        q = self.data.qpos
        q[qadr + 0] = SHOT_X
        q[qadr + 1] = aim
        q[qadr + 2] = BR
        q[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        dx = GOAL_LINE_X - SHOT_X
        travel_t = abs(dx) / speed
        vy = (aim - q[qadr + 1]) / max(travel_t, 1e-3)
        self.data.qvel[dofadr + 0] = -speed
        self.data.qvel[dofadr + 1] = vy
        self.data.qvel[dofadr + 2] = 0.0
        mujoco.mj_forward(self.model, self.data)
        seq = []
        substeps = max(1, int(round(frame_dt_s / self.model.opt.timestep)))
        # Hold the fly still by clamping its stance control and freezing its
        # root each substep WITHOUT resetting the whole data (which would also
        # reset the ball). We snapshot the fly's root qpos and restore only that.
        self.world.fly.set_command(0, 0, 0)
        rq = self.world.fly.root_qadr
        rd = self.world.fly.root_dofadr
        fly_root = self.data.qpos[rq:rq + 7].copy()
        for _ in range(n_frames):
            for _s in range(substeps):
                self.data.ctrl[:] = self.world.fly.stance_ctrl
                mujoco.mj_step(self.model, self.data)
                # re-pin only the fly's floating base; leave the ball untouched
                self.data.qpos[rq:rq + 7] = fly_root
                self.data.qvel[rd:rd + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            seq.append(self._render_luminance(transform))
        return seq

    def run_dynamic_trial(self, group, speed=5.0, transform="none",
                          window_ms=5.0, frames=24, warmup_ms=15.0):
        """Present a moving-shot retinal sequence and record per-window responses.

        Each control frame's luminance is presented for one window_ms brain step,
        so window index == frame index == a distinct point on the ball's
        trajectory. This preserves the temporal structure of a real shot."""
        frame_dt_s = window_ms / 1000.0
        seq = self._dynamic_luminance_sequence(group, speed, transform, frames,
                                               frame_dt_s)
        self.brain.reset()
        self._b = self.brain._brain
        n_warm = int(round(warmup_ms / window_ms))
        # warm up on the very first (far-away) frame
        for _ in range(n_warm):
            self.brain.stimulate_retinal_luminance(self.retina_ids, seq[0])
            self.brain.step(window_ms)
        windows = []
        for lum in seq:
            self.brain.stimulate_retinal_luminance(self.retina_ids, lum)
            self.brain.step(window_ms)
            counts = self._b.counts
            idx = np.flatnonzero(counts)
            windows.append((idx.astype(np.int32), counts[idx].astype(np.int32)))
        return {"group": group, "speed": speed, "transform": transform,
                "windows": windows, "window_ms": window_ms,
                "luminance_mean": float(np.mean([s.mean() for s in seq]))}

    def collect_dynamic(self, trials=6, window_ms=5.0, frames=24, warmup_ms=15.0):
        """Dynamic-shot conditions with speed and mirror variations."""
        specs = [
            ("left", 5.0, "none"), ("center", 5.0, "none"), ("right", 5.0, "none"),
            ("left", 8.0, "none"), ("right", 8.0, "none"),        # fast
            ("left", 3.0, "none"), ("right", 3.0, "none"),        # slow
            ("left", 5.0, "mirror"), ("right", 5.0, "mirror"),    # mirrored
            ("center", 5.0, "blind"),                              # blind control
        ]
        rng = np.random.default_rng(7)
        out = {}
        for group, speed, transform in specs:
            key = f"{group}_{int(speed)}_{transform}"
            results = []
            for t in range(trials):
                sp = speed * float(rng.uniform(0.95, 1.05))
                results.append(self.run_dynamic_trial(
                    group, speed=sp, transform=transform,
                    window_ms=window_ms, frames=frames, warmup_ms=warmup_ms))
            out[key] = results
        return out

    def collect(self, conditions=None, trials=4, jitter=0.12, **kw):
        """Run repeated trials per condition. Jitter adds small ball-position
        noise (except for transform-defined conditions) for reliability stats.

        Returns a nested dict: {condition: [trial_result, ...]}.
        """
        conditions = conditions or list(STIMULI.keys())
        rng = np.random.default_rng(99)
        out = {}
        for cond in conditions:
            base_y = STIMULI[cond]["y"]
            results = []
            for t in range(trials):
                # jitter the physical ball position slightly for non-blind stimuli
                if STIMULI[cond]["transform"] in ("blind",):
                    dy = 0.0
                else:
                    dy = float(rng.uniform(-jitter, jitter))
                saved = STIMULI[cond]["y"]
                STIMULI[cond]["y"] = base_y + dy
                try:
                    results.append(self.run_trial(cond, **kw))
                finally:
                    STIMULI[cond]["y"] = saved
            out[cond] = results
        return out
