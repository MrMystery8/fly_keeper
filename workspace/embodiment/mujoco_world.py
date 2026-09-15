"""Football goalkeeper arena built around the articulated flybody.

Everything is in the flybody model's native units (centimetres; gravity is
-981 cm/s^2). The scene is a DELIBERATELY NORMALIZED toy scale chosen so the
tiny fly and the ball/goal are mechanically compatible and collisions are
stable. It is NOT physically FIFA-accurate and does not claim to be.

Layout (world axes; the fly's forward walking axis is +x):
  - The goal mouth faces -x. The fly starts on the goal line at x=0, y=0,
    facing -x (yaw = pi) so incoming shots approach head-on.
  - The ball is fired from x = SHOT_X toward the goal (moving -x), with a
    lateral (y) offset and speed chosen per shot group (left/center/right).
  - SAVE  = the ball contacts the fly or the goal frame before crossing the
    goal line, or is deflected back out.
  - GOAL  = the ball crosses the goal line (x <= GOAL_LINE_X) inside the posts.

The world knows the ball's true state for physics and scoring; that state is
NEVER passed to the brain (only rendered pixels are).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco

from .fly_body import FlyBody

# --- normalized arena geometry (cm) ---
# DELIBERATELY NORMALIZED, not FIFA-scale. The ball is made large relative to
# the ~0.3 cm fly so that (a) it subtends a meaningful solid angle on the
# compound eye and thus recruits many R1-R6 receptors, and (b) the fly's ~cm/s
# walking range can actually influence a save. This is an engineering choice.
GOAL_HALF_WIDTH = 1.6     # posts at y = +/- 1.6 cm  (goal mouth 3.2 cm wide)
GOAL_HEIGHT = 1.4         # crossbar height
GOAL_LINE_X = -0.3        # ball center crossing this x (inside posts) = GOAL
SHOT_X = 5.0              # ball spawn distance in front of goal
BALL_RADIUS = 0.5         # large, salient ball (see note above)
BALL_MASS = 0.02          # light so the tiny fly can perturb it
POST_RADIUS = 0.12


def _arena_xml() -> str:
    """Extra MJCF spliced into the flybody scene: goal frame + ball + markers."""
    hw = GOAL_HALF_WIDTH
    h = GOAL_HEIGHT
    return f"""
  <asset>
    <material name="goalpost" rgba="0.35 0.35 0.4 1"/>
    <material name="ballmat" rgba="1 1 1 1" emission="0.6"/>
    <material name="netmat" rgba="0.3 0.32 0.4 0.25"/>
    <material name="pitch" rgba="0.15 0.45 0.18 1"/>
  </asset>
  <worldbody>
    <!-- goal frame: two posts + crossbar, behind the fly (toward +x is field) -->
    <geom name="post_left"  type="capsule" material="goalpost" size="{POST_RADIUS}"
          fromto="0 {hw} -0.13  0 {hw} {h}"/>
    <geom name="post_right" type="capsule" material="goalpost" size="{POST_RADIUS}"
          fromto="0 {-hw} -0.13  0 {-hw} {h}"/>
    <geom name="crossbar"   type="capsule" material="goalpost" size="{POST_RADIUS}"
          fromto="0 {hw} {h}  0 {-hw} {h}"/>
    <!-- back net plane, just behind the goal line -->
    <geom name="net_back" type="box" material="netmat" pos="{GOAL_LINE_X-0.4} 0 {h/2}"
          size="0.02 {hw} {h/2}" contype="0" conaffinity="0"/>
    <!-- the ball: a free body fired toward the goal -->
    <body name="ball" pos="{SHOT_X} 0 {BALL_RADIUS}">
      <freejoint name="ball_free"/>
      <geom name="ball" type="sphere" material="ballmat" size="{BALL_RADIUS}"
            mass="{BALL_MASS}" friction="0.6 0.02 0.001"/>
    </body>
  </worldbody>
"""


@dataclass
class ShotSpec:
    group: str          # 'left' | 'center' | 'right'
    y_target: float     # lateral aim at the goal line
    speed: float        # cm/s
    height: float       # launch/aim height (cm)


@dataclass
class EpisodeResult:
    result: str | None = None      # 'SAVE' | 'GOAL'
    frames: int = 0
    contact: bool = False


class GoalkeeperWorld:
    """Owns the fly + arena, fires shots, detects saves/goals, resets episodes."""

    def __init__(self, seed=7, timestep=None):
        self.fly = FlyBody(extra_xml=_arena_xml(), timestep=timestep)
        m = self.fly.model
        self.model = m
        self.data = self.fly.data
        self.rng = np.random.default_rng(seed)
        self._ball_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_qadr = m.jnt_qposadr[m.body_jntadr[self._ball_bid]]
        self._ball_dofadr = m.jnt_dofadr[m.body_jntadr[self._ball_bid]]
        self._ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        # fly collision geoms (any geom belonging to a body under 'thorax').
        self._fly_geoms = self._collect_fly_geoms()
        self._post_gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                           for n in ("post_left", "post_right", "crossbar")}
        self._result = None
        self._contact = False

    def _collect_fly_geoms(self):
        m = self.model
        fly_geoms = set()
        # A geom is the fly's if its body's ancestry reaches 'thorax'.
        thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b != 0:
                if b == thorax:
                    fly_geoms.add(g)
                    break
                b = m.body_parentid[b]
        return fly_geoms

    # ---------------------------------------------------------------- shots
    def sample_shot(self, group=None) -> ShotSpec:
        if group is None:
            group = str(self.rng.choice(["left", "center", "right"]))
        # Aim point at the goal line, within the posts.
        aim = {"left": 1.05, "center": 0.0, "right": -1.05}[group]
        aim += float(self.rng.uniform(-0.25, 0.25))
        speed = float(self.rng.uniform(10.0, 16.0))
        height = float(self.rng.uniform(BALL_RADIUS, 0.9))
        return ShotSpec(group, aim, speed, height)

    # --------------------------------------------------------------- episode
    def reset(self, shot: ShotSpec, fly_yaw=0.0):
        """Place the fly on the goal line facing the field (+x, toward the
        incoming ball), so its eye cameras see the approaching shot."""
        self.fly.set_pose(xy=(0.0, 0.0), yaw=fly_yaw)
        self.fly.set_command(0, 0, 0)
        self.fly.reset()  # keyframe stance, then reposition
        self.fly.set_pose(xy=(0.0, 0.0), yaw=fly_yaw)
        # Position ball at the shot origin and aim its velocity at the goal line.
        q = self.data.qpos
        q[self._ball_qadr + 0] = SHOT_X
        q[self._ball_qadr + 1] = float(np.clip(shot.y_target, -GOAL_HALF_WIDTH + 0.3, GOAL_HALF_WIDTH - 0.3))
        q[self._ball_qadr + 2] = shot.height
        q[self._ball_qadr + 3:self._ball_qadr + 7] = [1, 0, 0, 0]
        # Velocity toward the goal (-x), aimed so it arrives near y_target.
        dx = GOAL_LINE_X - SHOT_X
        travel_t = abs(dx) / shot.speed
        vy = (shot.y_target - q[self._ball_qadr + 1]) / max(travel_t, 1e-3)
        self.data.qvel[self._ball_dofadr + 0] = -shot.speed
        self.data.qvel[self._ball_dofadr + 1] = vy
        self.data.qvel[self._ball_dofadr + 2] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._result = None
        self._contact = False
        self._shot = shot
        return self._observe_ball()

    def _observe_ball(self):
        q = self.data.qpos
        return {
            "pos": q[self._ball_qadr:self._ball_qadr + 3].copy(),
            "vel": self.data.qvel[self._ball_dofadr:self._ball_dofadr + 3].copy(),
        }

    def _check_contacts(self):
        """Set contact flag if the ball touches the fly or a goalpost."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}
            if self._ball_gid in pair:
                other = (pair - {self._ball_gid}).pop() if len(pair) == 2 else self._ball_gid
                if other in self._fly_geoms or other in self._post_gids:
                    self._contact = True

    def _check_outcome(self):
        """Decide SAVE/GOAL when the ball reaches the goal-line plane."""
        if self._result is not None:
            return
        ball = self._observe_ball()
        x, y, z = ball["pos"]
        # Ball has crossed the goal line plane.
        if x <= GOAL_LINE_X:
            inside = abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT
            if inside and not self._contact:
                self._result = "GOAL"
            else:
                self._result = "SAVE"
        # Ball deflected back out past the shot line without scoring => SAVE.
        elif x > SHOT_X + 0.5 and self._contact:
            self._result = "SAVE"

    def step(self, dt_s):
        """Advance the coupled fly+ball physics by dt_s and update contacts."""
        # fly_body.step advances physics with the CPG; contacts are checked
        # against the shared data after each chunk.
        self.fly.step(dt_s)
        self._check_contacts()
        self._check_outcome()
        return self._observe_ball()

    @property
    def result(self):
        return self._result

    @property
    def contact(self):
        return self._contact
