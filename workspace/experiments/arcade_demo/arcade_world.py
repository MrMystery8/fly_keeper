"""Arcade goalkeeper world: ArcadeFlyBody + high shots + corrected scoring.

Subclasses the frozen `GoalkeeperWorld` and changes three things for the arcade
build (the frozen world is untouched):

  1. Uses `ArcadeFlyBody` (faster ground + vertical flight) instead of `FlyBody`.
  2. `sample_shot` can produce LOFTED / high shots (aimed up toward the crossbar)
     in addition to low rolling shots, and accepts an explicit height target so
     the player can place shots low or high.
  3. Corrected save/goal scoring: a ball that crosses the goal line inside the
     posts is a GOAL even if it grazed the fly on the way in (a deflection that
     fails to keep it out is NOT a save). Only a ball actually kept out counts as
     a SAVE. This fixes the sticky-contact leniency in the frozen scorer.
"""
from __future__ import annotations

import numpy as np
import mujoco

from embodiment.mujoco_world import (GoalkeeperWorld, ShotSpec, _arena_xml,
                                     GOAL_LINE_X, GOAL_HALF_WIDTH, GOAL_HEIGHT,
                                     SHOT_X, BALL_RADIUS)
from experiments.arcade_demo.arcade_body import ArcadeFlyBody


class ArcadeGoalkeeperWorld(GoalkeeperWorld):
    """GoalkeeperWorld with a flight-capable body, high shots, correct scoring."""

    def __init__(self, seed=7, timestep=None):
        # Reproduce the parent constructor but with the arcade body. We cannot
        # call super().__init__ because it hardcodes FlyBody, so we mirror it.
        self.fly = ArcadeFlyBody(extra_xml=_arena_xml(), timestep=timestep)
        m = self.fly.model
        self.model = m
        self.data = self.fly.data
        self.rng = np.random.default_rng(seed)
        self._ball_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_qadr = m.jnt_qposadr[m.body_jntadr[self._ball_bid]]
        self._ball_dofadr = m.jnt_dofadr[m.body_jntadr[self._ball_bid]]
        self._ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self._fly_geoms = self._collect_fly_geoms()
        self._post_gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                           for n in ("post_left", "post_right", "crossbar")}
        self._reset_shot_state()

    def _reset_shot_state(self):
        """Per-shot outcome + interception diagnostics (contact != save)."""
        self._result = None
        self._contact = False
        self._keeper_contact = False
        self._intercepted = False           # latch: one impulse per shot
        self._deflected_save = False
        self._touch_but_goal = False
        self._step_count = 0
        self._contact_time = None
        self._contact_position = None
        self._pre_contact_velocity = None
        self._post_contact_velocity = None
        self._ball_no_gravity = False
        self._loft_vz = 0.0
        self._stall_steps = 0               # consecutive not-approaching steps
        self._terminal_step = None          # step at which the shot resolved

    # ------------------------------------------------------------- shots
    def sample_shot(self, group=None, height_frac=None) -> ShotSpec:
        """Arcade shot: lateral aim by group + optional HIGH aim.

        height_frac in [0, 1]: 0 = low rolling ball (as in the frozen world),
        1 = aimed near the crossbar. If None, a random mix of low and lofted
        shots is drawn so flight is actually exercised.
        """
        if group is None:
            group = str(self.rng.choice(["left", "center", "right"]))
        aim = {"left": 0.55, "center": 0.0, "right": -0.55}[group]
        aim += float(self.rng.uniform(-0.12, 0.12))
        speed = float(self.rng.uniform(4.5, 6.0))
        if height_frac is None:
            # ~half low, ~half lofted, so the fly must sometimes fly
            height_frac = float(self.rng.choice([0.0, 0.0, 0.55, 0.85]))
        height_frac = float(np.clip(height_frac, 0.0, 1.0))
        # low ball sits at BALL_RADIUS; a high aim targets up toward the crossbar
        target_h = BALL_RADIUS + height_frac * (GOAL_HEIGHT - BALL_RADIUS - 0.1)
        return ShotSpec(group, aim, speed, float(target_h))

    def reset(self, shot: ShotSpec, fly_yaw=0.0):
        """Place the fly and fire the shot; supports lofted trajectories.

        For a high shot we launch the ball with an upward velocity component so
        it arrives near the target height at the goal line (a lobbed shot),
        rather than sitting at a fixed height, so it looks and plays like a real
        lofted attempt.
        """
        self._reset_shot_state()
        self.fly.set_pose(xy=(0.0, 0.0), yaw=fly_yaw)
        self.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
        self.fly.reset()
        self.fly.set_pose(xy=(0.0, 0.0), yaw=fly_yaw)
        q = self.data.qpos
        launch_h = BALL_RADIUS               # ball starts low, may be lobbed up
        q[self._ball_qadr + 0] = SHOT_X
        q[self._ball_qadr + 1] = float(np.clip(shot.y_target,
                                               -GOAL_HALF_WIDTH + 0.3,
                                               GOAL_HALF_WIDTH - 0.3))
        q[self._ball_qadr + 2] = launch_h
        q[self._ball_qadr + 3:self._ball_qadr + 7] = [1, 0, 0, 0]
        dx = GOAL_LINE_X - SHOT_X
        travel_t = abs(dx) / shot.speed
        vy = (shot.y_target - q[self._ball_qadr + 1]) / max(travel_t, 1e-3)
        # Vertical launch. The arena is tiny (cm scale) and flight is short
        # (~0.6 s), so a full ballistic arc under the -981 cm/s^2 gravity is
        # absurdly large. For a HIGH shot we instead disable gravity on the ball
        # (freeze its z drift) and give it a straight-line rise to the target
        # height at the goal line, so a "high shot" is a clean lofted line the
        # flying fly can meet. Low shots roll flat (vz = 0). This is an arcade
        # trajectory choice, not physical ballistics.
        target_h = float(shot.height)
        rise = target_h - launch_h
        if rise > 0.05:
            vz = rise / max(travel_t, 1e-3)      # straight rise to target height
            self._ball_no_gravity = True
            self._loft_vz = vz
        else:
            vz = 0.0
            self._ball_no_gravity = False
            self._loft_vz = 0.0
        self.data.qvel[self._ball_dofadr + 0] = -shot.speed
        self.data.qvel[self._ball_dofadr + 1] = vy
        self.data.qvel[self._ball_dofadr + 2] = vz
        mujoco.mj_forward(self.model, self.data)
        self._shot = shot
        return self._observe_ball()

# --- arcade goalkeeper reach (non-contact interception volume) ---
    # A reach envelope centered on the ACTUAL fly thorax world pose. It does NOT
    # generate rigid-body contact; it is a proximity test. When the live ball
    # enters it, we register ONE interception, apply ONE bounded deflection
    # impulse, and end the shot as a SAVE. The fly must physically be at the
    # right (y,z) at the right time -- the volume does not auto-save the goal.
    REACH_RADIUS_Y = 0.45     # cm lateral half-reach
    REACH_RADIUS_Z = 0.45     # cm vertical half-reach
    REACH_RADIUS_X = 1.0      # cm depth half-reach: spans the fly's rest depth
    #                           (x~0) to the goal line (x=-0.3) + ball radius, so
    #                           a correctly-positioned keeper intercepts in time.

    def step(self, dt_s):
        """Advance physics; loft the ball for high shots; run the arcade
        non-contact interception check.
        """
        loft = getattr(self, "_ball_no_gravity", False) and not self._intercepted
        if loft:
            g = float(self.model.opt.gravity[2])
            ball_mass = float(self.model.body_subtreemass[self._ball_bid])
            # cancel gravity on the ball for the duration of the loft so it
            # follows a clean straight rise (arcade trajectory). Independent of
            # any fly contact -- only the interception event stops the loft.
            self.data.xfrc_applied[self._ball_bid, 2] = -ball_mass * g
        self.fly.step(dt_s)
        if loft:
            self.data.xfrc_applied[self._ball_bid, 2] = 0.0
        self._step_count += 1
        self._check_interception()
        self._check_outcome()
        return self._observe_ball()

    def _check_interception(self):
        """Non-contact reach test on the ACTUAL fly pose.

        Registers keeper CONTACT and applies ONE bounded deflection impulse the
        first time the live ball enters the reach envelope. It does NOT decide
        the outcome -- the ball keeps being simulated and the authoritative
        goal-line test (`_check_outcome`) determines SAVE vs GOAL afterwards.
        The impulse is latched (`_intercepted`) so prolonged overlap cannot
        repeatedly accelerate the ball.
        """
        if self._intercepted:
            return
        ball = self._observe_ball()
        bx, by, bz = ball["pos"]
        vx, vy, vz = ball["vel"]
        # only a live shot still heading goalward can be intercepted.
        if vx >= -1e-3:
            return
        fly = self.fly.position
        fx, fy, fz = float(fly[0]), float(fly[1]), float(fly[2])
        dx = (bx - fx) / self.REACH_RADIUS_X
        dy = (by - fy) / self.REACH_RADIUS_Y
        dz = (bz - fz) / self.REACH_RADIUS_Z
        if dx * dx + dy * dy + dz * dz <= 1.0:
            self._register_interception(ball)

    def _register_interception(self, ball):
        """Apply ONE bounded deflection impulse (parry). Does NOT set a result.

        The ball is redirected away from the fly with a plausible, bounded
        velocity, then MuJoCo continues to simulate it. Whether it stays out
        (SAVE) or still creeps in (GOAL, keeper touch) is decided later by the
        goal-line test. Latched so it fires exactly once per shot.
        """
        vx, vy, vz = ball["vel"]
        speed = float(np.linalg.norm([vx, vy, vz]))
        fly = self.fly.position
        away_y = np.sign(float(ball["pos"][1]) - float(fly[1])) or 1.0
        away_z = np.sign(float(ball["pos"][2]) - float(fly[2])) or 1.0
        new_vx = +0.6 * speed                       # push back up-field
        new_vy = 0.35 * speed * away_y
        new_vz = 0.25 * speed * abs(away_z) + 1.0
        new_v = np.array([new_vx, new_vy, new_vz], dtype=float)
        maxs = 1.5 * max(speed, 6.0)
        nrm = np.linalg.norm(new_v)
        if nrm > maxs:
            new_v *= maxs / nrm
        # diagnostics (contact != save)
        self._keeper_contact = True
        self._intercepted = True
        self._contact = True
        self._contact_time = int(self._step_count)
        self._contact_position = [round(float(c), 4) for c in ball["pos"]]
        self._pre_contact_velocity = [round(float(vx), 4), round(float(vy), 4),
                                      round(float(vz), 4)]
        self._post_contact_velocity = [round(float(c), 4) for c in new_v]
        # apply the single impulse, then hand back to normal MuJoCo physics
        self.data.qvel[self._ball_dofadr + 0] = float(new_v[0])
        self.data.qvel[self._ball_dofadr + 1] = float(new_v[1])
        self.data.qvel[self._ball_dofadr + 2] = float(new_v[2])
        self._ball_no_gravity = False   # gravity resumes on the deflected ball

    # A shot stops being "live" once the ball is no longer approaching the goal
    # (vx not clearly goalward) while still in front of the line. Some lofted
    # shots hit the crossbar/post and stall in the goal mouth (x ~ +0.1..0.5,
    # vx -> ~0) without ever crossing x <= GOAL_LINE_X; the original two branches
    # never fired, so the episode padded to MAX_DECISIONS. We treat a persistent
    # stall as terminal so no shot runs forever and NO post-resolution frame is
    # collected. (This closes the stale-dataset padding bug.)
    _STALL_SPEED = 1.2            # total ball speed (cm/s) below which the shot
    #                               has lost its energy (dead / post-deflection
    #                               creep) and is no longer a live shot.
    _STALL_STEPS_TERMINAL = 5     # consecutive dead-speed steps -> resolved
    _AWAY_VX = 0.3                # clearly moving back up-field

    # ------------------------------------------------------------- scoring
    def _check_outcome(self):
        """Authoritative goal-line scoring (independent of keeper contact).

        A ball that crosses the goal-line plane INSIDE the posts and below the
        crossbar is a GOAL -- even if the keeper touched it (touch-but-goal).
        A ball that crosses outside the posts / above the bar is a miss (SAVE,
        not conceded). A touched ball that is clearly moving away from goal and
        safely non-threatening is a deflected SAVE. A ball that is no longer a
        live goalward threat but never crossed the line (stalled in front of the
        goal, e.g. deflected by the crossbar/post) resolves as SAVE once the
        stall persists -- it was kept out. This guarantees every shot reaches a
        terminal state and the dataset is never padded past resolution.
        """
        if self._result is not None:
            return
        ball = self._observe_ball()
        x, y, z = ball["pos"]
        vx = ball["vel"][0]
        if x <= GOAL_LINE_X:
            inside = abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT
            if inside:
                self._result = "GOAL"
                self._touch_but_goal = bool(self._keeper_contact)
            else:
                self._result = "SAVE"          # wide / over the bar
                self._deflected_save = bool(self._keeper_contact)
        elif self._keeper_contact and vx > self._AWAY_VX and x > 0.6:
            # deflected clearly back up-field and away from goal -> safe SAVE
            self._result = "SAVE"
            self._deflected_save = True
        else:
            # STALL detection: the ball is in front of the line but has lost its
            # shot energy (a lofted ball deflected by the crossbar/post that
            # dribbles slowly). A persistent low-speed state means the shot is
            # dead; resolve it by its position -- if it has effectively reached
            # the mouth inside the posts it is a (creeping) GOAL, otherwise it
            # was kept out (SAVE). This is a degenerate arcade-loft artifact, not
            # a live shot, so we stop here rather than collect dead frames.
            speed = float(np.linalg.norm(ball["vel"]))
            if x > GOAL_LINE_X and speed < self._STALL_SPEED:
                self._stall_steps += 1
            else:
                self._stall_steps = 0
            if self._stall_steps >= self._STALL_STEPS_TERMINAL:
                creeping_in = (x <= GOAL_LINE_X + 0.25 and vx < 0
                               and abs(y) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT)
                if creeping_in:
                    self._result = "GOAL"
                    self._touch_but_goal = bool(self._keeper_contact)
                else:
                    self._result = "SAVE"      # dead in front of goal, kept out
                    self._deflected_save = bool(self._keeper_contact)
        if self._result is not None and self._terminal_step is None:
            self._terminal_step = int(self._step_count)

    # ------------------------------------------------------------ diagnostics
    @property
    def shot_live(self):
        """True while the shot has not resolved (no terminal result yet)."""
        return self._result is None

    @property
    def terminal_step(self):
        """Step index at which the shot resolved (None until it does)."""
        return self._terminal_step

    @property
    def keeper_contact(self):
        return self._keeper_contact

    @property
    def deflected(self):
        return self._deflected_save

    def outcome_diagnostics(self):
        return dict(
            result=self._result,
            keeper_contact=self._keeper_contact,
            touch_but_goal=self._touch_but_goal,
            deflected_save=self._deflected_save,
            untouched_goal=bool(self._result == "GOAL" and not self._keeper_contact),
            terminal_step=self._terminal_step,
            contact_time=self._contact_time,
            contact_position=self._contact_position,
            pre_contact_velocity=self._pre_contact_velocity,
            post_contact_velocity=self._post_contact_velocity,
        )
