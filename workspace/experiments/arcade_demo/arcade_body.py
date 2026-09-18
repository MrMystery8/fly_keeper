"""Arcade flight locomotion: engineered takeoff + lateral flight motor layer.

WHY FLIGHT (not ground strafing). The frozen legged strafe controller is stable
only at low speed; pushing it faster overshoots and makes the articulated
free-joint body skid/bounce. At a stable setting the fly reaches only ~0.34 cm in
a full episode, so it cannot cover the wider arcade goal (±~0.9 cm) within the
~0.7-1.0 s ball flight. Ground strafing is therefore the wrong regime for the
arcade goalkeeper. This module replaces it with a deliberately ENGINEERED
takeoff + airborne-thrust controller: bounded PD forces on the thorax, with
damping, acceleration/velocity limits, and orientation stabilization. It is NOT
a biological flight model.

Control principle (force/actuator, never teleport / never per-frame velocity
overwrite): airborne motion is produced by bounded forces
    F_lat  = K_lat  * u_lat  * MAX      - D_lat * v_lat
    F_vert = hover(gravity comp) + K_z * (h* - h) - D_z * v_z
plus a bounded orientation-stabilizing torque. The fly stays a real dynamic
MuJoCo body and is hit by the ball through normal collision response.

State machine (continuous neural commands drive it; NO privileged ball state):
    READY            small |u| -> stand, tiny ground correction
    GROUND_CORRECTION small-moderate |u_lat| -> gentle grounded nudge
    TAKEOFF          large |u_lat| or |u_vert| -> brief upward launch
    FLIGHT           airborne -> strong lateral thrust + height hold/rise
    DIVE             negative vertical demand -> descend
    RECOVER          command released -> damp to rest, settle to ready height
Hysteresis + a takeoff cooldown prevent rapid mode flapping.

Command dict consumed: forward, lateral, turn, gait_on, vertical.
"""
from __future__ import annotations

import numpy as np

from embodiment.fly_body import FlyBody, LEGS, TRIPOD_B


class ArcadeFlyBody(FlyBody):
    """Engineered takeoff + lateral-flight goalkeeper body (force-driven)."""

    # ------- thrust / speed limits (bounded, damped: stable) -------
    # Tuned so FULL |u_lat|=1 reaches ~the goalpost (±1.5 cm) within a ~0.9 s
    # flight, not far beyond it; the oracle/bridge command proportionally.
    LAT_THRUST = 30.0        # lateral accel authority in flight
    LAT_DAMP = 7.0           # lateral velocity damping (1/s)
    MAX_LAT_SPEED = 5.0      # cm/s lateral speed cap (airborne)
    GROUND_LAT_THRUST = 14.0 # gentler lateral authority while grounded
    GROUND_MAX_SPEED = 3.0

    # ------- vertical / hover -------
    READY_HEIGHT = 0.0       # ready hover height above stand (0 = on ground)
    KEEPER_HEIGHT = 0.35     # useful airborne "ready" hover height (cm)
    MAX_HEIGHT = 1.45        # near crossbar
    KZ = 90.0                # height P gain
    DZ = 9.0                 # vertical velocity damping
    RISE_SPEED = 9.0         # cm/s target climb at full up-demand
    DIVE_SPEED = 9.0         # cm/s target descent at full down-demand

    # ------- orientation stabilization -------
    ORI_KP = 40.0            # upright restoring torque gain
    ORI_KD = 4.0             # angular velocity damping
    YAW_KP = 20.0            # keep facing the field (yaw ~ 0)
    YAW_KD = 3.0

    # ------- takeoff logic -------
    TAKEOFF_LAT = 0.45       # |u_lat| above this triggers takeoff
    TAKEOFF_VERT = 0.20      # u_vert above this triggers a full airborne takeoff
    TAKEOFF_IMPULSE = 6.0    # cm/s initial upward kick at takeoff
    LAND_HEIGHT = 0.04       # below this (and no demand) -> grounded
    TAKEOFF_COOLDOWN = 5     # decision steps before re-evaluating land->air

    # ---- continuous LOW-HOP regime (deadband removal, Phase 4) ----
    # A small vertical demand below TAKEOFF_VERT should NOT do nothing (that was
    # a severe deadband: u_vert 0..0.15 -> 0 cm, then a cliff to a full launch).
    # Instead it commands a bounded grounded hop whose target height scales
    # continuously with u_vert, so small u_vert -> small hop / stay low, and it
    # blends smoothly into the full takeoff at TAKEOFF_VERT. Still bounded PD
    # forces (no teleport / no velocity overwrite).
    LOW_HOP_MAX_H = 0.30     # max height (cm) reachable in the grounded low-hop

    # NOTE: Arcade goalkeeper REACH is handled by a NON-CONTACT interception
    # volume in the arcade world (see ArcadeGoalkeeperWorld), NOT by any dynamic
    # collision geom on the body. An earlier dynamic "keeper_reach" sphere was
    # removed because continuous rigid-body contact deflected the ball
    # prematurely and destabilized interception.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stand_z = None
        self.state = "READY"
        self._cooldown = 0
        self._airborne = False

    def reset(self):
        super().reset()
        self._stand_z = None
        self.state = "READY"
        self._cooldown = 0
        self._airborne = False

    def set_command(self, forward, turn, gait_on, lateral=0.0, vertical=0.0):
        super().set_command(forward, turn, gait_on, lateral=lateral)
        self._command["vertical"] = float(np.clip(vertical, -1.0, 1.0))

    # ------------------------------------------------------------------ helpers
    def _upright_torque(self, mass):
        """Bounded torque to keep the body upright and facing the field."""
        # body +z axis in world (from quaternion); restore toward world +z.
        w, x, y, z = self.orientation_quat
        # world-frame tilt of the body's up axis
        up_x = 2 * (x * z + w * y)
        up_y = 2 * (y * z - w * x)
        wvec = self.data.qvel[self.root_dofadr + 3:self.root_dofadr + 6]
        tqx = -self.ORI_KP * (-up_y) - self.ORI_KD * wvec[0]
        tqy = -self.ORI_KP * (up_x) - self.ORI_KD * wvec[1]
        # yaw toward 0 (face field)
        yaw = self.heading
        tqz = -self.YAW_KP * yaw - self.YAW_KD * wvec[2]
        return mass * np.array([tqx, tqy, tqz])

    # ------------------------------------------------------------------ drive
    def _apply_cpg(self):
        cmd = self._command
        cmd.setdefault("vertical", 0.0)
        self.data.ctrl[:] = self.stance_ctrl
        self.data.xfrc_applied[self.thorax_bid, :] = 0.0

        if self._stand_z is None:
            self._stand_z = float(self.position[2])

        u_lat = float(cmd.get("lateral", 0.0))
        u_vert = float(cmd.get("vertical", 0.0))
        mass = float(self.model.body_subtreemass[self.thorax_bid])
        vel = self.data.qvel[self.root_dofadr:self.root_dofadr + 3]
        height = float(self.position[2]) - self._stand_z

        if self._cooldown > 0:
            self._cooldown -= 1

        # ---- state transitions (continuous commands only; no ball state) ----
        want_air = (abs(u_lat) >= self.TAKEOFF_LAT or u_vert >= self.TAKEOFF_VERT)
        if not self._airborne:
            if want_air and self._cooldown == 0:
                self._airborne = True
                self.state = "TAKEOFF"
                self._cooldown = self.TAKEOFF_COOLDOWN
                # brief physical upward kick (bounded impulse via a one-tick
                # velocity add is avoided; instead a strong upward force below)
            elif 0.03 < u_vert < self.TAKEOFF_VERT:
                self.state = "LOW_HOP"      # small continuous vertical (no cliff)
            elif abs(u_lat) > 0.05:
                self.state = "GROUND_CORRECTION"
            else:
                self.state = "READY"
        else:
            if u_vert < -0.1:
                self.state = "DIVE"
            elif (height < self.LAND_HEIGHT and abs(u_lat) < 0.1
                  and u_vert < 0.05 and self._cooldown == 0):
                self._airborne = False
                self.state = "RECOVER"
            else:
                self.state = "FLIGHT"

        airborne = self._airborne
        # legs animate whenever moving/airborne (visual only)
        active = airborne or abs(u_lat) > 0.02 or abs(u_vert) > 0.02
        if active:
            for leg in LEGS:
                acts = self.leg_act[leg]
                nom = self._nominal.get(leg, {})
                phase = self._phase + (np.pi if leg in TRIPOD_B else 0.0)
                lift = max(0.0, np.cos(phase))
                self._set_clamped(acts["femur"],
                                  nom.get("femur", 0.0) - self.LIFT_AMP * lift)
                self.data.ctrl[acts["adhere"]] = 0.0 if (airborne or lift > 0.4) else 1.0
        else:
            for leg in LEGS:
                self.data.ctrl[self.leg_act[leg]["adhere"]] = 1.0

        gravity = float(self.model.opt.gravity[2])   # ~ -981 cm/s^2

        if not airborne:
            # grounded: gentle bounded lateral correction (PD).
            target_v = np.clip(u_lat, -1, 1) * self.GROUND_MAX_SPEED
            fy = self.GROUND_LAT_THRUST * mass * (target_v - vel[1])
            self.data.xfrc_applied[self.thorax_bid, 1] = fy
            # CONTINUOUS LOW-HOP (deadband removal): a small vertical demand
            # (0.03 <= u_vert < TAKEOFF_VERT) commands a bounded low hop whose
            # target height scales linearly with u_vert up to LOW_HOP_MAX_H, so
            # small u_vert -> small hop and the response blends smoothly into the
            # full takeoff at TAKEOFF_VERT (which reaches ~LOW_HOP_MAX_H). No cliff.
            if u_vert >= 0.03:
                frac = min(u_vert, self.TAKEOFF_VERT) / self.TAKEOFF_VERT
                low_target = frac * self.LOW_HOP_MAX_H
                self.data.xfrc_applied[self.thorax_bid, 2] = (
                    mass * (-gravity)
                    + self.KZ * mass * (low_target - height)
                    - self.DZ * mass * vel[2])
            elif height > 0.02:
                # no demand: let gravity/contact hold it down; settle damper
                self.data.xfrc_applied[self.thorax_bid, 2] = \
                    mass * (-gravity) - self.DZ * mass * vel[2] \
                    - self.KZ * mass * height
            return

        # ---- AIRBORNE: bounded lateral thrust + vertical hover/rise/dive ----
        # lateral PD thrust with velocity cap
        des_v = np.clip(u_lat, -1, 1) * self.MAX_LAT_SPEED
        f_lat = self.LAT_THRUST * mass * np.clip(u_lat, -1, 1) \
            - self.LAT_DAMP * mass * vel[1]
        # cap lateral speed: if over cap and thrust would increase it, damp only
        if abs(vel[1]) > self.MAX_LAT_SPEED and np.sign(f_lat) == np.sign(vel[1]):
            f_lat = -self.LAT_DAMP * mass * vel[1]
        self.data.xfrc_applied[self.thorax_bid, 1] = f_lat

        # vertical: hover (gravity comp) + height control toward a target height.
        # u_vert in [-1,1] maps monotonically to a target height spanning the
        # ground up to the crossbar, so the controller can hold ANY height
        # (including mid) rather than snapping between keeper-height and max.
        if u_vert >= 0.0:
            target_h = u_vert * self.MAX_HEIGHT
        else:
            target_h = 0.0
        target_h = float(np.clip(target_h, 0.0, self.MAX_HEIGHT))
        f_vert = (mass * (-gravity)
                  + self.KZ * mass * (target_h - height)
                  - self.DZ * mass * vel[2])
        self.data.xfrc_applied[self.thorax_bid, 2] = f_vert

        # hold forward (x) position on the goal line where shots cross (bounded
        # PD, no teleport), so the airborne keeper stays at the interception
        # depth rather than drifting up-field.
        GOAL_X = -0.15
        fx = -40.0 * mass * (float(self.position[0]) - GOAL_X) - 6.0 * mass * vel[0]
        self.data.xfrc_applied[self.thorax_bid, 0] = fx

        # orientation stabilization torque (bounded)
        tq = self._upright_torque(mass)
        tq = np.clip(tq, -8.0 * mass, 8.0 * mass)
        self.data.xfrc_applied[self.thorax_bid, 3:6] = tq
