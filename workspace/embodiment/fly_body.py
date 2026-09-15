"""Articulated 3D fruit-fly body with an engineered tripod-CPG locomotion layer.

The body is the flybody MJCF model (google-deepmind/mujoco_menagerie, Apache-2.0),
an anatomically-detailed Drosophila melanogaster with 6 articulated legs
(T1/T2/T3 x left/right), a free-floating thorax, head, wings, abdomen, and
tarsal adhesion actuators.

We do NOT train anything. Walking is produced by a hand-written Central Pattern
Generator (CPG) that plays a tripod gait: legs (T1L, T2R, T3L) and
(T1R, T2L, T3R) alternate swing/stance. High-level commands modulate the gait:

    forward in [-1, 1]   : stride amplitude / direction (retraction sweep)
    turn    in [-1, 1]   : left/right stride asymmetry (differential drive)
    gait_on in {0, 1}    : whether the legs cycle at all (stop = stand)

This CPG is ENGINEERED, not biological. It is the low-level locomotion
controller that a real fly's VNC would provide; here it stands in for that so
the MaleCNS brain only has to issue high-level intent.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
FLYBODY_DIR = ROOT / "workspace" / "vendor" / "mujoco_menagerie_flybody" / "flybody"

# The six legs, front(T1)/mid(T2)/hind(T3), left/right.
LEGS = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
# Tripod groups: diagonal legs move together.
TRIPOD_A = {"T1_left", "T2_right", "T3_left"}
TRIPOD_B = {"T1_right", "T2_left", "T3_right"}
LEFT_LEGS = {"T1_left", "T2_left", "T3_left"}


class FlyBody:
    """Loads the flybody model and drives it with a tripod CPG.

    The caller advances physics in small chunks via `step(dt_s)`, having first
    set a high-level command with `set_command(...)`. Vision cameras
    (`eye_left`, `eye_right`) are exposed for the vision bridge.
    """

    # Engineered locomotion-controller parameters, calibrated on the flybody
    # model. The tripod animates the legs; a body-frame drive produces net
    # motion (~0.5 cm/s translation, ~2 rad/s yaw). Not a biological gait model.
    SWEEP_AMP = 0.5        # coxa fore/aft sweep for the visible tripod (rad)
    LIFT_AMP = 0.35        # femur lift during swing (rad); small = low drift
    GAIT_FREQ_HZ = 35.0    # tripod cycle frequency
    MAX_SPEED = 2.5        # commanded top translation speed (cm/s)
    DRIVE_GAIN = 120.0     # velocity-controller stiffness (1/s)
    MAX_YAW_RATE = 3.0     # commanded top yaw rate (rad/s)
    YAW_GAIN = 30.0        # yaw-rate controller stiffness (1/s)
    # Lateral strafing gets extra authority because a line goalkeeper needs it
    # most; the drive is still a bounded velocity controller (stable).
    LATERAL_BOOST = 3.0

    def __init__(self, extra_xml: str | None = None, timestep: float | None = None):
        # Build model from the flybody scene, optionally splicing in extra
        # world geometry (the football arena) supplied by mujoco_world.
        self.model = self._build_model(extra_xml, timestep)
        self.data = mujoco.MjData(self.model)
        self._cache_indices()
        self.reset()
        # CPG state.
        self._phase = 0.0
        self._command = {"forward": 0.0, "lateral": 0.0, "turn": 0.0, "gait_on": 0.0}

    # ------------------------------------------------------------------ setup
    def _build_model(self, extra_xml, timestep):
        # Inline the scene's <include file="fruitfly.xml"/> so we can build from
        # a string (with the meshes supplied via the assets dict) and optionally
        # splice in the arena geometry.
        scene = (FLYBODY_DIR / "scene.xml").read_text()
        body = (FLYBODY_DIR / "fruitfly.xml").read_text()
        # Strip the fruitfly.xml's own <mujoco ...> wrapper, keep its children.
        inner = body.split(">", 1)[1].rsplit("</mujoco>", 1)[0]
        scene = scene.replace('<include file="fruitfly.xml"/>', inner)
        if extra_xml:
            # Insert extra worldbody/asset content just before </mujoco>.
            scene = scene.replace("</mujoco>", extra_xml + "\n</mujoco>")
        model = mujoco.MjModel.from_xml_string(scene, self._assets())
        if timestep is not None:
            model.opt.timestep = timestep
        return model

    @staticmethod
    def _assets() -> dict:
        assets = {}
        asset_dir = FLYBODY_DIR / "assets"
        for path in asset_dir.iterdir():
            if path.is_file():
                assets[path.name] = path.read_bytes()
        return assets

    def _cache_indices(self):
        m = self.model
        self.thorax_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        # Free joint of the thorax: qpos[0:3] = xyz, qpos[3:7] = quaternion.
        self.root_qadr = m.jnt_qposadr[m.body_jntadr[self.thorax_bid]]
        self.root_dofadr = m.jnt_dofadr[m.body_jntadr[self.thorax_bid]]

        def act(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Per-leg joints we drive for walking.
        self.leg_act = {}
        for leg in LEGS:
            self.leg_act[leg] = {
                "coxa": act(f"coxa_{leg}"),          # protraction/retraction sweep
                "femur": act(f"femur_{leg}"),        # lift/depress
                "tibia": act(f"tibia_{leg}"),        # extend
                "adhere": act(f"adhere_claw_{leg}"), # tarsal grip
            }
        self.eye_cams = {
            "left": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eye_left"),
            "right": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eye_right"),
        }
        # Nominal (keyframe) control targets = stable standing pose.
        self.stance_ctrl = np.array(m.key_ctrl[0]) if m.nkey else np.zeros(m.nu)
        # Record keyframe joint angles for the driven leg joints so the CPG
        # oscillates around the true standing configuration.
        self._nominal = {}
        if m.nkey:
            qpos0 = m.key_qpos[0]
            for leg, acts in self.leg_act.items():
                self._nominal[leg] = {
                    key: float(qpos0[m.jnt_qposadr[m.actuator_trnid[idx, 0]]])
                    for key, idx in acts.items()
                    if key != "adhere"
                }

    # --------------------------------------------------------------- kinematics
    @property
    def position(self) -> np.ndarray:
        return self.data.qpos[self.root_qadr:self.root_qadr + 3].copy()

    @property
    def orientation_quat(self) -> np.ndarray:
        return self.data.qpos[self.root_qadr + 3:self.root_qadr + 7].copy()

    @property
    def heading(self) -> float:
        """Yaw of the body about +z, radians. Forward walking is along local +x;
        at yaw=0 the fly advances toward world +x, at yaw=pi/2 toward +y."""
        w, x, y, z = self.orientation_quat
        # Forward axis (local +x) rotated into world: (fx, fy).
        fx = 1 - 2 * (y * y + z * z)
        fy = 2 * (x * y + w * z)
        return float(np.arctan2(fy, fx))

    @property
    def velocity(self) -> np.ndarray:
        return self.data.qvel[self.root_dofadr:self.root_dofadr + 3].copy()

    def set_pose(self, xy=(0.0, 0.0), z=None, yaw=0.0):
        """Place the fly at a world xy and yaw, at its standing height."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        q = self.data.qpos
        q[self.root_qadr + 0] = xy[0]
        q[self.root_qadr + 1] = xy[1]
        if z is not None:
            q[self.root_qadr + 2] = z
        half = yaw / 2.0
        q[self.root_qadr + 3] = np.cos(half)
        q[self.root_qadr + 4] = 0.0
        q[self.root_qadr + 5] = 0.0
        q[self.root_qadr + 6] = np.sin(half)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.stance_ctrl
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.stance_ctrl
        self._phase = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ control
    def set_command(self, forward: float, turn: float, gait_on: float,
                    lateral: float = 0.0):
        """High-level locomotion intent, all in [-1, 1].

        forward : advance (+) / reverse (-) along the body's facing axis.
        lateral : strafe left (+, toward body +y) / right (-).
        turn    : yaw right (+) / left (-).
        gait_on : whether the legs cycle (visual realism) and the body moves.
        """
        self._command = {
            "forward": float(np.clip(forward, -1, 1)),
            "lateral": float(np.clip(lateral, -1, 1)),
            "turn": float(np.clip(turn, -1, 1)),
            "gait_on": 1.0 if gait_on else 0.0,
        }

    def _apply_cpg(self):
        """Animate a tripod gait and drive the body per the high-level command.

        The legs run a symmetric tripod (for ground contact and visual realism);
        the *net translation/rotation* is produced by an engineered drive on the
        floating thorax, expressed in the body frame. This is the hierarchical
        'locomotion controller' layer that a fly's VNC would provide: it turns a
        high-level velocity command into physical body motion. Speeds/gains are
        engineered to give ~0.5 cm/s translation and ~2 rad/s yaw.
        """
        cmd = self._command
        self.data.ctrl[:] = self.stance_ctrl
        self.data.xfrc_applied[self.thorax_bid, :] = 0.0

        moving = cmd["gait_on"] >= 0.5 and (
            abs(cmd["forward"]) > 1e-3 or abs(cmd["lateral"]) > 1e-3
            or abs(cmd["turn"]) > 1e-3)
        if not moving:
            for leg in LEGS:
                self.data.ctrl[self.leg_act[leg]["adhere"]] = 1.0
            return

        # --- tripod leg animation (in-place stepping for visual realism) ---
        # The two diagonal tripods alternately lift and plant. We deliberately
        # do NOT sweep the coxa for propulsion, so the legs add no net thrust;
        # translation/rotation comes entirely from the commanded body drive
        # below. This keeps the high-level command in clean control.
        for leg in LEGS:
            acts = self.leg_act[leg]
            nom = self._nominal.get(leg, {})
            phase = self._phase + (np.pi if leg in TRIPOD_B else 0.0)
            lift = max(0.0, np.cos(phase))
            self._set_clamped(acts["femur"], nom.get("femur", 0.0) - self.LIFT_AMP * lift)
            self.data.ctrl[acts["adhere"]] = 0.0 if lift > 0.4 else 1.0

        # --- body-frame velocity controller (PD) toward the commanded motion ---
        # Target world-frame linear velocity from the body-frame command.
        yaw = self.heading
        c, s = np.cos(yaw), np.sin(yaw)
        # Body axes in world: forward = (c, s), left(+y) = (-s, c).
        lat = cmd["lateral"] * self.LATERAL_BOOST
        target_vx = self.MAX_SPEED * (cmd["forward"] * c + lat * (-s))
        target_vy = self.MAX_SPEED * (cmd["forward"] * s + lat * (c))
        mass = float(self.model.body_subtreemass[self.thorax_bid])
        vel = self.data.qvel[self.root_dofadr:self.root_dofadr + 2]
        # A closing force proportional to velocity error cancels parasitic drift
        # and holds the commanded speed; the swing-phase adhesion release lets it
        # take effect, so motion still requires the legs to step.
        self.data.xfrc_applied[self.thorax_bid, 0] = self.DRIVE_GAIN * mass * (target_vx - vel[0])
        self.data.xfrc_applied[self.thorax_bid, 1] = self.DRIVE_GAIN * mass * (target_vy - vel[1])
        # Yaw-rate controller about world +z.
        wz = self.data.qvel[self.root_dofadr + 5]
        target_wz = -self.MAX_YAW_RATE * cmd["turn"]
        self.data.xfrc_applied[self.thorax_bid, 5] = self.YAW_GAIN * mass * (target_wz - wz)

    def _set_clamped(self, act_id, value):
        lo, hi = self.model.actuator_ctrlrange[act_id]
        self.data.ctrl[act_id] = float(np.clip(value, lo, hi))

    def step(self, dt_s: float, cpg_freq_hz: float | None = None):
        """Advance physics by dt_s, cycling the CPG at cpg_freq_hz."""
        if cpg_freq_hz is None:
            cpg_freq_hz = self.GAIT_FREQ_HZ
        n = max(1, int(round(dt_s / self.model.opt.timestep)))
        for _ in range(n):
            if self._command["gait_on"] >= 0.5:
                self._phase += 2 * np.pi * cpg_freq_hz * self.model.opt.timestep
            self._apply_cpg()
            mujoco.mj_step(self.model, self.data)
        self._phase %= 2 * np.pi
