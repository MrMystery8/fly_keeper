"""Goalkeeper controller-mode abstraction for the interactive penalty demo.

One ``GoalkeeperEngine`` owns the frozen scientific stack (world + fixed MaleCNS
brain + vision + decoder + optional learned bridge) and exposes a clean,
UI-friendly seam:

    engine.new_shot(shot)          # place a human-chosen penalty
    diag = engine.decision_step()  # run ONE 20 ms scientific decision
    engine.result                  # None | 'SAVE' | 'GOAL'

Every controller mode reuses the EXISTING, validated code paths:

  * NATURAL   -> BridgeController(bridge=None)                (Natural MaleCNS)
  * LEARNED   -> BridgeController(bridge=<frozen LearnedBridge>, enabled=True)
  * BRIDGE_OFF-> the frozen LearnedBridge object, enabled=False (== Natural)
  * RANDOM    -> RandomController (existing)
  * HEURISTIC -> HeuristicController (existing oracle; privileged ball state)
  * PLASTICITY-> historical experimental mode (see PlasticityUnavailable)

Vision conditions (normal / blind / mirrored / shuffled-fixed / shuffled-perstep)
reuse the existing VisionBridge / ShuffleVisionBridge implementations verbatim.

SCIENTIFIC INVARIANT (enforced here): in LEARNED / NATURAL / BRIDGE_OFF modes the
only thing entering the brain is rendered pixels via ``vision.perceive()``. The
ball state (``world._observe_ball()``) is read by the engine ONLY for rendering,
scoring and debug -- and, for the HEURISTIC oracle, by that controller alone,
which never touches the brain. ``assert_no_privileged_leak`` documents and guards
this boundary.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GoalkeeperWorld, GOAL_LINE_X
from embodiment.vision_bridge import VisionBridge
from embodiment.motor_decoder import DescendingMotorDecoder
from experiments.embodied_flykeeper.controllers import (RandomController,
                                                        HeuristicController)
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN
from experiments.learned_bridge.bridge import (VisualFeatureExtractor,
                                               LinearBridgeModel, LearnedBridge)
from experiments.learned_bridge.runtime import (BridgeController, DECISION_MS,
                                                DECISION_S, MAX_DECISIONS)
from experiments.learned_bridge.shuffle_controls import ShuffleVisionBridge

from experiments.interactive_demo.artifact_integrity import (ModelArtifact,
                                                             load_verified_artifact)

# ----------------------------------------------------------------- modes/vision
MODES = ("natural", "learned", "random", "heuristic", "plasticity")

MODE_LABELS = {
    "natural": "Natural Fly",
    "learned": "Learned Bridge Fly",
    "random": "Random Fly",
    "heuristic": "Heuristic Fly",
    "plasticity": "Plasticity Fly",
}

MODE_DESCRIPTIONS = {
    "natural": "Native fixed MaleCNS connectivity. No learned bridge.",
    "learned": "Fixed MaleCNS + frozen 829-parameter visual->DN bridge.",
    "random": "Random lateral control (chance baseline).",
    "heuristic": "Oracle controller with direct simulator-state access.",
    "plasticity": "Historical: reward-modulated plasticity (negative result).",
}

# Which modes actually run the fixed MaleCNS brain (neural pipeline).
NEURAL_MODES = ("natural", "learned")
# Which modes are neural controllers vs privileged/oracle vs random.
ORACLE_MODES = ("heuristic",)

VISION_CONDITIONS = ("normal", "blind", "mirrored", "shuffled_fixed",
                     "shuffled_perstep")

VISION_LABELS = {
    "normal": "Normal Vision",
    "blind": "Blind",
    "mirrored": "Mirrored Vision",
    "shuffled_fixed": "Shuffled Vision (fixed permutation)",
    "shuffled_perstep": "Temporal / Per-frame Visual Shuffle",
}

# Historical numbers from LEARNED_BRIDGE_REPORT.md (48-shot held-out test set).
# Shown for context only; NEVER used as live results.
PLASTICITY_HISTORICAL = {
    "save_rate": 0.375,
    "note": ("Targeted reward-modulated plasticity on the optic-lobe->visual-"
             "projection route FAILED: it strengthened common-mode transmission "
             "but never built left/right opponent structure (trained ~= passive "
             "~= 37.5%, vision-independent). See LEARNED_BRIDGE_REPORT.md."),
}


class PlasticityUnavailable(RuntimeError):
    """The plasticity-trained state is not integrated into this game build."""


@dataclass
class StepDiag:
    """Per-decision diagnostics surfaced to the UI/debug panel.

    All values are read-outs of the scientific loop; none are fed back into the
    learned controller. Privileged ball fields are for rendering/scoring only.
    """
    decision: int
    mode: str
    vision: str
    bridge_on: bool
    bridge_u: float = 0.0
    left_dn_spikes: float = 0.0
    right_dn_spikes: float = 0.0
    dn_asymmetry: float = 0.0
    inj_total_mv: float = 0.0
    retinal_mean: float = 0.0
    brain_spikes: int = 0
    move: str = "STAY"
    lateral: float = 0.0
    # privileged (rendering/scoring/debug ONLY -- never enters the controller)
    fly_y: float = 0.0
    ball_x: float = 0.0
    ball_y: float = 0.0
    result: Optional[str] = None


class GoalkeeperEngine:
    """Owns the frozen stack for one controller mode and steps it decision-by-decision.

    Construct once per (mode, vision) selection. Reuse across shots via
    ``new_shot``. Not thread-safe; the game runs it inside a single worker
    thread (see game loop), decoupled from rendering.
    """

    def __init__(self, mode: str, *, vision: str = "normal",
                 artifact: ModelArtifact | None = None, seed: int = 1,
                 random_seed: int = 7):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if vision not in VISION_CONDITIONS:
            raise ValueError(f"unknown vision {vision!r}; choose from {VISION_CONDITIONS}")
        self.mode = mode
        self.vision_condition = vision
        self.seed = seed
        self.random_seed = random_seed
        self._decision = 0
        self._artifact = artifact

        if mode == "plasticity":
            raise PlasticityUnavailable(
                "Plasticity Fly is a historical experimental mode and is not "
                "runnable in this interactive build (the plasticity-trained state "
                "is intentionally not loaded, per the demo handoff Section 6/29). "
                f"Historical result: {PLASTICITY_HISTORICAL['save_rate']:.0%} "
                "save rate, vision-independent. See LEARNED_BRIDGE_REPORT.md.")

        # The MuJoCo world is always present (it is the physical body + ball).
        self.world = GoalkeeperWorld(seed=seed)
        self.brain = None
        self.vision = None
        self.decoder = None
        self.bridge = None
        self.controller = None
        self._build()

    # ------------------------------------------------------------------ build
    def _make_vision(self):
        """Reuse the EXACT validated vision path for the chosen condition."""
        from adapters.brain import MaleCNSBrain
        self.brain = MaleCNSBrain(backend="cpu")
        cond = self.vision_condition
        if cond in ("shuffled_fixed", "shuffled_perstep"):
            mode = "fixed" if cond == "shuffled_fixed" else "perstep"
            self.vision = ShuffleVisionBridge(
                self.world.fly, self.brain, camera="eye_left",
                condition="shuffled", shuffle_mode=mode)
        else:
            # normal / blind / mirrored are native VisionBridge conditions.
            self.vision = VisionBridge(self.world.fly, self.brain,
                                       camera="eye_left", condition=cond)

    def _make_decoder(self):
        # Same decoder the frozen scientific evaluation used: reads the SAME real
        # DNs used as the motor basis (opponent L/R pairs).
        self.decoder = DescendingMotorDecoder(
            left_ids=list(LEFT_DN), right_ids=list(RIGHT_DN),
            turn_gain=2.5, forward_bias=0.0, smoothing=0.5)

    def _make_bridge(self, enabled: bool) -> LearnedBridge:
        """Rebuild the frozen LearnedBridge ONLY from the verified artifact."""
        art = self._artifact or load_verified_artifact(strict=True)
        self._artifact = art
        extractor = VisualFeatureExtractor(art.graph_index[:art.n_neurons],
                                           n_windows=art.n_windows)
        model = LinearBridgeModel(w=art.w, b=art.b, mu=art.mu, sd=art.sd,
                                  feature_order=art.feature_order)
        basis = DNMotorBasis(left_ids=art.left_dn, right_ids=art.right_dn,
                             drive_mv=art.drive_mv, max_abs_mv=art.max_abs_mv)
        return LearnedBridge(extractor, model, basis, gain=art.runtime_gain,
                             enabled=enabled, cmd_smoothing=art.cmd_smoothing)

    def _build(self):
        mode = self.mode
        if mode in NEURAL_MODES:
            self._make_vision()
            self._make_decoder()
            if mode == "learned":
                self.bridge = self._make_bridge(enabled=True)
            else:  # natural == bridge OFF
                self.bridge = None
            self.controller = BridgeController(self.brain, self.vision,
                                               self.decoder, bridge=self.bridge)
        elif mode == "random":
            self.controller = RandomController(self.random_seed)
        elif mode == "heuristic":
            self.controller = HeuristicController(GOAL_LINE_X)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(mode)

    # ------------------------------------------------------------- properties
    @property
    def is_neural(self) -> bool:
        return self.mode in NEURAL_MODES

    @property
    def is_oracle(self) -> bool:
        return self.mode in ORACLE_MODES

    @property
    def bridge_enabled(self) -> bool:
        return bool(self.bridge is not None and self.bridge.enabled)

    @property
    def result(self):
        return self.world.result

    @property
    def contact(self):
        return self.world.contact

    def set_bridge_enabled(self, enabled: bool):
        """Toggle the frozen bridge ON/OFF at runtime (learned mode only).

        When OFF the learned mode reduces bit-for-bit to Natural MaleCNS (proved
        by the bridge-off regression). No other state changes.
        """
        if self.bridge is not None:
            self.bridge.enabled = bool(enabled)

    # ------------------------------------------------------------------ shots
    def new_shot(self, shot):
        """Place a penalty and reset the controller/decoder/bridge state."""
        self.world.reset(shot)
        self.controller.reset()
        self._decision = 0
        self._shot = shot

    def decision_step(self) -> StepDiag:
        """Run exactly ONE 20 ms scientific decision + physics micro-step.

        Returns a StepDiag with neural read-outs (for the debug panel) and the
        privileged rendering/scoring fields. The controller sees only what its
        own ``act`` reads -- for neural modes that is pixels; for the heuristic
        oracle that is the true ball state (by design, clearly labelled oracle).
        """
        # 1. the controller decides (pixels-only for neural modes).
        command, cdiag = self.controller.act(self.world)
        # 2. drive the CPG and advance the coupled fly+ball physics by 20 ms.
        self.world.fly.set_command(command["forward"], command.get("turn", 0.0),
                                   command["gait_on"],
                                   lateral=command.get("lateral", 0.0))
        self.world.step(DECISION_S)
        # 3. read privileged ball state for RENDERING/SCORING ONLY.
        ball = self.world._observe_ball()
        diag = StepDiag(
            decision=self._decision,
            mode=self.mode,
            vision=self.vision_condition,
            bridge_on=self.bridge_enabled,
            bridge_u=float(cdiag.get("bridge_u", 0.0)),
            left_dn_spikes=float(cdiag.get("left_spikes", 0.0) or 0.0),
            right_dn_spikes=float(cdiag.get("right_spikes", 0.0) or 0.0),
            dn_asymmetry=float(cdiag.get("asymmetry", 0.0) or 0.0),
            inj_total_mv=float(cdiag.get("inj_total_mv", 0.0) or 0.0),
            retinal_mean=float(cdiag.get("retinal_mean", 0.0) or 0.0),
            brain_spikes=int(cdiag.get("spikes", 0) or 0),
            move=str(command.get("move", "STAY")),
            lateral=float(command.get("lateral", 0.0)),
            fly_y=round(float(self.world.fly.position[1]), 4),
            ball_x=round(float(ball["pos"][0]), 4),
            ball_y=round(float(ball["pos"][1]), 4),
            result=self.world.result,
        )
        self._decision += 1
        return diag

    def episode_done(self) -> bool:
        return self.world.result is not None or self._decision >= MAX_DECISIONS

    def finalize_result(self) -> str:
        """Timeout without crossing/contact counts as a conceded GOAL."""
        return self.world.result or "GOAL"

    def close(self):
        if self.brain is not None:
            self.brain.close()
            self.brain = None


def assert_no_privileged_leak(engine: GoalkeeperEngine) -> dict:
    """Static audit that neural controllers cannot see ball state.

    Confirms (by construction / attribute inspection) that a neural-mode
    controller has NO reference to the world/ball and reads only its vision +
    brain + decoder. Returns a report; raises AssertionError on violation.
    Used by the privileged-leak regression test.
    """
    report = {"mode": engine.mode, "is_neural": engine.is_neural,
              "is_oracle": engine.is_oracle}
    if engine.is_neural:
        ctrl = engine.controller
        # A neural BridgeController holds brain/vision/decoder/bridge only.
        allowed = {"brain", "vision", "decoder", "bridge"}
        held = {k for k in vars(ctrl).keys()}
        leaked = held - allowed
        assert not leaked, (f"neural controller holds unexpected refs {leaked}; "
                            "privileged state may leak into the brain")
        # The vision bridge must not hold any ball/world reference either.
        vfields = set(vars(engine.vision).keys())
        for bad in ("ball", "shot", "world", "y_target", "y_cross"):
            assert bad not in vfields, f"vision holds privileged field {bad!r}"
        report["controller_fields"] = sorted(held)
        report["ok"] = True
    else:
        report["ok"] = True
        report["note"] = ("non-neural mode; oracle intentionally reads ball "
                           "state and never touches the brain"
                           if engine.is_oracle else "random/other")
    return report
