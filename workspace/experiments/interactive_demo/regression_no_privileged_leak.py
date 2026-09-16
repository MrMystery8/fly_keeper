"""Regression: no privileged ball state reaches the learned controller (Section 4).

In LEARNED (and NATURAL) mode the controller must NEVER receive ball position,
velocity, aim, power, direction, target, or any privileged simulator state --
only rendered pixels. This test enforces that in two complementary ways:

  1. STATIC AUDIT: ``assert_no_privileged_leak`` confirms the neural controller
     and its vision bridge hold no reference to the world/ball/shot.

  2. BEHAVIOURAL EQUIVALENCE: the learned controller's decisions depend ONLY on
     what its eye sees. We run the SAME frozen bridge on the SAME rendered
     scene twice, but with the environment's privileged ball object perturbed in
     a way that does NOT change the rendered pixels the fly's camera samples --
     it must produce byte-identical commands. Concretely, we monkeypatch the
     world's private ball-observation accessor to return GARBAGE during the
     neural controller's ``act`` and confirm the neural command stream is
     unchanged (the neural path never calls it). For the heuristic oracle the
     same perturbation DOES change behaviour, proving the probe is sensitive.

Exit non-zero on any violation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.interactive_demo.modes import (GoalkeeperEngine,
                                                assert_no_privileged_leak)
from experiments.interactive_demo.shots import science_demo_shots
from experiments.interactive_demo.artifact_integrity import load_verified_artifact


def _run_commands(engine, shot, corrupt_ball=False, n_steps=30):
    """Run a FIXED number of decisions, returning the neural command signature.

    If corrupt_ball, replace the world's ball observation with garbage strictly
    AROUND the controller's ``act`` (restored before physics/scoring, which
    legitimately need true state). A pixels-only neural controller must be
    immune; the oracle must be affected.

    We run a fixed ``n_steps`` (do NOT stop on result) so the clean and
    corrupt runs are compared over the same horizon -- otherwise a difference in
    episode length, not in the command law, could masquerade as a leak. We
    record the raw scalar command ``u`` (bridge) plus the decoded ``lateral``.
    """
    from experiments.learned_bridge.runtime import DECISION_S

    engine.new_shot(shot)
    world = engine.world
    real_observe = world._observe_ball
    garbage = {"pos": np.array([-999.0, 999.0, -999.0]),
               "vel": np.array([999.0, -999.0, 999.0])}

    sig = []
    for _ in range(n_steps):
        if corrupt_ball:
            world._observe_ball = lambda: garbage  # noqa: E731
        command, cdiag = engine.controller.act(world)
        if corrupt_ball:
            world._observe_ball = real_observe
        world.fly.set_command(command["forward"], command.get("turn", 0.0),
                              command["gait_on"],
                              lateral=command.get("lateral", 0.0))
        world.step(DECISION_S)
        sig.append((round(float(cdiag.get("bridge_u", 0.0)), 6),
                    round(float(command.get("lateral", 0.0)), 6)))
    world._observe_ball = real_observe
    return sig


def main():
    art = load_verified_artifact(strict=True)
    shots = science_demo_shots(n_per_group=1, base_seed=90004)  # a right shot
    _, group, shot = shots[-1]

    problems = []

    # --- 1. static audit across all runnable modes ---
    for mode in ("natural", "learned", "random", "heuristic"):
        eng = GoalkeeperEngine(mode, vision="normal", artifact=art)
        try:
            rep = assert_no_privileged_leak(eng)
            print(f"[audit] {mode:10s} ok={rep['ok']} "
                  f"neural={rep['is_neural']} oracle={rep['is_oracle']}")
        except AssertionError as exc:
            problems.append(f"static audit {mode}: {exc}")
        finally:
            eng.close()

    # --- 2. behavioural immunity of the learned controller ---
    # Use a FRESH engine per run: the MaleCNS brain's LIF state persists across
    # shots (only the decoder/bridge reset), so comparing two runs on one engine
    # would confound brain history with the ball-corruption probe.
    eng = GoalkeeperEngine("learned", vision="normal", artifact=art)
    clean = _run_commands(eng, shot, corrupt_ball=False)
    eng.close()
    eng = GoalkeeperEngine("learned", vision="normal", artifact=art)
    corrupt = _run_commands(eng, shot, corrupt_ball=True)
    eng.close()
    if clean != corrupt:
        problems.append("LEARNED command stream changed when ball state was "
                        "corrupted -> privileged leak into the neural controller")
    else:
        print(f"[immunity] learned: {len(clean)} commands byte-identical with "
              "corrupted ball state (pixels-only confirmed)")

    # --- 3. sensitivity check: the oracle MUST be affected ---
    eng = GoalkeeperEngine("heuristic", vision="normal", artifact=art)
    clean_h = _run_commands(eng, shot, corrupt_ball=False)
    eng.close()
    eng = GoalkeeperEngine("heuristic", vision="normal", artifact=art)
    corrupt_h = _run_commands(eng, shot, corrupt_ball=True)
    eng.close()
    if clean_h == corrupt_h:
        problems.append("probe insensitive: even the oracle was unaffected by "
                        "ball corruption -> the immunity test is not meaningful")
    else:
        print("[sensitivity] heuristic oracle DID change with corrupted ball "
              "state (probe is meaningful)")

    if problems:
        print("\n[FAIL] privileged-leak regression:")
        for pr in problems:
            print(f"  - {pr}")
        sys.exit(1)
    print("\n[PASS] no privileged ball state reaches the learned controller.")


if __name__ == "__main__":
    main()
