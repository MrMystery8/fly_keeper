"""Phase 7+8: joint (u_lat,u_vert) training + on-policy DAgger for the Arcade keeper.

Phase 7 rationale: the old demos were collected on the BROKEN loft (MID/HIGH shots
never rose), so any vertical target the policy learned was against a signal that
did not physically exist. We recollect on the CORRECTED environment where LOW/MID/
HIGH are genuinely distinct goal-bound heights, and train the joint (u_lat,u_vert)
residual so vertical commitment is learned, not scripted.

Phase 8 rationale (DAgger): supervised imitation is collected along the BASE
controller's trajectory; at deployment the learned policy visits DIFFERENT states
(covariate shift) and errors compound. DAgger fixes this by:
  1. rolling out the CURRENT policy closed-loop,
  2. recording the states IT actually visits (policy-visible obs only),
  3. querying the privileged teacher for the correct action at those states,
  4. aggregating with prior data and retraining,
  5. evaluating.
Privileged simulator state is used ONLY to generate teacher labels; the policy
never observes ball truth. 1-2 aggregation rounds establish whether covariate
shift is a major issue -- we do not spin here.

All CLEAN_GOAL cells only (shot_validity), matched seeds, full MaleCNS reset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (Arcade2AxisDecoder,
                                                    DECISION_MS, DECISION_S,
                                                    MAX_DECISIONS)
from experiments.arcade_demo.arcade_shots import teacher_command
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.arcade_action_policy import (OBS_NAMES, TinyPolicy,
                                                    BinocularActionPolicyBridge)
from experiments.arcade_demo.train_arcade_action_policy import train as train_policy
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))


def valid_cells(per_cell, base_seed):
    cells, seed = [], base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            got = 0
            while got < per_cell:
                cls, _, _ = SV.classify_shot(seed, g, hf)
                if cls == SV.CLEAN_GOAL:
                    cells.append((seed, g, hname, hf)); got += 1
                seed += 1
    return cells


def _obs(world, bridge, base, previous):
    fly = world.fly
    vel = fly.data.qvel[fly.root_dofadr:fly.root_dofadr + 3]
    stand = float(getattr(fly, "_stand_z", None) or fly.position[2])
    return np.asarray((bridge.last_pL, bridge.last_pR, base[1],
                       previous[0], previous[1],
                       fly.position[1], fly.position[2] - stand,
                       vel[1], vel[2], float(getattr(fly, "_airborne", False))),
                      np.float32)


def rollout_and_label(brain, cells, policy=None, residual_scale=0.35,
                      target_mode="residual"):
    """Roll out the CURRENT controller closed-loop; label teacher at each state.

    If policy is None -> roll out the frozen supervised base (Phase-7 seed data).
    Else -> roll out the residual policy (DAgger aggregation data).
    Records only policy-visible observations; teacher labels come from privileged
    state AFTER the observation is formed (never persisted as a feature).
    """
    rows, targets, seeds_col = [], [], []
    for seed, g, hname, hf in cells:
        world = ArcadeGoalkeeperWorld(seed=seed)
        visual, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
        vision = BinocularVisionBridge(world.fly, brain, condition="both",
                                       retina_map="fullframe")
        decoder = Arcade2AxisDecoder()
        if policy is not None:
            bridge = BinocularActionPolicyBridge(visual, policy,
                                                 residual_scale=residual_scale,
                                                 mode=target_mode)
        else:
            bridge = visual
        shot = world.sample_shot(g, height_frac=hf)
        world.reset(shot); brain.reset(); bridge.reset()
        previous = np.zeros(2)
        n = 0
        while world.result is None and n < MAX_DECISIONS:
            vision.perceive()
            bridge.inject(brain)
            brain.step(DECISION_MS)
            bridge.observe(brain)
            activity = brain.read(decoder.readout_ids())
            command, _ = decoder.decode(activity)
            # base action = the supervised bridge's own command (for residual target)
            base = np.asarray(visual.command(), float)
            # policy-visible observation at the state the CONTROLLER actually visits
            ob = _obs(world, visual, base, previous)
            # privileged teacher label at THIS visited state (labels only)
            teacher = np.asarray(teacher_command(world)[0], float)
            target = teacher if target_mode == "full" else np.clip(teacher - base, -1., 1.)
            rows.append(ob); targets.append(target); seeds_col.append(seed)
            # advance the actual controller (policy if present, else base)
            if policy is not None:
                u_lat, u_vert = bridge.command(world)
                world.fly.set_command(0.0, 0.0, 1.0, lateral=u_lat, vertical=u_vert)
            else:
                world.fly.set_command(command["forward"], command.get("turn", 0.),
                                      command["gait_on"],
                                      lateral=command.get("lateral", 0.),
                                      vertical=command.get("vertical", 0.))
            world.step(DECISION_S)
            previous = base
            n += 1
    return (np.asarray(rows, np.float32), np.asarray(targets, np.float32),
            np.asarray(seeds_col, np.int64))


def save_dataset(path, X, Y, S):
    np.savez(OUT / path, observations=X, targets=Y, episode_seeds=S,
             observation_names=np.asarray(OBS_NAMES))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=120000)
    p.add_argument("--rounds", type=int, default=2, help="DAgger aggregation rounds")
    p.add_argument("--residual-scale", type=float, default=0.35)
    p.add_argument("--hidden", type=int, default=12)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--target-mode", choices=("residual", "full"), default="residual")
    p.add_argument("--tag", default="dagger")
    a = p.parse_args()

    t0 = time.perf_counter()
    cells = valid_cells(a.per_cell, a.base_seed)
    print(f"[dagger] {len(cells)} CLEAN_GOAL training cells ({a.per_cell}/cell)")
    brain = MaleCNSBrain(backend="metal")
    summary = dict(per_cell=a.per_cell, rounds=a.rounds,
                   residual_scale=a.residual_scale, target_mode=a.target_mode,
                   n_cells=len(cells), rounds_log=[])
    try:
        # ---- Round 0: seed data along the frozen supervised base (Phase 7) ----
        print("[round 0] collecting base-trajectory demos (corrected env)...")
        X, Y, S = rollout_and_label(brain, cells, policy=None,
                                    target_mode=a.target_mode)
        ds = f"arcade_dagger_{a.tag}_r0.npz"
        save_dataset(ds, X, Y, S)
        pol_file = f"arcade_dagger_{a.tag}_policy_r0.npz"
        meta = train_policy(dataset=ds, out=pol_file, hidden=a.hidden,
                            epochs=a.epochs, target_mode=a.target_mode)
        summary["rounds_log"].append(dict(round=0, n_steps=int(len(X)),
                                          policy=pol_file, metrics=meta["metrics"]))
        print(f"[round 0] {len(X)} steps -> {pol_file}")

        # ---- DAgger aggregation rounds (Phase 8) ----
        for r in range(1, a.rounds + 1):
            print(f"[round {r}] rolling out current policy closed-loop...")
            policy, _ = TinyPolicy.load(OUT / pol_file)
            Xr, Yr, Sr = rollout_and_label(brain, cells, policy=policy,
                                           residual_scale=a.residual_scale,
                                           target_mode=a.target_mode)
            # aggregate with all prior data
            X = np.concatenate([X, Xr]); Y = np.concatenate([Y, Yr])
            S = np.concatenate([S, Sr])
            ds = f"arcade_dagger_{a.tag}_r{r}.npz"
            save_dataset(ds, X, Y, S)
            pol_file = f"arcade_dagger_{a.tag}_policy_r{r}.npz"
            meta = train_policy(dataset=ds, out=pol_file, hidden=a.hidden,
                                epochs=a.epochs, target_mode=a.target_mode)
            summary["rounds_log"].append(dict(round=r, n_steps=int(len(X)),
                                              new_steps=int(len(Xr)),
                                              policy=pol_file,
                                              metrics=meta["metrics"]))
            print(f"[round {r}] +{len(Xr)} new steps (total {len(X)}) -> {pol_file}")
    finally:
        brain.close()
    summary["final_policy"] = pol_file
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / f"arcade_dagger_{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[dagger] done in {summary['wall_seconds']}s; final policy {pol_file}")


if __name__ == "__main__":
    main()
