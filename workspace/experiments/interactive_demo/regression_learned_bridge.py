"""Frozen learned-bridge regression for the interactive demo (Section 21).

Loads the frozen model through the SAME game engine the interactive demo uses
(``GoalkeeperEngine('learned')``), runs a fixed deterministic shot set, and
compares the outcomes and key neural scalars against stored expected values.

This detects accidental changes to any part of the frozen chain:
  * neuron ordering / selection (visual_manifest slice)
  * temporal feature windows (n_windows)
  * normalization (mu/sd)
  * weight/bias loading
  * DN opponent basis (IDs, drive, bounds)
  * bridge gain / command smoothing
  * decision timing / injection lag

The learned closed loop is fully deterministic (verified), so the outcomes must
match exactly and the summed scalar command `u` must match to a tight tolerance.
On mismatch the script exits non-zero and does NOT silently continue.

Expected values live in ``regression_learned_bridge_expected.json`` (frozen at
the interactive build). Regenerate ONLY with ``--bless`` after a deliberate,
documented change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.interactive_demo.modes import GoalkeeperEngine
from experiments.interactive_demo.shots import science_demo_shots
from experiments.interactive_demo.artifact_integrity import (load_verified_artifact,
                                                             IntegrityError)

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "regression_learned_bridge_expected.json"

# Deterministic regression shot set: 2 per group from the held-out base seed.
N_PER_GROUP = 2
BASE_SEED = 90000
U_SUM_TOL = 1e-3          # summed |u| tolerance (deterministic; kept small)


def run_learned_regression(artifact=None):
    art = artifact or load_verified_artifact(strict=True)
    shots = science_demo_shots(n_per_group=N_PER_GROUP, base_seed=BASE_SEED)
    engine = GoalkeeperEngine("learned", vision="normal", artifact=art)
    records = []
    try:
        for seed, group, shot in shots:
            engine.new_shot(shot)
            us, last = [], None
            while not engine.episode_done():
                last = engine.decision_step()
                us.append(last.bridge_u)
            records.append({
                "seed": int(seed),
                "group": group,
                "result": engine.finalize_result(),
                "decisions": int(last.decision + 1),
                "u_sum": round(float(np.sum(us)), 6),
                "final_fly_y": float(last.fly_y),
            })
    finally:
        engine.close()
    return {
        "model_version": art.model_version,
        "scientific_tag": art.scientific_tag,
        "n_per_group": N_PER_GROUP,
        "base_seed": BASE_SEED,
        "checksums": art.checksums,
        "records": records,
    }


def compare(expected: dict, actual: dict):
    problems = []
    exp_recs = {r["seed"]: r for r in expected["records"]}
    act_recs = {r["seed"]: r for r in actual["records"]}
    if set(exp_recs) != set(act_recs):
        problems.append(f"shot-seed set changed: expected {sorted(exp_recs)}, "
                        f"got {sorted(act_recs)}")
    for seed in sorted(set(exp_recs) & set(act_recs)):
        e, a = exp_recs[seed], act_recs[seed]
        if e["result"] != a["result"]:
            problems.append(f"seed {seed}: result {a['result']} != {e['result']}")
        if e["decisions"] != a["decisions"]:
            problems.append(f"seed {seed}: decisions {a['decisions']} != "
                            f"{e['decisions']}")
        if abs(e["u_sum"] - a["u_sum"]) > U_SUM_TOL:
            problems.append(f"seed {seed}: u_sum {a['u_sum']} != {e['u_sum']} "
                            f"(|Δ|>{U_SUM_TOL})")
    # checksum drift is a hard failure (the artifact itself changed)
    for name, cs in expected.get("checksums", {}).items():
        if actual.get("checksums", {}).get(name) != cs:
            problems.append(f"artifact checksum drift for {name}")
    return problems


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bless", action="store_true",
                   help="regenerate expected values (deliberate change only)")
    a = p.parse_args()

    try:
        actual = run_learned_regression()
    except IntegrityError as exc:
        print(f"[FAIL] integrity: {exc}", file=sys.stderr)
        sys.exit(1)

    if a.bless:
        EXPECTED_PATH.write_text(json.dumps(actual, indent=2))
        print(f"[BLESSED] wrote {EXPECTED_PATH.name} with "
              f"{len(actual['records'])} records")
        for r in actual["records"]:
            print(f"  seed {r['seed']} {r['group']:6s} -> {r['result']:4s} "
                  f"({r['decisions']} dec, u_sum={r['u_sum']:+.3f})")
        return

    if not EXPECTED_PATH.exists():
        print(f"[FAIL] no expected file {EXPECTED_PATH}; run --bless first",
              file=sys.stderr)
        sys.exit(1)

    expected = json.loads(EXPECTED_PATH.read_text())
    problems = compare(expected, actual)

    print(f"[learned-regression] {len(actual['records'])} deterministic shots "
          f"(model {actual['model_version']}, tag {actual['scientific_tag']})")
    for r in actual["records"]:
        print(f"  seed {r['seed']} {r['group']:6s} -> {r['result']:4s} "
              f"({r['decisions']} dec, u_sum={r['u_sum']:+.3f})")

    if problems:
        print("\n[FAIL] frozen learned-bridge regression detected changes:")
        for pr in problems:
            print(f"  - {pr}")
        sys.exit(1)
    print("\n[PASS] frozen learned bridge reproduces stored outcomes/scalars "
          "exactly.")


if __name__ == "__main__":
    main()
