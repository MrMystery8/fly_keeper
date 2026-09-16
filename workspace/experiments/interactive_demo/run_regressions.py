"""Run the full interactive-demo regression suite (Sections 20, 21, 4).

Executes, in order:
  1. artifact integrity     -- frozen bridge checksums + metadata (Section 22)
  2. bridge-OFF regression  -- Natural MaleCNS unchanged by the UI layer (Sec 20)
  3. frozen learned-bridge  -- exact frozen outcomes/scalars reproduce (Sec 21)
  4. no-privileged-leak     -- ball state never reaches the neural controller (Sec 4)

Exits non-zero if ANY check fails, so CI / the demo build can gate on it.

    upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.run_regressions
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable

CHECKS = [
    ("artifact integrity",
     [PY, "-m", "workspace.experiments.interactive_demo.artifact_integrity"]),
    ("bridge-OFF regression (Natural MaleCNS unchanged)",
     [PY, "-m", "workspace.experiments.learned_bridge.regression_bridge_off",
      "--episodes", "6", "--seed", "9000"]),
    ("frozen learned-bridge regression",
     [PY, "-m", "workspace.experiments.interactive_demo.regression_learned_bridge"]),
    ("no privileged leak into learned controller",
     [PY, "-m", "workspace.experiments.interactive_demo.regression_no_privileged_leak"]),
]


def main():
    results = []
    for name, cmd in CHECKS:
        print("\n" + "=" * 72)
        print(f"[suite] {name}")
        print("=" * 72, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        ok = proc.returncode == 0
        results.append((name, ok))

    print("\n" + "#" * 72)
    print("# interactive-demo regression suite summary")
    print("#" * 72)
    all_ok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    if not all_ok:
        print("\n[SUITE FAIL] one or more regressions failed.")
        sys.exit(1)
    print("\n[SUITE PASS] all interactive-demo regressions passed.")


if __name__ == "__main__":
    main()
