"""Run the full experiment suite: baselines + MaleCNS + sensory controls.

Produces a single aggregated JSON/CSV comparing:
  - heuristic (mechanical ceiling), random (chance), malecns (closed-loop brain)
  - MaleCNS sensory controls: normal / blind / mirrored / static_ball

Usage:
    python -m workspace.experiments.embodied_flykeeper.evaluate --episodes 60 --seed 7
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.embodied_flykeeper.run import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    results = []
    started = time.perf_counter()

    def record(label, summary):
        by = summary.get("by_group", {})
        row = {"experiment": label, "controller": summary["controller"],
               "condition": summary["condition"], "episodes": summary["episodes"],
               "saves": summary["saves"], "save_rate": summary["save_rate"],
               "left_rate": (by.get("left") or {}).get("save_rate"),
               "center_rate": (by.get("center") or {}).get("save_rate"),
               "right_rate": (by.get("right") or {}).get("save_rate"),
               "wall_seconds": summary["wall_seconds"]}
        results.append(row)
        print(json.dumps(row))

    # Baselines and the neural controller under normal vision.
    for ctrl in ("heuristic", "random", "malecns"):
        _, _, summary = run(ctrl, args.episodes, args.seed, condition="normal")
        record(f"{ctrl}/normal", summary)

    # Sensory controls for MaleCNS: does behavior depend on the visual input?
    for cond in ("blind", "mirrored", "static_ball"):
        _, _, summary = run("malecns", args.episodes, args.seed, condition=cond)
        record(f"malecns/{cond}", summary)

    out = ROOT / "workspace" / "outputs" / "embodied_flykeeper" / "suite"
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite_summary.json").write_text(json.dumps(
        {"episodes": args.episodes, "seed": args.seed,
         "wall_seconds": round(time.perf_counter() - started, 1),
         "results": results}, indent=2))
    with (out / "suite_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsuite written to {out}")


if __name__ == "__main__":
    main()
