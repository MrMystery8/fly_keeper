"""Clean closed-loop comparison for the diagnostics report.

Runs passive / random / heuristic / malecns(dnp20) / malecns(leaky) under normal
and blind conditions and reports save rates + group breakdown. The blind control
is the key test: a genuine visual controller must differ between normal and blind.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

from experiments.embodied_flykeeper.run import run


def main():
    episodes, seed = 30, 7
    rows = []

    def rec(label, summary):
        by = summary.get("by_group", {})
        rows.append(dict(experiment=label, save_rate=summary["save_rate"],
                         left=(by.get("left") or {}).get("save_rate"),
                         center=(by.get("center") or {}).get("save_rate"),
                         right=(by.get("right") or {}).get("save_rate")))
        print(json.dumps(rows[-1]))

    # non-neural baselines (normal only; they ignore vision)
    for ctrl in ("passive", "random", "heuristic"):
        _, _, s = run(ctrl, episodes, seed, condition="normal")
        rec(f"{ctrl}", s)

    # neural controllers under normal + blind, both decoders
    for kind in ("dnp20", "leaky"):
        for cond in ("normal", "blind"):
            _, _, s = run("malecns", episodes, seed, condition=cond,
                          decoder_ids={"kind": kind})
            rec(f"malecns[{kind}]/{cond}", s)

    (OUT / "closed_loop_suite.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved closed_loop_suite.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
