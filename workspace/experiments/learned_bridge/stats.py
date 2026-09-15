"""Statistical uncertainty for the learned-bridge comparisons (Section 29).

Reads evaluate_test.json (+ shuffle_controls.json) and reports, for the key
contrasts, Wilson binomial CIs (already per condition), a two-proportion z-test
and its normal-approximation CI on the DIFFERENCE bridge - natural, and a
parametric bootstrap CI on each save rate (resampling Bernoulli outcomes from the
observed counts). No new simulation -- pure post-hoc statistics on the recorded
saves/n.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "workspace" / "outputs" / "learned_bridge"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 4), round(min(1, c + h), 4))


def bootstrap_rate(k, n, iters=10000, seed=0):
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    p = k / n
    samples = rng.binomial(n, p, size=iters) / n
    return (round(float(np.percentile(samples, 2.5)), 4),
            round(float(np.percentile(samples, 97.5)), 4))


def two_prop_ztest(k1, n1, k2, n2):
    """Test H0: p1 == p2 (bridge vs natural). Returns diff, z, p, 95% CI(diff)."""
    p1 = k1 / n1; p2 = k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 > 0 else 0.0
    # normal-approx two-sided p
    pval = math.erfc(abs(z) / math.sqrt(2))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return dict(diff=round(diff, 4), z=round(z, 3), p_value=round(pval, 6),
                ci95_diff=(round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)))


def main():
    ev = json.load(open(OUT / "evaluate_test.json"))
    sh = json.load(open(OUT / "shuffle_controls.json")) if (OUT / "shuffle_controls.json").exists() else {}

    def kn(key):
        v = ev[key]; return v["saves"], v["n"]

    conds = ["passive", "random", "heuristic", "natural", "bridge_normal",
             "bridge_blind", "bridge_mirrored", "bridge_shuffled", "bridge_off"]
    out = {"per_condition": {}}
    for c in conds:
        if c in ev:
            k, n = kn(c)
            out["per_condition"][c] = dict(saves=k, n=n, rate=round(k / n, 4),
                                           wilson95=wilson(k, n),
                                           bootstrap95=bootstrap_rate(k, n))
    # per-step shuffle from shuffle_controls
    if "shuffle_perstep" in sh:
        v = sh["shuffle_perstep"]; k, n = v["saves"], v["n"]
        out["per_condition"]["shuffle_perstep"] = dict(
            saves=k, n=n, rate=round(k / n, 4), wilson95=wilson(k, n),
            bootstrap95=bootstrap_rate(k, n))

    # key contrasts vs natural
    kb, nb = kn("bridge_normal"); kn_, nn = kn("natural")
    out["contrasts"] = {
        "bridge_vs_natural": two_prop_ztest(kb, nb, kn_, nn),
        "bridge_vs_blind": two_prop_ztest(kb, nb, *kn("bridge_blind")),
        "bridge_vs_bridge_off": two_prop_ztest(kb, nb, *kn("bridge_off")),
    }
    if "shuffle_perstep" in sh:
        v = sh["shuffle_perstep"]
        out["contrasts"]["bridge_vs_shuffle_perstep"] = two_prop_ztest(
            kb, nb, v["saves"], v["n"])

    (OUT / "stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
