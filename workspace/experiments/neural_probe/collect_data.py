"""Collect static + dynamic probe data and save compact per-neuron arrays.

Saves, per collection (static / dynamic):
  - condition list
  - per (condition, trial) dense window matrix (windows x n_neurons) as int16,
    stored sparsely via np.savez per condition to keep size manageable.

Because only ~10k of 166,700 neurons ever fire, we store the union of active
neuron indices plus a compact (conditions, trials, windows, n_active) tensor.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"


def _active_union(collection):
    active = set()
    for cond, trials in collection.items():
        for tr in trials:
            for idx, _ in tr["windows"]:
                active.update(idx.tolist())
    return np.array(sorted(active), dtype=np.int64)


def _tensor(collection, active):
    """(n_cond, n_trials, n_windows, n_active) int16 tensor + condition list."""
    pos = {int(a): k for k, a in enumerate(active)}
    conds = list(collection.keys())
    n_trials = min(len(v) for v in collection.values())
    n_win = min(len(tr["windows"]) for v in collection.values() for tr in v)
    T = np.zeros((len(conds), n_trials, n_win, len(active)), dtype=np.int16)
    for ci, c in enumerate(conds):
        for ti in range(n_trials):
            tr = collection[c][ti]
            for wi in range(n_win):
                idx, cnt = tr["windows"][wi]
                cols = np.fromiter((pos[int(i)] for i in idx), dtype=np.int64, count=len(idx))
                T[ci, ti, wi, cols] = cnt
    return conds, T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from experiments.neural_probe.probe import NeuralProbe
    probe = NeuralProbe(seed=7)
    ids = probe.ids

    print("collecting STATIC probe data ...")
    static = probe.collect(trials=6, window_ms=5.0, n_windows=24, warmup_ms=25.0)
    a_static = _active_union(static)
    conds_s, T_s = _tensor(static, a_static)
    np.savez_compressed(OUT / "static_tensor.npz", ids=ids,
                        active=a_static, conditions=np.array(conds_s),
                        tensor=T_s, window_ms=5.0)
    print(f"  static: {len(conds_s)} conditions, tensor {T_s.shape} "
          f"({T_s.nbytes/1e6:.1f} MB), {len(a_static)} active neurons")

    print("collecting DYNAMIC probe data ...")
    dyn = probe.collect_dynamic(trials=6, window_ms=5.0, frames=24, warmup_ms=15.0)
    a_dyn = _active_union(dyn)
    conds_d, T_d = _tensor(dyn, a_dyn)
    np.savez_compressed(OUT / "dynamic_tensor.npz", ids=ids,
                        active=a_dyn, conditions=np.array(conds_d),
                        tensor=T_d, window_ms=5.0)
    print(f"  dynamic: {len(conds_d)} conditions, tensor {T_d.shape} "
          f"({T_d.nbytes/1e6:.1f} MB), {len(a_dyn)} active neurons")
    print("saved static_tensor.npz and dynamic_tensor.npz")


if __name__ == "__main__":
    main()
