"""Targeted reward-modulated plasticity engine (EXPERIMENTAL).

Wraps the unmodified MaleCNS CPU runtime and applies a reward-modulated,
subthreshold-eligibility learning rule to ONLY the synapses in the pathway mask
(build_mask.py). All other synapses are frozen. The connectome file on disk is
never modified; weights live in the in-memory array of a fresh brain instance
and can be reset to the exact fixed-weight baseline at any time.

=================  EXPERIMENTAL PLASTICITY MECHANISM  =========================
This learning rule is a modeling choice, NOT a validated model of Drosophila
synaptic plasticity. It borrows structure from the DoomFly v6 experimental rule
(eligibility traces, bounded sign-preserving efficacies, three-factor reward
modulation) but is a new, minimal, interpretable rule adapted to the silent-
visual-projection bootstrap problem identified by the pathway audit.
==============================================================================

Rule (per plastic synapse i->j on the mask):
  Each control window (20 ms):
    pre_activity_i   = presynaptic spike count in the window (>=0)
    post_depol_j     = max(0, v_j - V_REST)        # SUBTHRESHOLD depolarization
                       (does NOT require the postsynaptic neuron to spike -
                        essential because the visual-projection targets are
                        nearly silent)
    coact_ij        = pre_activity_i * post_depol_j
    eligibility_ij  <- decay * eligibility_ij + coact_ij      # decaying trace

  At episode end, with scalar reward R in {+1 (SAVE), -1 (GOAL)}:
    dw_ij = lr * R * eligibility_ij
    w_ij  = clip(w_ij + dw_ij, lo_ij, hi_ij)   # bounds are 0.1x..2.0x baseline,
                                                # sign-preserving

The reward is task outcome ONLY. No ball position, direction label, target, or
heuristic action is ever injected into the network or the eligibility.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

V_REST = -52.0
MASK_PATH = ROOT / "workspace" / "outputs" / "plasticity" / "plasticity_mask.npz"


class PlasticPathway:
    """Holds the mask, eligibility traces, bounds, and the update rule."""

    def __init__(self, brain, *, lr=0.02, elig_decay=0.8, min_frac=0.1,
                 max_frac=2.0, elig_mode="subthreshold"):
        self.b = brain._brain
        m = np.load(MASK_PATH, allow_pickle=True)
        self.edge = m["edge_index"].astype(np.int64)
        self.pre = m["pre_index"].astype(np.int64)
        self.post = m["post_index"].astype(np.int64)
        self.baseline = m["baseline_weight"].astype(np.float64)
        # per-edge sign-preserving bounds relative to baseline
        self.lo = self.baseline * min_frac
        self.hi = self.baseline * max_frac
        self.lr = float(lr)
        self.elig_decay = float(elig_decay)
        self.elig_mode = elig_mode
        self.eligibility = np.zeros(len(self.edge), dtype=np.float64)
        # bookkeeping for the bootstrap diagnostics
        self.n_updates = 0
        self.total_abs_dw = 0.0

    # ------------------------------------------------------------- eligibility
    def accumulate(self):
        """Call once per control window (after brain.step) to update traces."""
        pre_act = self.b.counts[self.pre].astype(np.float64)     # pre spikes
        if self.elig_mode == "subthreshold":
            post_signal = np.maximum(0.0, self.b.v[self.post] - V_REST)
        else:  # spike-based (for comparison / bootstrap test)
            post_signal = self.b.counts[self.post].astype(np.float64)
        coact = pre_act * post_signal
        self.eligibility = self.elig_decay * self.eligibility + coact

    def eligible_count(self):
        return int((self.eligibility > 1e-9).sum())

    # ------------------------------------------------------------------ update
    def apply_reward(self, reward):
        """Apply dw = lr * reward * eligibility to masked weights, bounded."""
        dw = self.lr * float(reward) * self.eligibility
        w = self.b.weight[self.edge].astype(np.float64) + dw
        w = np.clip(w, self.lo, self.hi)
        applied = w - self.b.weight[self.edge].astype(np.float64)
        self.b.weight[self.edge] = w.astype(np.float32)
        self.n_updates += int((np.abs(applied) > 1e-9).sum())
        self.total_abs_dw += float(np.abs(applied).sum())
        if not np.isfinite(self.b.weight[self.edge]).all():
            raise RuntimeError("Nonfinite weight after plasticity update")

    def clear_eligibility(self):
        self.eligibility[:] = 0.0

    # ------------------------------------------------------------- weight I/O
    def current_weights(self):
        return self.b.weight[self.edge].copy()

    def reset_to_baseline(self):
        self.b.weight[self.edge] = self.baseline.astype(np.float32)
        self.clear_eligibility()

    def set_weights(self, w):
        self.b.weight[self.edge] = np.asarray(w, dtype=np.float32)

    def weight_report(self):
        w = self.b.weight[self.edge].astype(np.float64)
        frac = w / self.baseline
        return dict(n_plastic=len(self.edge),
                    n_changed=int(np.count_nonzero(w != self.baseline)),
                    mean_frac=round(float(frac.mean()), 4),
                    min_frac=round(float(frac.min()), 4),
                    max_frac=round(float(frac.max()), 4),
                    total_abs_dw=round(self.total_abs_dw, 4),
                    n_updates=self.n_updates)
