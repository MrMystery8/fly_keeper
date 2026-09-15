"""Use-case-agnostic, fixed-baseline access to the unmodified MaleCNS runtime.

The adapter never changes DoomFly's graph, LIF constants, kernel, retinal
projection, or plasticity. Generic stimulation is bounded, one-step additive
external current delivered to real biological neuron IDs.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream" / "doomfly"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


class _AdapterCheckpointGame:
    """Minimal fixed-baseline recovery context; no environment state is saved."""
    scenario = "combat_survival"

    def __init__(self):
        self.episode = 0
        self.tick = 0

    def observation(self):
        return {"episode": self.episode, "tick": self.tick, "adapter_mode": True}


class MaleCNSBrain:
    """Real MaleCNS v1.0 graph through DoomFly's integrity-checked NativeBrain.

    `stimulate` accepts source/body IDs and bounded current-equivalent values.
    Values are additive only for the immediately following `step`; this makes
    timing explicit and prevents a forgotten stimulus from becoming persistent.
    The default 30 mV-equivalent bound matches the upstream maximum retinal
    current scale, but is a safety policy, not a biological calibration.
    """

    def __init__(self, graph_path: Path | None = None, *, max_abs_current_mv: float = 30.0, backend: str = "cpu"):
        if not math.isfinite(max_abs_current_mv) or max_abs_current_mv <= 0:
            raise ValueError("max_abs_current_mv must be a positive finite value")
        if backend not in {"cpu", "metal"}:
            raise ValueError("backend must be 'cpu' or 'metal'")
        if backend == "metal":
            from gpu.backend import MetalBackendUnavailable
            raise MetalBackendUnavailable("Metal is experimental and unavailable until full Xcode and CPU/GPU parity validation are complete; use backend='cpu'.")
        self.backend = backend
        self.graph_path = Path(graph_path or UPSTREAM / "outputs/doom/malecns_v1/graph.npz")
        self.max_abs_current_mv = float(max_abs_current_mv)
        self._new_runtime()
        self._graph_sha256 = self._digest(self.graph_path)

    @classmethod
    def from_config(cls, config: Mapping):
        """Construct from externally supplied configuration without embedded IDs."""
        stimulation = config.get("adapter", {}).get("stimulation", {})
        return cls(max_abs_current_mv=float(stimulation.get("max_abs_current_mv", 30.0)), backend=config.get("brain", {}).get("backend", "cpu"))

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _new_runtime(self) -> None:
        from doom.native import NativeBrain
        self._brain = NativeBrain(self.graph_path)
        self._id_to_index = {int(identifier): index for index, identifier in enumerate(self._brain.ids)}
        self._retina_id_to_position = {int(self._brain.ids[index]): position for position, index in enumerate(self._brain.retina)}
        self._pending_current = np.zeros(self._brain.n, dtype=np.float32)
        self._pending_luminance = np.zeros(len(self._brain.retina), dtype=np.float32)

    @property
    def neuron_count(self) -> int:
        return self._brain.n

    @property
    def timestep_ms(self) -> float:
        return self._brain.dt

    def reset(self) -> None:
        """Restore a fresh fixed-weight native state; graph and kernel are unchanged."""
        self._new_runtime()

    def _indices(self, neuron_ids: Iterable[int]) -> tuple[list[int], np.ndarray]:
        ids = [int(identifier) for identifier in neuron_ids]
        unknown = [identifier for identifier in ids if identifier not in self._id_to_index]
        if unknown:
            raise KeyError(f"Unknown MaleCNS neuron ID(s): {', '.join(str(i) for i in unknown[:5])}")
        return ids, np.asarray([self._id_to_index[identifier] for identifier in ids], dtype=np.int32)

    def stimulate(self, neuron_ids: Iterable[int], values: Iterable[float]) -> None:
        """Queue bounded additive external current for any real MaleCNS neuron.

        IDs are biological source IDs—not graph-array positions. Currents are
        summed when a neuron is repeated and consumed by the next `step`; they
        never alter weights, thresholds, or plasticity.
        """
        ids, indices = self._indices(neuron_ids)
        current = np.asarray(list(values), dtype=np.float32)
        if len(ids) != len(current) or not np.all(np.isfinite(current)):
            raise ValueError("neuron_ids and finite values must have equal length")
        if np.any(np.abs(current) > self.max_abs_current_mv):
            raise ValueError(f"Current exceeds configured ±{self.max_abs_current_mv:g} mV-equivalent safety bound")
        proposed = self._pending_current.copy()
        np.add.at(proposed, indices, current)
        if np.any(np.abs(proposed[indices]) > self.max_abs_current_mv):
            raise ValueError("Combined queued current exceeds the configured safety bound")
        self._pending_current[:] = proposed

    def stimulate_retinal_luminance(self, neuron_ids: Iterable[int], values: Iterable[float]) -> None:
        """Queue existing R1-R6 luminance input as an optional sensory encoder."""
        ids = [int(identifier) for identifier in neuron_ids]
        luminance = np.asarray(list(values), dtype=np.float32)
        if len(ids) != len(luminance) or not np.all(np.isfinite(luminance)):
            raise ValueError("neuron_ids and finite values must have equal length")
        non_retinal = [identifier for identifier in ids if identifier not in self._retina_id_to_position]
        if non_retinal:
            raise ValueError("Retinal luminance requires IDs from the mapped R1-R6 population")
        self._pending_luminance.fill(0)
        for identifier, value in zip(ids, luminance):
            self._pending_luminance[self._retina_id_to_position[identifier]] = value

    def _activate(self, indices: np.ndarray) -> None:
        """Register externally driven nodes with the unchanged active-node kernel."""
        for index in np.unique(indices):
            if self._brain.active_flag[index] == 0:
                position = int(self._brain.nactive[0])
                self._brain.active[position] = index
                self._brain.active_flag[index] = 1
                self._brain.nactive[0] += 1

    def step(self, duration_ms: float = 1.0) -> dict:
        """Advance exact upstream native dynamics with queued inputs for one interval."""
        from doom.native import _f
        if not math.isfinite(duration_ms):
            raise ValueError("duration_ms must be finite")
        steps = int(round(duration_ms / self._brain.dt))
        if steps < 1:
            raise ValueError(f"duration_ms must be at least {self._brain.dt} ms")
        self._activate(np.flatnonzero(self._pending_current))
        # NativeBrain.step's unmodified input setup, plus declared external current.
        self._brain.luminance += (1 - math.exp(-steps * self._brain.dt / 10)) * (np.clip(self._pending_luminance, 0, 1) - self._brain.luminance)
        self._brain.drive.fill(0)
        self._brain.drive[self._brain.lamina] = 12.0
        self._brain.drive[self._brain.retina] = 30 * self._brain.luminance / (0.02 + self._brain.luminance)
        self._brain.drive += self._pending_current
        self._brain.counts.fill(0)
        clock = np.asarray([self._brain.cursor], dtype=np.int64)
        arrays = [self._brain.ptr, self._brain.post, self._brain.weight, self._brain.v, self._brain.g, self._brain.refractory, self._brain.drive, self._brain.previous_drive, self._brain.queue, self._brain.queue_count, clock]
        started = time.perf_counter()
        _f(self._brain.n, *[array.ctypes.data for array in arrays], steps, self._brain.dt, *[array.ctypes.data for array in [self._brain.counts, self._brain.active, self._brain.active_flag, self._brain.nactive, self._brain.last]])
        elapsed_s = time.perf_counter() - started
        self._brain.cursor = int(clock[0])
        self._brain.total_spikes += int(self._brain.counts.sum())
        self._brain.sim_ms += steps * self._brain.dt
        self._pending_current.fill(0)
        self._pending_luminance.fill(0)
        return {"sim_ms": self._brain.sim_ms, "spikes": int(self._brain.counts.sum()), "wall_seconds": elapsed_s}

    def read(self, neuron_ids: Iterable[int]) -> dict[int, dict[str, float | int]]:
        """Return generic fixed-baseline spike count and membrane voltage by body ID."""
        ids, indices = self._indices(neuron_ids)
        return {identifier: {"spikes": int(self._brain.counts[index]), "voltage_mv": float(self._brain.v[index])} for identifier, index in zip(ids, indices)}

    def _checkpoint_identity(self):
        from doom.native import BUILD
        return {"adapter": "generic-malecns-base", "dataset": "malecns_v1", "graph_sha256": self._graph_sha256, "kernel": BUILD, "plasticity": False, "retinal_mapping": "upstream-unchanged"}

    def save_checkpoint(self, directory: Path | str) -> str:
        """Pass through DoomFly's hash-checked fixed-baseline checkpoint format."""
        from doom.checkpoint import Checkpoints
        from doom.engine import NeuralControls
        if np.any(self._pending_current) or np.any(self._pending_luminance):
            raise RuntimeError("Step or clear queued stimulation before saving a checkpoint")
        return Checkpoints(Path(directory), self._checkpoint_identity()).save(self._brain, NeuralControls([], mode="bci"), _AdapterCheckpointGame(), {"adapter": "generic-malecns-base"})

    def load_checkpoint(self, directory: Path | str) -> dict:
        """Restore through DoomFly validation logic; only fixed baseline is accepted."""
        from doom.checkpoint import Checkpoints
        from doom.engine import NeuralControls
        restored = Checkpoints(Path(directory), self._checkpoint_identity()).restore(self._brain, NeuralControls([], mode="bci"), _AdapterCheckpointGame())
        if restored is None:
            raise FileNotFoundError(f"No checkpoint found in {directory}")
        self._pending_current.fill(0)
        self._pending_luminance.fill(0)
        return restored
