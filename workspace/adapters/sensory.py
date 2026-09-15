"""Optional sensory encoders; none is privileged as the generic API."""
from __future__ import annotations

import numpy as np


class SensoryEncoder:
    def encode(self, observation):
        raise NotImplementedError("Implement a documented environment-specific encoding.")


class RetinalLuminanceEncoder(SensoryEncoder):
    """Adapter for DoomFly's existing mapped R1-R6 luminance pathway."""

    def __init__(self, neuron_ids):
        self.neuron_ids = tuple(int(identifier) for identifier in neuron_ids)

    def encode(self, luminance):
        values = np.asarray(luminance, dtype=np.float32)
        if values.shape != (len(self.neuron_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("Expected one finite luminance value for each mapped retinal neuron")
        return {"kind": "retinal_luminance", "neuron_ids": self.neuron_ids, "values": values}
