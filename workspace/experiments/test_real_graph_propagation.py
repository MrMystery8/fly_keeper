"""Real-graph integration test for generic stimulation, readout, and recovery."""
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workspace"))
from adapters.brain import MaleCNSBrain


def strongest_excitatory_edge(brain):
    """Select an actual retained positive edge; do not create test connectivity."""
    positive = np.flatnonzero(brain._brain.weight > 0)
    edge = positive[np.argmax(brain._brain.weight[positive])]
    pre_index = int(np.searchsorted(brain._brain.ptr, edge, side="right") - 1)
    post_index = int(brain._brain.post[edge])
    return int(brain._brain.ids[pre_index]), int(brain._brain.ids[post_index])


def main():
    brain = MaleCNSBrain.from_config({"adapter": {"stimulation": {"max_abs_current_mv": 30.0}}})
    pre_id, post_id = strongest_excitatory_edge(brain)

    # Matched no-input control: neither selected real neuron spikes.
    brain.step(10.0)
    control = brain.read([pre_id, post_id])
    assert control[pre_id]["spikes"] == 0
    assert control[post_id]["spikes"] == 0

    # Generic body-ID current drives the presynaptic cell and its retained edge.
    brain.reset()
    brain.stimulate([pre_id], [30.0])
    state = brain.step(10.0)
    propagated = brain.read([pre_id, post_id])
    assert state["sim_ms"] == 10.0
    assert propagated[pre_id]["spikes"] > 0
    assert propagated[post_id]["spikes"] > 0

    # Adapter checkpoint API delegates to DoomFly's integrity-checked baseline format.
    with tempfile.TemporaryDirectory(prefix="malecns-adapter-checkpoint-") as directory:
        generation = brain.save_checkpoint(directory)
        saved_ms = brain._brain.sim_ms
        saved_voltage = propagated[post_id]["voltage_mv"]
        brain.reset()
        restored = brain.load_checkpoint(directory)
        assert restored["adapter"] == "generic-malecns-base"
        assert generation
        assert brain._brain.sim_ms == saved_ms
        assert brain.read([post_id])[post_id]["voltage_mv"] == saved_voltage

    print({"pre_id": pre_id, "post_id": post_id, "control": control,
           "propagated": propagated, "checkpoint_generation": generation})


if __name__ == "__main__":
    main()
