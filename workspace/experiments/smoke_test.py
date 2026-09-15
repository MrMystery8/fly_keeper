"""Integration smoke test against the real prepared MaleCNS v1.0 graph."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workspace"))
from adapters.brain import MaleCNSBrain


def main():
    brain = MaleCNSBrain()
    assert brain.neuron_count > 160_000
    source_id = int(brain._brain.ids[brain._brain.retina[0]])
    brain.stimulate_retinal_luminance([source_id], [1.0])
    state = brain.step(10.0)
    activity = brain.read([source_id])
    assert state["sim_ms"] == 10.0
    assert activity[source_id]["spikes"] > 0
    print({"neurons": brain.neuron_count, "retinal_neuron_id": source_id, "state": state, "activity": activity})


if __name__ == "__main__":
    main()
