# Connecting the first future use case

The prepared baseline is intentionally generic and fixed-weight. A future task should implement only the edges of this loop, keeping the MaleCNS core intact:

```text
external environment
       ↓
workspace/adapters/sensory.py: SensoryEncoder.encode
       ↓
selected, documented MaleCNS sensory neuron IDs
       ↓
workspace/adapters/brain.py: MaleCNSBrain.stimulate / step
       ↓
selected, documented readout or descending neuron IDs
       ↓
workspace/adapters/output.py: OutputDecoder.decode
       ↓
environment action
       ↓
optional, explicitly modeled reward/modulatory signal
       ↺
```

## Starting points

1. Add the use-case configuration—not biological IDs baked into core code—to `workspace/config/default.yaml` or a new experiment-specific YAML file. Record source IDs, annotations, why they were selected, units, and whether the mapping is measured or inferred.
2. Implement the observation-to-stimulation mapping in `workspace/adapters/sensory.py`. Use `MaleCNSBrain.stimulate(body_ids, current_values)` for explicitly documented one-step external-current experiments, or `stimulate_retinal_luminance` / `RetinalLuminanceEncoder` for the preserved R1-R6 path. Do not mutate `doom/engine.py` or replace the graph.
3. Implement action decoding in `workspace/adapters/output.py`. Read selected biological source IDs through `MaleCNSBrain.read`, state filtering/windowing explicitly, and map only that activity to the external environment’s action API.
4. Keep the orchestration in a new `workspace/experiments/<use_case>.py`; it should create the environment, encode an observation, call `brain.stimulate`, `brain.step`, `brain.read`, decode an action, and call `environment.step`.
5. Add real-graph integration tests beside `workspace/experiments/smoke_test.py`. Unit mocks are fine for encoders/decoders, but the end-to-end smoke test must remain on the prepared MaleCNS graph.

## Upstream references to preserve

- Dataset/import: `upstream/doomfly/doom/datasets.json`, `doom/connectome.py`, `doom/prepare.py`
- Core LIF/native kernel: `doom/engine.py`, `doom/native.py`, `doom/kernel.cpp`
- Existing visual input example: `doom/game.py:retinal_samples` and `doom/server.py:run_loop`
- Existing output example: `doom/engine.py:NeuralControls.decode`
- Fixed checkpoint mechanism: `doom/checkpoint.py`
- Optional experimental reward/plasticity only: `doom/reward.py`, `doom_learning_v6/`, `doom/training.py`, `doom/training_checkpoint.py`

## Guardrails

Keep `plasticity.enabled: false` unless a future experiment explicitly opts in and labels the experimental model as unvalidated. Do not prune weak or self edges, crop circuitry, substitute an artificial policy, or call this an exact fly brain. Separate observed biological data from every encoding, timing, dynamical, reward, and decoding assumption.
