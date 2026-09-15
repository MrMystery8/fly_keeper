# Reusable MaleCNS workspace

`upstream/doomfly/` is a clean DoomFly checkout. This directory contains only a thin future-integration boundary and no application-specific encoder, decoder, or learning policy.

Run the real-graph smoke test after preparation:

```sh
upstream/doomfly/.venv-neural/bin/python workspace/experiments/smoke_test.py
```

`adapters/brain.py` exposes a bounded generic external-current API for any real MaleCNS body ID, structured spike/voltage readouts, and fixed-baseline checkpoint save/load through DoomFly's integrity-checked format. Inputs persist for one `step` only. This adapter-level external current is an experimenter-controlled modeling input, not a claim about sensory physiology.

Retinal luminance remains available through `stimulate_retinal_luminance` and `RetinalLuminanceEncoder`, but it is now only one optional sensory encoder. `output.py` remains a documented placeholder. Experimental plasticity remains disabled in `config/default.yaml`.

Run the real-graph propagation and checkpoint integration test:

```sh
upstream/doomfly/.venv-neural/bin/python workspace/experiments/test_real_graph_propagation.py
```
