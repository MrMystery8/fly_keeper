# Experimental Metal backend investigation

CPU remains the default scientific/reference backend. The Metal prototype must upload `outputs/doom/malecns_v1/graph.npz` unchanged once, retain persistent state, and remain explicitly experimental until per-step CPU parity is established. See `CPU_MODEL_SPEC.md` and `REFERENCE_NOTES.md`.

The CPU backend remains the default scientific reference. `backend="metal"`
selects the optional, fixed-weight throughput backend used by FlyKeeper. It is
GPU-resident and runs above biological realtime on the tested M4, but uses
relaxed atomic propagation and is therefore not an exact-parity backend.

The frozen serial oracle, exact deterministic v2, v3 worklist primitives, exact
batched v4 experiment, sparse v6 experiment, and their tests remain available
for comparison. Build all non-oracle backends with
`python -m workspace.gpu.build_accelerators`; benchmark the selected fast path
with `python -m workspace.gpu.benchmark_v5`.
