# `metal-deterministic-v2` propagation design

The frozen serial oracle is not modified. V2 uses the original source-major
CSR and `-fno-fast-math`; it does not use the target-major index or float
atomics.

## Required two-phase handling of one delayed source

Source-row targets are unique, so lanes may independently materialize a target
and add that source's weight to its `g` state. That alone is not sufficient for
CPU equivalence: `awaken(target)` appends a previously inactive target to the
active list in CSR-edge order.

For each dynamic delayed source rank, one threadgroup must therefore do:

1. Lane 0 publishes the source row bounds.
2. Lanes process strided edges and write a per-edge `needs_awaken` flag.
3. `threadgroup_barrier(mem_flags::mem_device)` makes all target writes and
   flags visible within the group.
4. Lane 0 scans that row in increasing edge ordinal and performs stable
   `awaken` appends only for flagged targets.
5. A second device-scope threadgroup barrier completes source rank *N* before
   rank *N + 1* starts.

The serial append scan is necessary bookkeeping, not a second propagation
algorithm. It preserves the CPU active-list semantics exactly and avoids an
unordered atomic append. All neuron update, ring, compaction, reset,
refractory, and readout behavior remains the frozen reference implementation.

## Frozen target-local materialization

Before an edge checks refractory state, phase A calls `v2_evolve_target` with
the target, current clock, and `drive[target]`. It is a literal target-local
copy of the frozen serial `evolve_one`: it reads/writes `v:float32`,
`g:float32`, `refractory:int16`, and `last:int64`; computes `d=clock-last`;
returns without mutation for `d<=0`; consumes refractory elapsed time; then
uses the same host-generated float32 `av/ag` table (or the same analytic
fallback), updates `v` before `g`, and finally writes `last=clock`.

Thus a second delayed source arriving at the same clock sees `d==0` and does
not decay a target twice before adding its own weight. There is no cross-target
state mutation in this helper, so source-row target uniqueness is sufficient
for phase-A independence.

## Validation order

V2 is compared independently with both native CPU and `metal-reference-serial`
at each stage. A mismatch stops progression and records source rank, CSR row,
lane, edge ordinal, target, contribution, and before/after target state.
