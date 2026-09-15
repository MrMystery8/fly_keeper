# Metal backend investigation report

## Practical throughput backend (v5, 2026-09-15)

FlyKeeper can now opt into `backend="metal"` / `--backend metal`; CPU remains
the default and unchanged. The selected v5 backend keeps graph, neuron state,
delay ring, active flags, and spike counters in persistent Metal buffers. A
20-ms observation advances 200 internal 0.1-ms ticks in one command buffer and
copies back only requested readouts. Drive upload is one 166,700-element float
vector per observation boundary.

The fast path uses dense parallel neuron evolution, GPU-generated indirect
dispatches for only delayed source rows, and relaxed atomic float accumulation.
Spike resets are deferred to the start of the next internal tick (with a final
boundary reset), which is unobservable inside the batch and removes two kernel
phases per tick. The graph, weights, delay, threshold, refractory period, drive
equations, and fixed-weight model are unchanged. Plasticity remains unsupported.

Measured on this Apple M4 after a realistic propagation warm-up, a sustained
100-repeat run of the 14-frame retinal sequence (280,000 ticks) produced 12,994
steps/s, 1.299x biological realtime, versus CPU 13,056 steps/s / 1.306x (0.995x
kernel-level speedup: effectively tied). A 100-episode closed-loop FlyKeeper run completed in
26.16 s versus CPU 48.43 s, a 1.85x end-to-end speedup and 1.07x biological
realtime including rendering, retinal sampling, Python control, and readout.
Persistent Metal allocation is approximately 225.8 MB, excluding small driver
objects; the Python adapter currently also retains the NativeBrain graph arrays
for IDs, retinal geometry, and CPU-compatible metadata. `/usr/bin/time -l` on
a one-episode process measured 654.0 MB maximum RSS / 659.2 MB peak footprint
for Metal, versus 404.1 MB / 375.5 MB for CPU.

V5 is compiled with fast math because its atomic propagation already relaxes
exact arithmetic; all exact/reference backends retain `-fno-fast-math`.

This speed is not exact parity. On the 14-frame open-loop fixture, 9/14 DNp20
readout frames were exact, maximum left/right spike-count difference was 1,
maximum readout voltage difference was 6.21 mV, and 4/14 decoded actions
differed. In 100 closed-loop seed-7 episodes, CPU saved 26 and Metal saved 25;
9 episode results and 728/1,400 actions differed. CPU actions were LEFT/STAY/
RIGHT = 134/694/572; Metal = 7/859/534. The large trajectory divergence from
small readout differences is expected from closed-loop feedback, but the
left-action collapse is behaviorally material. Use CPU for scientific parity;
use Metal only for throughput experiments that explicitly accept this measured
model variant.

The exact batched v4 experiment proved that command-buffer overhead was not the
v2 bottleneck: blank batches reached 38,371 steps/s, but realistic activity
fell to about 92 steps/s because stable active/source ordering remained serial
inside one threadgroup. The sparse v6 target-deduplication experiment reached
14,052 steps/s (1.405x realtime) on one sequence but added enough kernel phases
to offer only ~3% over CPU and did not improve DNp20/action differences. These
implementations remain isolated rather than replacing the selected v5 path.

## Status

**CPU REFERENCE + OPTIONAL RELAXED METAL.** The CPU backend remains default and
scientifically authoritative. `metal-reference-serial` is a test-only frozen
oracle; the selectable `metal` backend is the explicitly relaxed v5 path
described above.

## Frozen bridge current-toolchain identity (2026-09-15)

The historical serial bridge binary SHA-256 is retained as
`c5fb5956040981da5b4145aa1fa8151cd9156dcd8ecfcac3c8742397df7d63bc`.
With unchanged frozen source and `build.py`, the current Apple toolchain
reproducibly emits `f3bc6ca88852029bb2fcdc89fc3e972a9a98cee8c215be1e75b943683e675ba4`.
This is accepted as a toolchain-specific Mach-O identity only after behavioral
parity checks; it is not a scientific-model or serial-source change. V2/V3
builds are isolated in `build_accelerators.py`, leaving `build.py` frozen.

## Deterministic-v2 continuation (2026-09-15)

`metal/MaleCNS_v2.metal`, `metal/bridge_v2.mm`, and `metal_v2.py` now form a
separate persistent v2 implementation.  The frozen serial shader and bridge
are untouched.  V2 owns persistent CSR/state/ring/active-list buffers and
uses `-fno-fast-math`.

The initial complete-step implementation retains serial reference ordering for
active-list iteration, delayed-source rank, reset/refractory, and ring handling.
Within each dynamically ordered source, source-row target uniqueness permits a
parallel strided phase-A update; lane zero performs the required CSR-order
stable awakening append.  Independent drive-settle and observation-boundary
materialization passes were subsequently parallelized, with serial stable
append/reset passes preserved.  `metal_step_v2_batch` also retains logical
0.1-ms semantics while placing 1/10/50/100 ticks in one command buffer.

Automated `python -m workspace.gpu.validate_v2 --through full-100` (using the
project virtualenv) passes: targeted long-row/convergent/refractory synthetic
coverage for 1,000 steps and full MaleCNS CPU = frozen-serial = v2 full-state
comparisons at 1, 10, and 100 steps.  State comparison includes v, g,
refractory, event ring/counts, active list/flags, spike counts, last clock and
simulation clock.

Measured blank-workload benchmark (M4, 200 steps; no state readback in timed
loop): CPU native is 282.1 µs/step (0.355× realtime).  Frozen serial Metal is
76.9 ms/step (0.00130×).  V2 before the independent-pass optimization was
77.1 ms/step at width 64.  After it, v2 width 64 is 10.84 ms/step in a
50-tick batch (0.00922× realtime); widths 32/64/128/256 before that change
were all approximately 77 ms/step, so row width was not the bottleneck.

Recorded FlyKeeper retinal CPU workload statistics: delayed sources per step
mean 29.34, median 23, P95 63, maximum 6,948; delivered-source out-degree
mean 96.65, median 43, P95 483, maximum 9,636. Full graph out-degree mean is
153.47, median 112, P95 404, maximum 11,203. V2 unified-memory allocation is
approximately 250.6 MB (CSR, state, 19-slot ring, flags and awakening bytes).

Conclusion: v2 is exact through the completed ladder but is not usable and is
not integrated into FlyKeeper. Its hard bottleneck is still per-tick serial
work plus source-by-source barriers; batching removes little after 10 ticks.
The next meaningful implementation is deterministic-v3: parallel expansion
across dynamic source ranks, rank-preserving target segmented reductions, and
stable parallel active-list compaction. Static target-major order alone cannot
be used because it loses delayed-source rank. CPU remains the recommended and
only selectable FlyKeeper backend. No thermal conclusion is claimed because a
useful sustained accelerated workload does not yet exist.

## Preflight result

Apple M4 reports Metal 4 support. Xcode's MetalToolchain component was installed after the initial preflight. `xcrun metal -v` reports Apple metal 32023.883 / air64, and a Swift `MTLCreateSystemDefaultDevice` probe reports `Apple M4`.

`workspace/gpu/metal/MaleCNS.metal` compiled to `build/MaleCNS.metallib` (SHA-256 `5db08e6d509a00f11b4e000bf5ee1815fb5ce565023e77fa7f57e3d95dad9d91`). It retains the limited `malecns_update` building block and now adds a deliberately serial complete-step reference kernel; neither is user-selectable.

## Architecture recommendation

Persistent graph buffers should contain the unmodified CSR arrays once. State buffers require v, g, refractory, prior drive, active flags/list, queue [19,N], queue counts, and observation outputs. A faithful port must preserve CPU source/queue iteration and float accumulation. SiliconFly-style compact-spiker dispatch is promising only with an order-equivalent accumulation strategy; float atomic scatter is not currently accepted because it changes source addition order.

## Persistent bridge

`metal/bridge.mm` compiles to `build/libmalecns_metal_bridge.dylib` (arm64). `metal_create_deterministic_v1` successfully instantiates an M4 persistent context from the complete verified CSR graph and `metal_destroy` releases it. The context owns MTLDevice, MTLCommandQueue, compiled library/pipelines, copied CSR ptr/post/weight buffers, host-generated native-equivalent analytic tables, and persistent v/g/refractory/last/drive/active/flag/spike buffers plus `event_sources[19,N]` and `event_counts[19]`.

The bridge now exposes test-only state load/copy, stable-compaction, and `metal_step_reference_serial`. The step dispatch uses one GPU thread for a whole 0.1-ms timestep, preserving the CPU's ascending drive scan, active-list traversal, stable spike append, dynamic delayed-event rank, source CSR order, float32 accumulation, reset/refractory sequence, and observation-boundary materialization. It is intentionally too slow to expose to FlyKeeper.

## Deterministic execution index

`build_target_major.py` now generates `build/target_major.npz` directly from the verified CSR. It contains `target_offsets:int64[N+1]`, `incoming_source:int32[E]`, `incoming_weight:float32[E]`, and `incoming_original_ordinal:int32[E]`. A stable sort by target preserves original CSR ordinal within each target; it does not merge, prune, or alter graph data. The unit gate covering convergent float32 inputs passes.

Important limitation discovered before GPU use: this static ordering is **not yet proven CPU-equivalent**. CPU propagation order is the dynamic delayed-source queue insertion order, derived from active-list spike order, rather than globally sorted CSR source order. Multiple simultaneous sources may therefore contribute to one target in an order a static target-major view cannot reconstruct. The index is retained as an investigation artifact, but deterministic-v1 must either carry dynamic source-event ranks and deterministically order each target’s active incoming contributions by that rank, or use a conservative serial source-order propagation stage. No target-major GPU propagation has been implemented or claimed correct.

## Source-row uniqueness audit

`SOURCE_ROW_DUPLICATES.json` reports 166,700 source rows, **0 rows with duplicate target IDs**, 0 duplicate source-target pairs, and maximum pair multiplicity 1. Therefore a deterministic source-serial GPU design may parallelize the edges within one source CSR row without same-source target write races. It must still serialize source ranks from the delayed queue because different sources may converge on the same target.

## Current parity results

`tests/test_metal_compaction.py` validates the actual GPU serial-compaction
kernel byte-for-byte for empty, unsorted, alternating, and 199 deterministic
pseudo-random active lists. It does not use atomic append.

`tests/test_metal_step.py` compares full state after every step against
`doom.native.neural_advance(..., steps=1)`: one-edge delay, dynamically
ordered convergent sources, refractory/ring wraparound, and a 10-neuron
randomized graph for 1,000 timesteps pass. Compared state includes `v`, `g`,
refractory, last-materialization clock, active list/flags, spike counts, and
all 19 event-ring counts/source arrays. The first full-MaleCNS blank,
deterministic check passed exact state/spike parity for 1, 10, and 100 steps
(100 steps: 7.592 s including repeated full debug snapshots on this M4).

## Realistic retinal parity

`run_fixture_parity.py` consumes only
`fixtures/flykeeper_retinal_sequence.npz`; it does not instantiate a renderer
or FlyKeeper environment. It applies the audited fixed-baseline retinal/lamina
drive transform once per recorded 20-ms frame, then advances both native CPU
and serial Metal in lockstep at 0.1 ms. The diagnostic compares full state on
every tick, including active ordering and the full 19-slot event source arrays.

Initial Metal fast-math contraction produced the first float difference at
tick 78 (`v[55485]`: CPU `-49.034706115722656`, Metal
`-49.03470993041992`). Building the shader with `-fno-fast-math` removed that
discrepancy. The FlyKeeper fixture now has exact state and spike parity through
all 2,800 ticks (14 frames; repeated delayed-event delivery). Repeated blank,
left-side, center, and right-side luminance fields each pass 1,000 ticks. No FlyKeeper
action/episode parity claim is made yet.

`fixtures/asymmetric_retinal_fixtures.npz` now provides deterministic blank,
left, center, right, strong-left, and strong-right retinal fields derived only
from the genuine receptor UV layout. They are regression inputs, not privileged
environment state.

## Reference parity completion

The serial reference is frozen for the fixed-weight workloads listed in
`fixtures/reference_manifest.json`; `DETERMINISTIC_V2_AUTHORIZED = true`.
Fast-math must remain disabled: fused contraction previously caused a one-ULP
divergence at tick 78, eliminated by `-fno-fast-math`.

| Test | Duration | Result |
|---|---:|---|
| Synthetic randomized | 1,000 steps | Exact |
| Blank, left, center, right | 1,000 ticks each | Exact |
| Strong-left/right | 1,000 ticks each | Exact |
| Mirrored left/right/strong-left/strong-right | 1,000 ticks each | Exact |
| Recorded FlyKeeper retinal fixture | 2,800 ticks | Exact |
| FlyKeeper closed loop | 10 episodes / 140 decisions | Exact |
| FlyKeeper closed loop | 100 episodes / 1,400 decisions | Exact |

This is not a claim of equivalence outside the tested fixed-weight model,
stimulus, and control workloads.

## Serial reference identity

`METAL_REFERENCE_HASHES.md` freezes source, bridge, wrapper, build, and binary
hashes for this oracle. Any future deterministic-v2 or fast path must compare
against this reference as well as the native CPU implementation.

## Correctness and performance

No performance claim is made: the reference serial dispatch is an oracle, not
an accelerator. CPU baseline remains approximately 0.5× realtime. Required
next parity scenarios are a single retinal stimulus, small retinal pattern,
and the deterministic FlyKeeper fixture, comparing spikes, v, g, refractory,
DNp20 readouts, and first divergence before any parallel implementation.

CPU delay harness: `workspace/gpu/tests/test_delay_semantics.py` passes, establishing one source event queued at t=0 is delivered at t=18 (not t=17 or t=19), while the source reset/refractory assignment occurs at t=0. `debug_divergence.py` is ready to report the first state mismatch once bridge snapshots are available.
