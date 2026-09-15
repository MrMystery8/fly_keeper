# CPU timestep semantics (directly from `doom/kernel.cpp`)

All arrays mutate in place and all floating operations are `float`/float32. CSR is source-major: `ptr:int64[N+1]`, `post:int32[E]`, `weight:float32[E]`; outgoing edges are traversed in increasing edge index for sources in the queue's insertion order.

1. Before the timestep, loop `i=0..N-1`: if `drive[i] != previous_drive[i]`, lazily `evolve(i, clock-1, previous_drive[i])`, set `previous_drive`, and awaken it. This is how external retinal/lamina currents enter; CPU `NativeBrain` has already formed the exact drive array.
2. For current `slot=clock%19`, future `future=(clock+18)%19`, loop the active list in its stored order. `evolve` consumes elapsed refractory first, then analytically updates `v` and `g`; it writes `last`, `refractory`, `v`, `g` in place.
3. If `refractory==0 && v>-45` (strict comparison), append the source ID to `queue[future]` and increment `counts`.
4. Compact the active list only after each active source is examined.
5. Consume `queue[slot]` in append order. For each queued source, visit `e=ptr[source]..ptr[source+1]-1`; materialize target at current clock; only if target is not refractory, add `weight[e]` to `g[target]` in source/edge order and awaken target.
6. Zero `queue_count[slot]`. Then loop sources already in `queue[future]`—including a spike just appended this step—and set `v=-52`, `g=0`, `refractory=22`. Queue entries remain for delivery at `future` 18 ticks later.
7. Increment clock. After the requested step count, materialize all neurons at `clock-1` for observable state.

Queue: `int32[19,N]` with `int32[19]` lengths. Delay is 18 integration steps (1.8 ms at 0.1 ms); 19 slots avoid a same-slot read/write collision. It is a source-event queue, not per-edge delayed values. No OpenMP/thread behavior exists.

**Ordering consequence:** the queue's source sequence is dynamically appended while the active list is traversed. It is not necessarily globally ascending source ID or CSR ordinal. A target-major GPU reduction must preserve this dynamic source-event rank for each delivered source, not merely sort incoming edges by static CSR ordinal.

## Active-list ordering

`active:int32[N]`, `flags:uint8[N]`, and `nactive:int32[1]` are persistent. Initialization appends `unique(retina, lamina, sugar)` in NumPy sorted order. `awaken(i)` appends only when `flags[i]==0`; therefore no duplicates occur. At each timestep, the first `original=nactive` entries are traversed in their existing order, then surviving entries are compacted stably into `active[0:kept]`. Newly awakened targets are appended after those originals while queue propagation proceeds. Refractory neurons can remain active if the conservative firing bound says they may later fire; they leave only when that bound is false. A drive change scans neuron indices `0..N-1`, awakening in ascending internal-index order before active-list traversal. Thus simultaneous spikes are appended to a future queue in stable active-list order, not source-ID order.

## Metal reference dispatch boundary

`metal-reference-serial` executes exactly one logical native timestep per
dispatch. A native `neural_advance(..., steps=1, ...)` call materializes every
neuron at the observation boundary (`clock - 1` after incrementing). The serial
Metal reference does the same before returning a snapshot, so every 0.1-ms
step has directly comparable complete state. It deliberately does not batch.
