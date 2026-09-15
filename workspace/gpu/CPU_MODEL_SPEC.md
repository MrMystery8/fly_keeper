# DoomFly CPU fixed-baseline model specification

Source of truth: `upstream/doomfly/doom/kernel.cpp`, loaded through `doom/native.py`. This document describes it; it does not redefine it.

## Graph and state

- MaleCNS graph: CSR by presynaptic source: `ptr:int64[N+1]`, `post:int32[E]`, `weight:float32[E]`; `N=166700`, `E=25582938`. Every retained edge, including weak and self edges, remains present.
- Per neuron: `v:float32` (initial -52), `g:float32` (0), `refractory:int16` (0), `drive:float32` (0), `previous_drive:float32` (0), `last:int64` (-1), `counts:int32` (0), active flag `uint8`, active index list `int32`.
- Event queue: `queue:int32[19,N]`, `queue_count:int32[19]`. 1.8/0.1 = 18 delay steps; 19 slots prevent a read/write collision.

## Exact step ordering

`dt=0.1 ms`; `av=exp(-dt/20)`, `ag=exp(-dt/5)`. Before each step, changing external drives are settled to `clock-1` using the old current and awaken that neuron. For each active neuron at time `t`, evolve exactly from `last` to `t`: refractory time is consumed first; then `v=-52+(v+52)*a+drive*(1-a)+g*(a-b)/3`, `g*=b`.

After evolution, if `refractory==0 && v>-45`, the source ID is queued at `(t+18)%19` and its count increases. The current queue slot is then propagated in queue insertion order: for every source CSR edge in ascending edge order, target state is materialized at `t`; only non-refractory targets receive `g[target] += weight[edge]`; targets become active. The current queue slot is cleared. Finally, every source scheduled in the future delay slot is reset (`v=-52`, `g=0`, `refractory=22`). The clock advances.

`kernel.cpp` additionally removes only neurons provably unable to fire under their current/decaying state, and materializes every neuron at the observation boundary. This is lazy exact evolution, not graph pruning.

## Input and semantics

`NativeBrain.step` updates R1-R6 luminance with a 10-ms low pass, supplies lamina 12 mV-equivalent drive, retinal `30*luminance/(.02+luminance)`, and optional sugar. No OpenMP directives or CPU thread parallelism exist in `kernel.cpp`; ordering is serial and load-bearing. Any Metal prototype must retain dt, strict threshold, event order, float32 behavior, queue semantics, and drive-change settling. Float atomics reorder additions and can change later spikes; that is a correctness risk, not an acceptable optimization by default.
