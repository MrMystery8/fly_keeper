# SiliconFly reference notes

Inspected `dawsonamf/siliconfly` commit `8839d84cd24888a4251a2e227792b6f26fbee776`; code is MIT (copyright Denis Shiryaev and Dawson Metzger-Fleetwood). Its FlyWire data is CC BY-NC 4.0 and is not used.

Transferable engineering ideas: persistent shared CSR/state buffers, one update dispatch per neuron, spike-list compaction, a propagation dispatch over only spiking sources, command-buffer batching, and explicit CPU/GPU verification hooks. Its Int32 fixed-point atomic accumulation is particularly useful as a deterministic alternative to float atomics.

Not transferable: its FlyWire graph, rest drives, noise, inhibitory ring, 1-ms timestep, equations, roles, weight scaling, or behavior. DoomFly requires a 0.1-ms 18-step queue, lazy state materialization, signed float32 conductance accumulation in source/queue order, and reset timing different from SiliconFly. A direct dense/faithful Metal prototype must be verified first; sparse propagation is only eligible after proving queue ordering and accumulation equivalence.
