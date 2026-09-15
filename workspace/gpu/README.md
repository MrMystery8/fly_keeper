# Experimental Metal backend investigation

CPU remains the default scientific/reference backend. The Metal prototype must upload `outputs/doom/malecns_v1/graph.npz` unchanged once, retain persistent state, and remain explicitly experimental until per-step CPU parity is established. See `CPU_MODEL_SPEC.md` and `REFERENCE_NOTES.md`.

No GPU backend is selectable yet because a race-free, order-equivalent propagation design has not been validated.
