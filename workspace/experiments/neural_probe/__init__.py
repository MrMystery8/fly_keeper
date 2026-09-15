"""Neural probing experiment for the embodied MaleCNS goalkeeper.

Presents controlled visual stimuli to the unmodified MaleCNS brain with the fly
body held in a FIXED pose (open loop, no motor feedback), and records sparse
per-neuron spike responses preserving biological body IDs. The goal is purely
diagnostic: determine whether shot direction is represented anywhere in MaleCNS,
and if so where, before any change to the controller or any plasticity.

Nothing here modifies the MaleCNS scientific core, the physics, or the body.
"""
