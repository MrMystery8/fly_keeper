"""Minimal learned visual -> descending-neuron bridge (EXPERIMENTAL).

This package inserts a small, learned transformation across the representational
bottleneck the pathway audit identified (optic-lobe -> visual-projection ->
descending). MaleCNS synaptic weights, LIF dynamics, retinal mapping, R1-R6
encoding, body physics, MuJoCo mechanics, the downstream locomotion controller,
and the CPU scientific-reference backend all remain FIXED. Only the
visual-features -> lateral-command -> DN-current mapping is learned.

Scientific framing (do not overstate): the fixed MaleCNS visual system provides
a useful neural representation; a small learned bridge connects that
representation to real MaleCNS descending neurons controlling the embodied fly.
The native MaleCNS synapses did NOT learn the task.
"""
