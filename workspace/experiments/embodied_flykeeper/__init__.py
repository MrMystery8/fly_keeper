"""Embodied MaleCNS fly goalkeeper experiment.

A closed-loop system: the fly perceives the arena through its own eye camera,
the unmodified 166,700-neuron MaleCNS brain processes that image, its
descending-neuron activity is decoded into high-level locomotion commands, an
engineered tripod-CPG walks the articulated flybody in MuJoCo, and the body's
movement changes what the fly next sees. See the module docstrings in
workspace/embodiment/ for the biological-vs-engineered boundary.
"""
