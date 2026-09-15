"""Targeted reward-modulated plasticity on the visual -> projection -> descending
pathway identified by the pathway-activation audit.

Scope is strictly limited by an explicit synapse mask (build_mask.py). All other
MaleCNS synapses stay frozen. The fixed-weight CPU reference remains the
scientific ground truth; nothing here mutates the connectome file on disk.

The learning rule is EXPERIMENTAL and is not a validated model of fruit-fly
synaptic plasticity.
"""
