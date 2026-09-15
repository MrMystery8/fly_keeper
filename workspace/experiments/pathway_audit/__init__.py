"""Pathway activation audit.

Determines whether the FIXED MaleCNS network (no synaptic weight changes) can
transmit visual left/right shot direction from the optic lobe, through
visual-projection / intermediate populations, to the descending neurons, and
pinpoints the exact stage where the signal fails.

Read-only with respect to the MaleCNS scientific core, the embodied environment,
and the frozen baseline. All analysis lives here; artifacts under
workspace/outputs/pathway_audit/.
"""
