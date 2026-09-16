"""Interactive 3D penalty-game demo for the frozen learned-bridge goalkeeper.

This package is a GAME/UI layer ONLY. It imports and reuses the frozen
scientific stack (adapters.brain, embodiment.*, experiments.learned_bridge.*,
experiments.embodied_flykeeper.controllers) without modifying any scientific
parameter. See INTERACTIVE_DEMO_REPORT.md.

Scientific invariant (Learned Bridge mode):

    human penalty -> rendered 3D scene -> fly eye camera -> R1-R6 retina
      -> full fixed MaleCNS -> 207 frozen optic-lobe features
      -> frozen 829-parameter bridge -> real MaleCNS descending neurons
      -> existing decoder / CPG -> articulated MuJoCo body -> SAVE / GOAL

The learned controller NEVER receives ball position, velocity, aim, power, or
any privileged simulator state. Only rendered pixels enter the brain.
"""
