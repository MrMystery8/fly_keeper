"""Embodiment subsystem for the MaleCNS fly goalkeeper.

This package places the *unmodified* MaleCNS brain (via workspace.adapters.brain)
into an *existing* articulated 3D fruit-fly body (the flybody MJCF model from
google-deepmind/mujoco_menagerie, Apache-2.0) inside a MuJoCo football-goal world.

Modules
-------
fly_body.py      : loads the flybody model, exposes a tripod-CPG locomotion
                   controller driven by high-level commands (turn/forward/stop).
mujoco_world.py  : builds the goalkeeper arena (goal, ball, shot logic) around
                   the fly and detects saves/goals.
vision_bridge.py : renders a body-relative eye view and maps it, through the
                   *existing* R1-R6 retinal sampling, into MaleCNS drive.
motor_decoder.py : engineered decoder from MaleCNS descending-neuron activity
                   to high-level locomotion commands.

BIOLOGICAL vs ENGINEERED
------------------------
Biological (data, reused unchanged): the MaleCNS connectome, neuron identities,
the R1-R6 retinal projection, and the flybody anatomical skeleton/actuators.
Engineered (our modeling choices): the tripod CPG gait, the command set, the
descending-neuron -> command decoder, thresholds/gains, the football world, and
the chosen simulation scale. The fly does not natively understand football.
"""
