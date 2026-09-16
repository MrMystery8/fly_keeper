"""Arcade version of the fruit-fly goalkeeper demo.

This is a deliberately non-frozen, gameplay-oriented build that lives ALONGSIDE
the validated scientific demo (workspace/experiments/interactive_demo/), which it
never touches. Differences from the frozen system, all intentional:

  * a livelier locomotion layer: faster/stronger ground strafing plus a NEW
    vertical hop/lunge/flight mechanic that does not exist in the frozen body;
  * a retrained 2-output bridge (lateral + vertical) that drives both a lateral
    and a vertical descending-neuron group, TRAINED AND RUN on the Metal backend;
  * corrected save/goal scoring (a deflection that still crosses the line inside
    the posts is a GOAL, not a SAVE);
  * lofted/high shots and vertical player aim.

The full fixed 166,700-neuron MaleCNS still runs in the loop (vision -> brain ->
real descending neurons -> body). The Metal backend is fast (~1.6x realtime) but
not bit-exact with the CPU reference, so the arcade bridge is trained on Metal to
match its runtime. The frozen scientific numbers and the interactive_demo package
are unchanged.
"""
