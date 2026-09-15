# Embodied MaleCNS goalkeeper — initial results

Fixed weights, no learning. 60 episodes per condition, seed 7, shots balanced
across left/center/right groups. Save/goal are logged only; nothing becomes a
reward or modulatory signal. Artifacts:
`workspace/outputs/embodied_flykeeper/suite/`.

## Save rates

| Experiment | Save rate | Left | Center | Right |
| --- | ---: | ---: | ---: | ---: |
| **heuristic** / normal (mechanical ceiling, uses true ball state) | **98.3 %** | 94.4 % | 100 % | 100 % |
| random / normal | 33.3 % | 0 % | 100 % | 0 % |
| **MaleCNS** / normal (closed-loop brain) | **33.3 %** | 0 % | 100 % | 0 % |
| MaleCNS / blind | 33.3 % | 0 % | 100 % | 0 % |
| MaleCNS / mirrored | 33.3 % | 0 % | 100 % | 0 % |
| MaleCNS / static_ball | 35.0 % | 0 % | 100 % | 4.5 % |

Total suite wall time ≈ 26 min; each 60-episode MaleCNS condition integrated
~28 s of neural time (166,700 neurons, 0.1 ms steps) in ~350 s wall on an
Apple M4 (CPU backend).

## Interpretation (honest)

- **The full closed loop works end to end.** A real articulated 3D fly stands
  in the goal, a ball is fired at it, the fly renders its own eye view, the
  unmodified 166,700-neuron MaleCNS processes that image, its descending-neuron
  activity is decoded into a locomotion command, the tripod controller walks the
  MuJoCo body, and the body's motion changes the next eye view. Body–ball
  contact, saves, goals, and episode resets all function.

- **The body and environment are capable.** The heuristic controller — which is
  allowed to use the true ball state — saves 98 % of shots across all groups.
  So the ~33 % floor is *not* a mechanical limitation.

- **The fixed MaleCNS controller performs at chance (33 %), the same as random.**
  Its only saves are center shots, which any stationary body blocks. It does not
  make vision-appropriate lateral saves.

- **Sensory controls confirm the behavior does not depend on the visual input.**
  Blinding the eye, mirroring it, or freezing the ball image all leave the save
  rate unchanged (33–35 %). If MaleCNS were using vision to steer, at least the
  mirrored and blind conditions would differ. They do not.

- **Why:** driven only through the retinal luminance pathway, the fixed
  connectome produces almost no *lateralized* descending-neuron output for this
  stimulus. In a direct calibration, nearly all 1,332 descending neurons stay
  silent; the few that fire (e.g. DNp20, DNpe017) differ by <1 spike between
  left and right shots. Reading a larger 160-vs-158 left/right DNp population
  makes the body move more, but the motion is not correlated with shot side, so
  the save rate is unchanged.

This is the expected outcome of the pre-registered question — *does useful
sensorimotor goalkeeping behavior exist in the fixed connectome before any
learning?* On this task, with this retinal encoding and this decoder, it does
not. That is a result, not a failure of the pipeline: the embodiment,
perception, and motor pathway are all demonstrably functional, which is exactly
what is needed before asking whether plasticity could shape the behavior.

## Per-frame diagnostics

`decisions.csv` records, for every 20 ms decision: sim time, fly x/y/heading and
lateral velocity, ball x/y and velocity, decoded move / lateral command,
left/right descending spike counts, retinal-drive mean, condition, and contact.
These make it straightforward to see that the decoded command is uncorrelated
with the ball's side.
