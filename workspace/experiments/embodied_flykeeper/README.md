# Embodied MaleCNS Fly Goalkeeper

The existing MaleCNS whole-CNS simulation (166,700 neurons, unmodified) placed
inside an existing anatomically-detailed 3D fruit-fly body (the **flybody**
MuJoCo model) standing in a football goal. The fly perceives incoming shots
through its own eye camera; that image drives the real retinal pathway into
MaleCNS; descending-neuron activity is decoded into high-level locomotion
commands; an engineered tripod locomotion controller walks the articulated
body in MuJoCo; and the body's motion changes what the fly next sees. The loop
is closed.

```
football arena (MuJoCo)
        │ renders
        ▼
fly eye camera  ──►  R1-R6 retinal sampling (doom.game.retinal_samples, unchanged)
                              │
                              ▼
                     MaleCNS (166,700-neuron LIF, unchanged)
                              │  descending-neuron spikes
                              ▼
                     engineered motor decoder  (L/R asymmetry → strafe)
                              │  high-level command (forward / lateral / turn)
                              ▼
                     tripod locomotion controller  (engineered)
                              │  leg actuators + body drive
                              ▼
                     MuJoCo physics  ──►  body moves ──►  new eye view (loop)
```

## What is biological vs engineered

**Biological (reused, unchanged):** the MaleCNS connectome and neuron
identities; the R1-R6 retinal projection and `retinal_samples` luminance
sampling; the flybody anatomical skeleton, joints, and actuators
(google-deepmind/mujoco_menagerie, Apache-2.0).

**Engineered (our modeling choices):** the football world, the normalized
scale, the tripod locomotion controller, the high-level command set, the
descending-neuron → command decoder (gains, thresholds, activity window), and
the choice of which descending neurons to read. **The fly does not natively
understand football.**

## Commands

Headless MaleCNS closed loop:
```bash
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.embodied_flykeeper.run \
    --controller malecns --episodes 60 --headless
```

Baselines:
```bash
... .run --controller random    --episodes 60 --headless
... .run --controller heuristic  --episodes 60 --headless   # mechanical ceiling (uses true ball state)
```

Sensory controls (behavior should change if it truly depends on vision):
```bash
... .run --controller malecns --episodes 60 --headless --condition blind
... .run --controller malecns --episodes 60 --headless --condition mirrored
... .run --controller malecns --episodes 60 --headless --condition static_ball
```

Full comparison suite (baselines + MaleCNS + controls):
```bash
... -m workspace.experiments.embodied_flykeeper.evaluate --episodes 60
```

Visual demo (mp4 with main view + eye view + retinal panel + readout bars):
```bash
... -m workspace.experiments.embodied_flykeeper.visualize \
    --controller malecns --episodes 5 --out workspace/outputs/embodied_flykeeper/demo.mp4
```

## Simulation timing

| layer | rate |
| --- | --- |
| MaleCNS neural integration | 0.1 ms step (10 kHz), 200 steps per decision |
| MaleCNS decision window / vision / control | 20 ms (50 Hz) |
| MuJoCo physics | 0.1 ms (flybody native timestep) |
| rendering (demo) | 30 fps |

## Scale (normalized, not FIFA-accurate)

flybody uses centimetre units (gravity −981 cm/s²); the fly is ~0.3 cm. The
goal is ±1.6 cm wide, the ball is a 0.45 cm bright rolling sphere, shots start
3.2 cm out at 4.5–6 cm/s. The ball is deliberately large relative to the fly so
it recruits enough R1-R6 receptors to be visible, and the locomotion controller
gives the fly a lateral range (~±0.5 cm per shot) compatible with the goal. This
is an engineered balance for an observable control problem, not a biological
scale claim.

See `RESULTS.md` for measured save rates and the sensory-control analysis.
