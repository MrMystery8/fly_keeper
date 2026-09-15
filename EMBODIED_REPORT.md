# Embodied MaleCNS Fly Goalkeeper — final report

Prepared on the `embodied-flykeeper` branch. This puts the existing, unmodified
MaleCNS whole-CNS simulation (166,700 neurons) into an existing anatomically
detailed 3D fruit-fly body inside a MuJoCo football goal, and runs a genuine
closed sensorimotor loop: the fly sees the incoming ball through its own eye,
the connectome processes that image, descending-neuron activity drives its body,
the body physically moves, and the movement changes what the fly next sees.

The existing MaleCNS simulator, CPU reference, retinal pathway, FlyKeeper, and
Metal work were all preserved untouched. Embodiment was added as a new
subsystem (`workspace/embodiment/`) plus a new experiment
(`workspace/experiments/embodied_flykeeper/`). No GPU/Metal optimization was
resumed.

---

## External stack selected

- **Physics:** [MuJoCo](https://github.com/google-deepmind/mujoco) 3.13.0
  (`pip install mujoco`), installed into the existing neural venv without
  disturbing the pinned scientific stack (numpy stayed 1.24.4; `pip check`
  clean).
- **Body:** the **flybody** MJCF model from
  [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody),
  the anatomically detailed *Drosophila melanogaster* developed by Google
  DeepMind and HHMI Janelia (Vaxenburg et al., 2024). Sparse-cloned to
  `workspace/vendor/mujoco_menagerie_flybody/` (git-ignored, ~137 MB).

**Why this over FlyGym/NeuroMechFly:** FlyGym 2.x rewrote its API in March 2026
and the legacy interface moved to `flygym-gymnasium`; both pull a heavier
dependency chain. The flybody MJCF is the same authoritative articulated body
that FlyGym and DeepMind's own work build on, but as a self-contained model I
can drive directly with `mujoco`. This gave full, stable control of the closed
loop on Python 3.11 with minimal dependencies. FlyGym's locomotion controllers
were used as a *conceptual* reference (tripod gait), not imported.
`artem-x-meta/fly-arena` (MaleCNS + FlyGym) was noted as prior art for the same
overall idea but not used.

## Licensing

All reused external components are Apache-2.0 and compatible:

| Component | License | Use |
| --- | --- | --- |
| MuJoCo (`mujoco` 3.13.0) | Apache-2.0 | physics + offscreen rendering |
| flybody MJCF + meshes (mujoco_menagerie) | Apache-2.0 | the 3D fly body (LICENSE vendored) |
| imageio / imageio-ffmpeg | BSD-2 / Apache-2.0 | mp4 demo encoding |

No code was copied from FlyGym/NeuroMechFly; its tripod-gait idea was
reproduced independently. The MaleCNS data and DoomFly core are unchanged and
governed by their existing terms.

## Architecture

```
football arena (MuJoCo)
        │ renders eye_left camera (fovy 140°, on the fly head)
        ▼
doom.game.retinal_samples(rgb, uv)        ← UNCHANGED R1-R6 sampling, 3,335 receptors
        ▼
MaleCNSBrain.stimulate_retinal_luminance  ← UNCHANGED adapter
        ▼
MaleCNS 166,700-neuron LIF (0.1 ms)       ← UNCHANGED brain, 20 ms decision window
        ▼
DescendingMotorDecoder                    ← ENGINEERED: L/R DN spike asymmetry → strafe
        ▼
FlyBody.set_command(forward, lateral, turn, gait_on)
        ▼
tripod locomotion controller             ← ENGINEERED: in-place tripod + body-frame PD drive
        ▼
MuJoCo physics → body moves → new eye view (loop closes)
```

Only pixels reach the brain. Ball position/velocity/target are never given to
MaleCNS; the environment uses them only for physics and scoring.

## Body

The flybody model: 68 bodies, 103 joints, 108 DOFs, **78 actuators**, 9
cameras. A 6-DOF free joint floats the thorax; each of the six legs
(T1/T2/T3 × left/right) has coxa/femur/tibia/tarsus joints plus a tarsal
adhesion actuator; the head carries `eye_left`/`eye_right` cameras.

Because 78 actuators is far too many for the brain to drive directly (as the
brief anticipated), locomotion is **hierarchical**. A hand-written tripod CPG
animates the legs (diagonal tripods alternately lift/plant, adhesion released
during swing) for ground contact and visual realism, and a body-frame PD
**velocity controller** on the thorax realizes the high-level command
(`forward`, `lateral`, `turn`). Calibrated on the model, this gives a stable
~±0.5 cm lateral reach per shot and ~1.3 rad/s turning with no vertical
instability. The goalkeeper primarily **strafes** left/right along the goal line
while facing the incoming ball, which also keeps the ball in view.

## Neural outputs used

- **Default decoder:** the two DNp20 descending neurons, body IDs **10162
  (left)** and **10059 (right)** — the same anatomically identified cells the
  existing 2D FlyKeeper used, tracing to the DoomFly experimental BCI mapping.
- **Optional:** a pooled population of all posterior descending neurons by soma
  side (**160 left / 158 right** DNp cells present in the graph). Available via
  the `decoder_ids` argument.

**Biological** = the neuron identities and connectome. **Engineered** = the map
from left/right spike-count asymmetry to a lateral strafe command, the 20 ms
activity window, the normalization, smoothing, and gains.

## Vision

Each decision, the fly's `eye_left` camera (140° FOV, head-mounted) is rendered
to a 160×96 RGB frame. That frame is sampled by the **unchanged**
`retinal_samples(rgb, uv)` at the 3,335 mapped R1-R6 receptor UV coordinates
(bilinear luminance with sRGB linearization) and delivered as the existing
saturating retinal drive. The ball is a bright, large sphere so it recruits
enough receptors to be visible; its image position tracks its lateral position
monotonically (verified: left/center/right shots land the ball at image columns
~99/123/157).

## Simulation timing

| Layer | Rate |
| --- | --- |
| MaleCNS neural integration | 0.1 ms step (10 kHz); 200 steps per decision |
| Decision window / vision / control | 20 ms (50 Hz) |
| MuJoCo physics | 0.1 ms (flybody native timestep) |
| Rendering (demo only) | 30 fps |

On an Apple M4 (CPU backend), a 60-episode MaleCNS condition integrates ~28 s of
neural time in ~350 s wall. Physics alone runs ~11,000 steps/s.

## Scale (normalized, documented)

flybody uses centimetre units (gravity −981 cm/s²); the fly is ~0.3 cm. The goal
is ±1.6 cm wide × 1.4 cm high, the ball is a 0.45 cm bright rolling sphere, shots
start 3.2 cm out at 4.5–6 cm/s aimed left/center/right (±0.55 cm). This is a
deliberately normalized toy scale chosen so the ball is visible to the eye and
reachable by the body — **not** a physically FIFA-accurate scale.

## Commands

```bash
VENV=upstream/doomfly/.venv-neural/bin/python
export PYTHONPATH=workspace:upstream/doomfly

# Visual demo (mp4: main view + eye view + retinal panel + readout bars)
$VENV -m experiments.embodied_flykeeper.visualize --controller malecns --episodes 5 \
    --out workspace/outputs/embodied_flykeeper/demo.mp4

# Headless MaleCNS closed loop
$VENV -m experiments.embodied_flykeeper.run --controller malecns --episodes 60 --headless

# Baselines
$VENV -m experiments.embodied_flykeeper.run --controller random    --episodes 60 --headless
$VENV -m experiments.embodied_flykeeper.run --controller heuristic  --episodes 60 --headless

# Sensory controls
$VENV -m experiments.embodied_flykeeper.run --controller malecns --episodes 60 --headless --condition blind
$VENV -m experiments.embodied_flykeeper.run --controller malecns --episodes 60 --headless --condition mirrored
$VENV -m experiments.embodied_flykeeper.run --controller malecns --episodes 60 --headless --condition static_ball

# Full comparison suite (all of the above, aggregated)
$VENV -m experiments.embodied_flykeeper.evaluate --episodes 60
```

## Results (60 episodes/condition, seed 7)

| Experiment | Save rate | Left | Center | Right |
| --- | ---: | ---: | ---: | ---: |
| heuristic / normal (mechanical ceiling; uses true ball state) | **98.3 %** | 94.4 % | 100 % | 100 % |
| random / normal | 33.3 % | 0 % | 100 % | 0 % |
| **MaleCNS / normal** (closed-loop brain) | **33.3 %** | 0 % | 100 % | 0 % |

The full pipeline runs end to end: a real articulated 3D fly stands in goal, a
ball is fired, the fly perceives it through its eye, the 166,700-neuron MaleCNS
processes the image, descending activity drives the body, the body physically
moves and can contact the ball, saves/goals are detected, and episodes reset —
headlessly over many episodes and in the mp4 demo.

The **heuristic ceiling (98 %)** shows the body and environment are fully capable
of saving across all shot groups. The **fixed MaleCNS controller performs at
chance (33 %), identical to random**: its only saves are center shots that any
stationary body blocks.

## Controls

| MaleCNS condition | Save rate |
| --- | ---: |
| normal | 33.3 % |
| blind (eye zeroed) | 33.3 % |
| mirrored (eye flipped) | 33.3 % |
| static_ball (frozen image) | 35.0 % |

The save rate is unchanged whether the fly sees normally, is blinded, sees a
mirror image, or sees a frozen scene. **The behavior does not depend on the
visual input.** Driven only through the retinal pathway, the fixed connectome
produces almost no *lateralized* descending output for this stimulus: in direct
calibration, nearly all 1,332 descending neurons stay silent, and the few that
fire (DNp20, DNpe017) differ by <1 spike between left and right shots. Pooling
160/158 left/right DNp cells makes the body move more, but the motion is
uncorrelated with the ball's side, so the save rate does not improve.

This is the honest answer to the pre-registered question — *does useful
sensorimotor goalkeeping exist in the fixed connectome before any learning?* On
this task, with this retinal encoding and this decoder, **it does not**. That is
a result, not a pipeline failure: perception, the 166,700-neuron brain, the
motor decode, and the physical body are all demonstrably functional and
closed-loop.

## Limitations — engineered vs biologically grounded

**Biologically grounded (reused data):** the MaleCNS connectome and neuron
identities; the R1-R6 retinal projection and luminance sampling; the flybody
anatomical skeleton, joints, and actuators.

**Engineered (modeling choices, no biological claim):** the football world and
its normalized scale; the tripod CPG + body-frame velocity drive; the
high-level command set; the descending-neuron→strafe decoder (gains,
thresholds, 20 ms window, smoothing); the choice of readout neurons; making the
ball large/bright for visibility; the save/goal rules. The fly does not natively
understand football, and the locomotion controller is a stand-in for the VNC
gait circuitry, not a simulation of it. The scale is normalized, not
scale-accurate. LIF dynamics, neurotransmitter signs, delays, and the sensory
encoding remain the same modeling assumptions documented in `SETUP_REPORT.md`.

## Recommended next step

Because the fixed connectome shows **no** vision-dependent goalkeeping (chance
performance, flat across all sensory controls) while the body is provably
capable (98 % heuristic ceiling), the highest-value next step is to test
**learning/plasticity**: keep this exact embodied loop, add SAVE→reward /
GOAL→negative modulation, and let a bounded plasticity rule shape the
visual→descending mapping, checking whether the save rate rises above chance and
becomes sensory-dependent (i.e. collapses again under the blind/mirrored
controls). Secondary options, in order: (1) a better sensory→descending mapping
(explicitly identify and drive optic-glomerulus DNs that are lateralized for a
looming/translating target); (2) richer motor decoding (forward interception in
addition to strafing); (3) only later, direct VNC/muscle control. Environment
and rendering are already sufficient and should not be the focus.

---

### Files

- `workspace/embodiment/{fly_body,mujoco_world,vision_bridge,motor_decoder}.py`
- `workspace/experiments/embodied_flykeeper/{controllers,run,evaluate,visualize}.py`, `README.md`, `RESULTS.md`
- Artifacts: `workspace/outputs/embodied_flykeeper/suite/` and per-run folders
- Vendored body (git-ignored): `workspace/vendor/mujoco_menagerie_flybody/flybody/`
