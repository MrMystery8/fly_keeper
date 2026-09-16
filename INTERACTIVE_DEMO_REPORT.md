# Interactive 3D penalty-game demo — report

**What this is.** A playable 3D penalty game in which a human takes penalties
against a fruit-fly goalkeeper that is physically embodied in MuJoCo. The main
mode, **Learned Bridge Fly**, produces every save/goal through the *exact frozen
scientific chain* validated in `LEARNED_BRIDGE_REPORT.md` (tag
`learned-visual-dn-bridge-complete`) — no retraining, no tuning, and no
privileged ball information reaching the neural controller.

> **Scientific status (unchanged from the experiment).** A fixed MaleCNS
> connectome processes the fly's visual input. A small learned bridge (829
> trainable parameters) maps informative visual activity to real descending
> motor neurons controlling the embodied fly. The native MaleCNS synaptic
> weights remain fixed — the natural fly connectome did **not** naturally know
> how to play football; the bridge is an engineered learned transformation
> across an experimentally identified visual-to-motor bottleneck.

Branch `flykeeper-interactive`, forked from tag
`learned-visual-dn-bridge-complete`. The scientific branch and tag are
untouched. CPU scientific-reference backend only.

---

## Architecture — how the game connects to the frozen scientific system

The game is a thin UI/orchestration layer over the existing validated code. It
adds **no** scientific parameters and reuses the frozen closed loop verbatim.

Learned Bridge runtime path (preserved exactly):

```
human penalty (aim, power)
  → GoalkeeperWorld fires the ball (MuJoCo physics)
  → rendered 3D scene
  → fly eye_left camera  → R1–R6 retinal sampling            [fixed]
  → full fixed MaleCNS visual processing                     [fixed]
  → 207 frozen selected optic-lobe features (4 × 20 ms)      [frozen]
  → frozen 829-parameter linear bridge → tanh → scalar u     [frozen]
  → fixed DN opponent basis B_lr → bounded additive current  [fixed]
  → 4 real MaleCNS descending neurons (native LIF dynamics)  [fixed]
  → existing DescendingMotorDecoder → tripod-CPG locomotion  [fixed]
  → articulated MuJoCo fly body
  → physical collision → SAVE / GOAL
```

New code (game layer only), under
`workspace/experiments/interactive_demo/`:

| File | Role |
| --- | --- |
| `artifact_integrity.py` | Verify frozen artifact checksums + metadata; assemble a versioned `ModelArtifact`. |
| `modes.py` | `GoalkeeperEngine` — one controller mode + vision condition; steps the frozen loop one 20 ms decision at a time; `assert_no_privileged_leak`. |
| `shots.py` | Human aim/power → clamped `ShotSpec`; deterministic science-demo shot sets. |
| `session.py` | `SimSession` — runs the engine + all MuJoCo rendering in one worker thread; publishes thread-safe snapshots for the UI. |
| `game.py` | pygame front-end: player controls, mode selector, scoring, panels, insets. |
| `regression_learned_bridge.py` | Frozen learned-bridge output regression. |
| `regression_no_privileged_leak.py` | Static + behavioural no-leak regression. |
| `run_regressions.py` | Runs the whole regression suite. |

The game reuses, unmodified: `GoalkeeperWorld`, `FlyBody`/tripod-CPG,
`VisionBridge` + `ShuffleVisionBridge`, `MaleCNSBrain`, `DescendingMotorDecoder`,
`DNMotorBasis`, `LearnedBridge`/`LinearBridgeModel`/`VisualFeatureExtractor`,
`BridgeController`, and the existing `Random`/`Heuristic` controllers.

---

## Controller modes

Selectable before or during a match (keys `1`–`5`, or click):

| # | Mode | What it is |
| --- | --- | --- |
| 1 | **Natural Fly** | Full fixed MaleCNS, no learned bridge (= bridge OFF). Reproduces the validated Natural-MaleCNS behaviour (mostly passive; saves centre by standing). |
| 2 | **Learned Bridge Fly** | The main mode. Fixed MaleCNS + the frozen 829-parameter visual→DN bridge. Vision-driven lateral saves. |
| 3 | **Random Fly** | Existing random lateral controller (chance baseline). |
| 4 | **Heuristic Fly** | Existing oracle. **Reads privileged simulator ball state**; labelled *(oracle)* in the UI. It never touches the brain. Shows the approximate physical ceiling. |
| 5 | **Plasticity Fly** | *Historical, not runnable in this build.* The targeted-plasticity experiment's negative result (≈37.5 %, vision-independent) is shown for context; the plasticity-trained state is deliberately not loaded (handoff §6/§29). |

Natural and Learned are neural modes: the only thing entering the brain is
rendered pixels. Random and Heuristic bypass the brain entirely.

---

## Player controls

| Input | Action |
| --- | --- |
| Mouse move over the pitch | Continuous left/right aim |
| `↑` / `↓` or scroll wheel | Shot power |
| `Space` or click | Shoot |
| `R` | Clear the result card / ready the next shot |
| `N` | Restart match (reset score) |
| `1`–`5` | Select goalkeeper mode |
| `B` | Toggle bridge ON/OFF (Learned mode) |
| `V` | Cycle vision condition |
| `D` | Toggle science / neural debug panel |
| `S` | Run the deterministic Science-Demo shot set |
| `Esc` / `Q` | Quit |

Aim is continuous and clamped to the validated shot regime
(`AIM_MAX = 0.67` lateral; speed 4.5–6.0 cm/s) so shots stay in-distribution
(handoff §14). The `group` label (left/centre/right) is derived from aim sign
for scoring only and is **never** given to any controller.

---

## Scientific controls

Cycle with `V` (or `--vision`). Each reuses the exact validated implementation:

| Condition | Implementation | Expected effect on Learned Bridge |
| --- | --- | --- |
| **Normal Vision** | `VisionBridge` passthrough | vision-driven saves (~71 % on held-out test) |
| **Blind** | eye frame zeroed | collapses toward Natural baseline |
| **Mirrored Vision** | horizontal flip | laterality degrades |
| **Shuffled Vision (fixed permutation)** | one fixed R1–R6 permutation | **does not necessarily collapse** — a fixed permutation is a bijection that preserves L/R separability |
| **Temporal / Per-frame Visual Shuffle** | fresh permutation every 20 ms | collapses toward baseline (the genuine structure-dependence control) |

The two shuffle conditions are labelled distinctly and honestly: the report
established that a fixed spatial permutation preserves usable L/R separability
whereas per-timestep shuffling destroys it. **Bridge OFF** (key `B` in Learned
mode) reduces the loop bit-for-bit to Natural MaleCNS.

---

## Frozen model loading

The game never reconstructs the model from ad hoc defaults. On startup
`artifact_integrity.load_verified_artifact()`:

1. **Checksums (SHA-256).** Verifies `bridge_model.npz`, `visual_manifest.npz`,
   and the MaleCNS `graph.npz` against the values frozen at
   `learned-visual-dn-bridge-complete`. The `MaleCNSBrain` adapter also
   independently checks the graph SHA-256 at construction.
2. **Metadata invariants.** Confirms `n_windows = 4`, `n_neurons = 207`,
   `n_params = 829`, `runtime_gain = 3.5`, `cmd_smoothing = 0.8`,
   `drive_mv = 25`, `feature_order = newest_first`, tanh output transform, and
   that the manifest neuron order matches the frozen `sel_body` order.
3. **Assembles a versioned `ModelArtifact`** carrying feature/body IDs, ordering,
   temporal-window definition, normalization (mu/sd), weights, bias, output
   transform, bridge gain, DN IDs + opponent basis, current bounds, and
   model/version metadata (`model_version = learned-bridge-v1`).

Any mismatch raises `IntegrityError` and the game refuses to start — a
deliberate tripwire, not a silent fallback.

> **Artifact location.** Following the existing project convention,
> `workspace/outputs/` is git-ignored, so the frozen `.npz` artifacts live
> locally (regenerated at the `learned-visual-dn-bridge-complete` state) rather
> than being committed. The checksums are hard-coded in `artifact_integrity.py`
> and the expected regression values in
> `regression_learned_bridge_expected.json` (both tracked), so integrity is
> fully verifiable whenever the artifacts are present.

Frozen checksums (recorded at the tag):

```
bridge_model.npz     7cd477b9ed597d78db04c4286a34b74454fcfb3a9ff3c839d49093df1f97a9a0
visual_manifest.npz  de67c07edcdb0841ebfc1c435cca435781f8520eca16b5349224f6203ce689ee
graph.npz            346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b
```

---

## Simulation timing — rendering vs scientific simulation

The full 166,700-neuron MaleCNS step costs ~0.1 s of wall time per 20 ms neural
decision on the CPU reference backend, so the scientific loop runs at roughly
**0.12× real time**. Per handoff §18/§19 we do **not** alter any scientific
timing to smooth rendering. Instead:

- The entire scientific simulation (retina timing, 20 ms neural decision,
  MuJoCo timestep, bridge temporal windows, injection lag) runs in a background
  **worker thread** at its true cadence, unchanged.
- The UI thread renders whatever snapshot the worker has produced, at display
  rate. A `sim/real` pace indicator is shown during a shot.
- **All MuJoCo rendering (the scientific `VisionBridge` retina render, the
  scene view, and the fly-eye inset) runs on the same worker thread.** On macOS
  a MuJoCo GL context is bound to its creating thread and a second context on
  another thread deadlocks the native kernel; keeping every GL context on the
  worker avoids this. The main thread only runs pygame/SDL.

Nothing in the game changes the MuJoCo timestep, neural timestep, retina timing,
bridge windows, or bridge update timing.

---

## Regression tests

Run the whole suite:

```sh
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.run_regressions
```

1. **Artifact integrity** — frozen checksums + metadata
   (`artifact_integrity.py`). **PASS.**
2. **Bridge-OFF regression (Natural MaleCNS unchanged)** — the existing
   `learned_bridge.regression_bridge_off`: a present-but-disabled bridge is
   bit-for-bit identical to no bridge across all matched steps (max abs diff
   0.0 in fly_y, lateral, and left/right DN spikes) and reproduces the Natural
   signature (100 % centre). Confirms the game/UI layer does not perturb the
   baseline. **PASS.**
3. **Frozen learned-bridge regression** (`regression_learned_bridge.py`) — runs
   the frozen bridge through the *game engine* on a fixed deterministic shot set
   and compares outcomes, decision counts, and the summed scalar command `u`
   against stored expected values (`regression_learned_bridge_expected.json`).
   The learned loop is deterministic and bit-stable, so outcomes must match
   exactly. Detects accidental changes to neuron ordering, feature windows,
   normalization, weight loading, DN basis, bridge gain, or timing. Fails loudly
   (never silently continues). **PASS.**
4. **No-privileged-leak regression** (`regression_no_privileged_leak.py`) —
   (a) a static audit that the neural controller/vision hold no reference to the
   world/ball/shot; (b) a behavioural immunity test: corrupting the world's
   ball observation around the controller's decision leaves the learned command
   stream **byte-identical** (pixels-only confirmed), while the same corruption
   *does* change the heuristic oracle (proving the probe is meaningful).
   **PASS.**

---

## UI / demo instructions

**Dependencies.** The project virtualenv `upstream/doomfly/.venv-neural`
(Python 3.11) with MuJoCo 3.13, NumPy, `pygame` (2.5.x), and `imageio`. No web
server or extra install is required. Uses the CPU scientific-reference backend.

**Launch (from the repo root):**

```sh
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.game
```

Optional flags: `--mode {natural|learned|random|heuristic|plasticity}`,
`--vision {normal|blind|mirrored|shuffled_fixed|shuffled_perstep}`,
`--match-len N`.

**Headless smoke tests** (no display; useful for CI):

```sh
# session-level (one shot per mode)
… -m workspace.experiments.interactive_demo.game --selftest
# full-UI (dummy video driver, auto-shoots one penalty)
… -m workspace.experiments.interactive_demo.game --ui-selftest
```

**Recommended demo sequence** (tells the research story in a few minutes):

1. **Learned Bridge Fly, Normal** — vision-driven lateral saves.
2. Press `B` → **Bridge OFF** (Natural Fly) — mostly passive; only centre saved.
3. Press `V` to **Blind** — performance collapses toward the natural baseline.
4. `V` to **Mirrored** — laterality degrades / inverts.
5. Switch to **Heuristic (oracle)** — the physical ceiling.
6. Press `S` for the **Science-Demo** — the same fixed shots for repeatable
   comparison.

**Screen layout.** Third-person penalty scene (a presentation-only camera behind
the goal line, framing ball + goal + fly), continuous aim + power bars, a
`SAVE`/`GOAL` result card, the mode selector and match scoreboard, a
collapsible **Science / Neural Debug** panel (controller/vision, bridge ON/OFF,
scalar `u` with a command-history sparkline, left/right DN spikes and drive,
injected current, retina mean, brain spikes, outcome, cumulative save rate), and
a **FLY VISION** inset with the **R1–R6 RETINA** panel. The fly-eye inset shows
the *actual* input used by the retina under the current vision condition (blind
is truly black; mirrored is truly flipped) — it is not a mock-up.

---

## Deterministic Science-Demo mode

`S` runs a fixed list of held-out shots (`science_demo_shots`, reusing the same
per-seed `GoalkeeperWorld(seed).sample_shot(group)` construction as the frozen
evaluation, base seed 90000). Because the shots are identical across modes, you
can compare Natural / Learned / Random / Heuristic on the same penalties. This
is a live demonstration only; it does **not** overwrite or replace the
experiment's benchmark numbers in `LEARNED_BRIDGE_REPORT.md`.

---

## Known limitations

- **Not real time.** Learned/Natural modes run at ~0.12× real time on CPU; each
  penalty takes a handful of seconds of wall clock. This is inherent to
  simulating a 166,700-neuron brain honestly and is displayed to the player.
- **Toy scale.** The arena is deliberately normalized (large ball, cm-scale
  fly), not FIFA-accurate — inherited from the frozen environment.
- **Left/right asymmetry.** As reported in the science, left saves lag right; a
  real, mechanistically-explained embodiment/competition effect, not tuned away.
- **Plasticity mode is historical only** — not runnable in this build.
- **macOS single-GL-thread.** All MuJoCo rendering is confined to the sim worker
  thread; on other platforms the same design works but the constraint is a macOS
  specific reason for it.
- The fly-eye inset renders `eye_left` for visualization; the scientific retina
  drive is produced inside the engine's own `vision.perceive` during each
  decision (same camera, same condition transform).

---

## Scientific integrity

**No retraining or model tuning occurred during game development.** Verified:

- The frozen `bridge_model.npz` and `visual_manifest.npz` are byte-identical to
  the `learned-visual-dn-bridge-complete` state (checksums above; the artifact
  integrity check and the learned-bridge regression both enforce this).
- No bridge weights, selected neurons, DN stimulation, bridge gain, command
  smoothing, retinal encoding, MaleCNS weights, LIF dynamics, decoder, or CPG
  were changed. The only new "parameter" is a presentation-only free camera pose
  used for the player's view; it touches no scientific state (confirmed: the
  learned-bridge regression outcomes are unchanged after adding it).
- In Learned Bridge mode the controller receives only rendered pixels; the
  no-privileged-leak regression proves ball state cannot reach it.
- The 48-shot held-out result and all other numbers in
  `LEARNED_BRIDGE_REPORT.md` remain the experiment's official results; the game
  and its Science-Demo do not modify them.
