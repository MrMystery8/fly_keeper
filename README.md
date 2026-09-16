# Fruit-Fly-Brain-Powered Goalkeeper

An articulated fruit fly, embodied in MuJoCo, plays goalkeeper. A **fixed MaleCNS
connectome** (≈166,700 neurons, ≈25.6 M synapses) processes the fly's own
simulated vision; a **small learned bridge (829 trainable parameters)** maps
informative visual activity onto real descending motor neurons that drive the
body. The native synaptic weights are never changed — the fly connectome did not
naturally know how to play football; the bridge is an engineered learned
transformation across an experimentally identified visual-to-motor bottleneck.

> This is a simulation using measured MaleCNS connectivity — not a conscious
> fly, an exact digital fly, or a validated reproduction of fly physiology. The
> LIF dynamics, retinal encoding, body, locomotion CPG, motor decoder, and the
> learned bridge are engineered components, transparently reported as such.

## Project arc

Each phase is frozen at a git tag with a standalone report; nothing below is
rewritten by later phases.

1. **Natural MaleCNS baseline** — the fixed brain, embodied, saves ~33 % of
   penalties: 0 % left, 100 % centre, 0 % right. It only "saves" centre shots by
   standing still. The body itself is capable (a privileged heuristic saves
   ~94 %). See `EMBODIED_REPORT.md` (`embodied-flykeeper-baseline`).

2. **Pathway audit** — diagnostics show the retina and early optic lobe decode
   left-vs-right essentially perfectly, and real descending neurons can causally
   drive useful left/right movement, but the directional signal **dies at the
   optic-lobe → visual-projection stage** and never reaches the descending
   neurons as opponent structure. See `NEURAL_DIAGNOSTICS_REPORT.md`,
   `PATHWAY_ACTIVATION_REPORT.md` (`neural-diagnostics-complete`,
   `pathway-audit-complete`).

3. **Targeted-plasticity negative result** — reward-modulated plasticity on that
   exact route *failed*: it strengthened a common-mode signal but never built
   left/right structure (trained ≈ passive ≈ 37.5 %, vision-independent). The
   defect is representational, not a lack of training (`targeted-plasticity-complete`).

4. **Minimal learned bridge** — a single-degree-of-freedom linear bridge from
   selected optic-lobe neurons to a fixed opponent descending-neuron basis
   raises held-out save rate from **33 % → 71 %** (p = 2 × 10⁻⁴). It is
   genuinely vision-dependent: it collapses when blinded (~35 %) or under a
   per-timestep-shuffled retina (~38 %), degrades when mirrored (~50 %), and is
   bit-for-bit Natural MaleCNS when switched off. It shrinks to as few as 16
   neurons / 65 parameters or a single DN pair. **The native synapses did not
   learn the task.** Full details in `LEARNED_BRIDGE_REPORT.md`
   (`learned-visual-dn-bridge-complete`).

5. **Interactive demo** — a playable 3D penalty game where you take penalties
   against the fly goalkeeper in several modes. The main **Learned Bridge Fly**
   mode runs the *exact frozen* chain from phase 4; no privileged ball state
   reaches the neural controller. Details in `INTERACTIVE_DEMO_REPORT.md`
   (branch `flykeeper-interactive`).

Headline numbers (48-shot held-out test set, `LEARNED_BRIDGE_REPORT.md`):

| Controller | Save rate |
| --- | ---: |
| Natural MaleCNS / Random / Passive | 0.33 |
| Targeted plasticity (historical) | 0.375 |
| **Learned bridge** | **0.71** |
| Heuristic (oracle ceiling) | 0.94 |

## Launch the interactive demo

From the repo root, using the project virtualenv:

```sh
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.game
```

Optional: `--mode {natural|learned|random|heuristic|plasticity}`,
`--vision {normal|blind|mirrored|shuffled_fixed|shuffled_perstep}`,
`--match-len N`.

Controls: **mouse** aim · **↑/↓ or scroll** power · **Space/click** shoot ·
**R** reset · **N** restart match · **1–5** mode · **B** bridge ON/OFF ·
**V** vision condition · **D** debug panel · **S** deterministic Science-Demo ·
**Q** quit.

The Learned Bridge mode simulates a 166,700-neuron brain per 20 ms decision, so
it runs at roughly 0.12× real time on CPU; each penalty takes a few seconds and
a `sim/real` pace indicator is shown. Scientific timing is never altered for
rendering.

Regression suite (artifact integrity, bridge-OFF, frozen learned bridge,
no-privileged-leak):

```sh
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.run_regressions
```

## Layout

- `workspace/` — reusable MaleCNS workspace: `adapters/` (brain), `embodiment/`
  (MuJoCo world, fly body, vision bridge, motor decoder), `experiments/`
  (`learned_bridge/`, `embodied_flykeeper/`, `neural_probe/`, `pathway_audit/`,
  `interactive_demo/`), `outputs/` (frozen artifacts + results).
- `upstream/doomfly/` — clean DoomFly checkout providing the native MaleCNS
  kernel and the Python venv (`.venv-neural`).
- `*_REPORT.md` — per-phase scientific reports (historical; do not edit).

Dependencies: the `upstream/doomfly/.venv-neural` environment (Python 3.11,
MuJoCo 3.13, NumPy, pygame, imageio). CPU scientific-reference backend.
