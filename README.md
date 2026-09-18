# Fruit Fly Brain-Powered Goalkeeper

An engineered fruit-fly goalkeeper built around a **fixed MaleCNS connectome**
(about 166,700 neurons and 25.6 million synapses), simulated eyes, real
descending-neuron (DN) activity, and a MuJoCo body. The project finished with
an arcade **WHERE/HOW early-intent controller** that saved **61/90 (67.8%)**
on an untouched, clean-goal test battery—compared with **21/90 (23.3%)** for
the continuous monocular baseline and **85/90 (94.4%)** for the privileged
physical oracle.

Read [the end-of-project history](docs/PROJECT_HISTORY.md) for the complete, evidence-linked
development record: all phases, branch decisions, successes, negative results,
architecture changes, compute accounting, and final limitations.

> This is a transparent simulation and research prototype, not a conscious fly,
> an exact digital organism, or a validated reproduction of fly physiology.
> The body, retinal encoding, control interfaces, learned bridge, and final
> policies are engineered; the fixed connectivity is preserved.

## Final result

The final champion pauses movement for eight 20-ms decisions (about 160 ms) to
form a bilateral visual early-intent estimate, then uses a frozen,
non-privileged 15-dimensional observation to choose physical DN-driven actions.
Every one of its 61 saves was a body deflection, and all 90 evaluation shots
were verified as natural misses without the keeper.

| Matched 90-shot controller | Saves |
| --- | ---: |
| Privileged physical oracle | 85/90 (94.4%) |
| Continuous monocular v2 baseline | 21/90 (23.3%) |
| Early-intent baseline | 51/90 (56.7%) |
| **Final early-intent champion** | **61/90 (67.8%)** |

The champion's bilateral visual dependence was checked on the same protocol:
left-eye blind 10.0%, right-eye blind 25.6%, both eyes blind 8.9%.

## Project arc

1. **Reference and embodiment.** CPU-native MaleCNS stayed the scientific
   reference; an optional Metal backend improved some throughput but did not
   retain exact closed-loop behavior. The embodied fixed brain saved only
   centre shots by standing still (about 33% overall).
2. **Diagnosis and negative results.** Probes showed direction in early visual
   activity and functional DNs, but the signal died before useful DNs.
   Targeted reward-modulated native plasticity strengthened common-mode activity
   without learning left/right control.
3. **Learned-bridge science result.** A frozen 829-parameter visual→DN bridge
   achieved 71% on a held-out 48-shot science-mode test and failed the right
   controls when blinded, mirrored, shuffled, or switched off.
4. **Arcade redesign.** The project fixed a stale padded-dataset bug, showed
   that two eyes alone did not solve continuous control, and moved to early
   intent plus an explicit action policy.
5. **Final champion.** Differential vertical refinement produced the final
   67.8% untouched-test result and a regression-frozen presentation.

See [the end-of-project history](docs/PROJECT_HISTORY.md) for dates, commits, tags, branch
topology, evidence, and the decisions connecting every stage.

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

## Key reports

- [Documentation index](docs/README.md)
- [End-of-project history](docs/PROJECT_HISTORY.md)
- [Final early-intent champion](docs/reports/EARLY_INTENT_ACTION_POLICY_REPORT.md)
- [Learned visual→DN bridge](docs/reports/LEARNED_BRIDGE_REPORT.md)
- [Embodied baseline](docs/reports/EMBODIED_REPORT.md)
- [Neural diagnostics](docs/reports/NEURAL_DIAGNOSTICS_REPORT.md) and [pathway audit](docs/reports/PATHWAY_ACTIVATION_REPORT.md)
- [Arcade v2 data-integrity correction](docs/reports/ARCADE_BRIDGE_V2_REPORT.md)
- [Binocular experiment](docs/reports/ARCADE_BINOCULAR_REPORT.md)
- [Metal backend investigation](workspace/gpu/METAL_BACKEND_REPORT.md)

Historical reports are frozen evidence; this README and the `docs/` index are
the public map across them.

## Layout

- `workspace/` — reusable MaleCNS workspace: `adapters/` (brain), `embodiment/`
  (MuJoCo world, fly body, vision bridge, motor decoder), `experiments/`
  (`learned_bridge/`, `embodied_flykeeper/`, `neural_probe/`, `pathway_audit/`,
  `interactive_demo/`), `outputs/` (frozen artifacts + results).
- `upstream/doomfly/` — clean DoomFly checkout providing the native MaleCNS
  kernel and the Python venv (`.venv-neural`).
- `docs/` — project history, a report index, and frozen per-phase reports.

Dependencies: the `upstream/doomfly/.venv-neural` environment (Python 3.11,
MuJoCo 3.13, NumPy, pygame, imageio). CPU is the scientific-reference backend;
Metal is an optional, behaviorally relaxed throughput variant.
