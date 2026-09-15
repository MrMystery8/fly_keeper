# Targeted plasticity for visual → descending routing — report

**Question:** Can reward-modulated changes restricted to the anatomically
identified visual → projection → descending pathway convert the existing
upstream directional information into genuinely vision-dependent motor behavior?

**Answer (honest, robustly confirmed):** No — not on this pathway with a
reward-modulated rule. The plasticity engages cleanly, stays bounded, and
strengthens *transmission* through the route, but it does **not** create
left/right directional selectivity, because the intermediate neurons on the
route do not carry a direction-differentiated signal for a reward-modulated rule
to exploit. The trained network evaluates at the passive floor and is fully
vision-independent (identical under blind / mirrored / shuffled). This matches
and extends the pathway-audit finding that direction dies at the optic-lobe →
visual-projection stage.

**Outcome: PLASTICITY FAILED — the missing ingredient is a direction-preserving
intermediate representation, not more learning on this route. Next step: engineer
a trainable readout, or re-scope the plastic pathway to include direction-carrying
intermediates (details in Recommendation).**

All fixed-weight work was preserved: this phase ran on a new branch
(`targeted-visual-motor-plasticity`), the CPU reference / retinal mapping / body
/ physics / diagnostics were not modified, and the connectome file on disk was
never written. Learning touched only an in-memory masked-weight copy.

---

## Rule (exact) — EXPERIMENTAL

This learning rule is a modeling choice, **not** a validated model of
*Drosophila* synaptic plasticity. It borrows structure from the DoomFly v6
experimental rule (eligibility traces, bounded sign-preserving efficacies,
three-factor reward modulation) but is a new, minimal, interpretable rule adapted
to the silent-visual-projection bootstrap problem the audit identified.

Per plastic synapse i→j, each 20 ms control window:

```
pre_i        = presynaptic spike count in the window
post_depol_j = max(0, v_j − V_rest)          # SUBTHRESHOLD depolarization,
                                              # v_rest = −52 mV; NO post spike required
eligibility_ij ← 0.8 · eligibility_ij + pre_i · post_depol_j
```

At episode end, reward R ∈ {+1 (SAVE), −1 (GOAL)}:

- **scalar credit:** `Δw_ij = lr · R · eligibility_ij`
- **directional credit:** `Δw_ij = lr · R · (a · s_ij) · eligibility_ij`, where
  `a ∈ {+1,−1}` is the fly's OWN committed exploratory strafe direction (motor
  self-knowledge, not ball information) and `s_ij ∈ {+1,−1}` is the side of the
  descending population the edge feeds.

Weights are clipped to `[0.1×, 2.0×]` baseline, sign-preserving; NaN/Inf are
rejected. Parameters used: `lr` 0.02–0.06, eligibility decay 0.8, subthreshold
eligibility (chosen after the bootstrap check below).

The subthreshold eligibility is essential and was justified empirically: with
spike-only eligibility only 9–26 of 1,211 masked synapses were ever eligible per
episode (the visual-projection targets are near-silent), versus 85–117 with the
subthreshold form.

## Plasticity scope

Explicit mask (`build_mask.py`): **1,211 synapses = 0.0047 % of the graph**.
- 114 `source → mid` edges: strongly directional visual neurons → pre-descending
  intermediates (Tm4, MeVP9, …).
- 1,097 `mid → DN` edges: intermediates → the target descending opponent pairs
  DNp20 (L 10162 / R 10059), DNpe017 (L 10527 / R 555871), DNp11 (L 10259 /
  R 10106).
All other synapses frozen. Baseline weights and the mask are saved; the brain can
be reset to the exact fixed baseline at any time.

## Curriculum

Strict **left/right-only** early curriculum, as required — center shots are
never used in training, so the fly cannot be rewarded for standing still and
blocking centre shots. Shots were made reachable (aim ±0.25 cm, slower speed) so
committed exploration yields balanced left/right saves (~50–80 %), giving a real
reward signal on both sides. Motor-level exploration is a per-episode committed
random strafe direction (never ball-informed), ON during training, OFF during
all evaluation. Held-out evaluation uses the full distribution (incl. centre) on
seeds disjoint from training.

## Neural effect (before → after training)

Directional separation of the target-DN populations (left-signal − right-signal),
fly held fixed, exploration off:

| rule / run | normal before→after | blind before→after |
| --- | --- | --- |
| scalar, 40 ep | −0.030 → −0.009 | −0.028 → −0.027 |
| scalar, 150 ep (lr 0.06) | −0.030 → −0.009 | −0.028 → −0.016 |
| directional, 100 ep | −0.030 → −0.016 | −0.028 → −0.002 |

Overall DN drive **rose** with training (e.g. 0.19 → 0.36 both sides) but the
left/right **separation stayed at ~0**, and normal ≈ blind throughout. Learning
strengthened common-mode transmission, not direction.

**Why (eligibility diagnosis, `diagnose_eligibility.py`):** the eligibility
available to the rule is ~87 % common-mode. At the `source → mid` stage the
left-vs-right eligibility differs by only ~13 % (1 of 114 edges strongly
directional); at the `mid → DN` stage the eligibility is tiny (~0.02) and
non-directional (4 of 1,097 edges). A reward-modulated rule can only amplify
structure the co-activity already contains; the directional structure is
essentially absent on this route, so both scalar and side-gated credit end up
strengthening the shared (common-mode) component.

## Learned weights (`analyze_weights.py`)

99 of 1,211 edges changed, bounded (0.36×–1.61×). The changes concentrate at the
**source → mid** stage (91/114 edges, mean |Δ| 15 %) and barely touch **mid → DN**
(5/1,097 edges, mean |Δ| 0.02 %) — because the near-silent DN targets have tiny
subthreshold eligibility, so almost no credit reaches the final synapse. The
left-DN vs right-DN **opponent asymmetry is −0.0003 (≈ 0)**: no directional
structure formed at the descending output.

## Behaviour — held-out evaluation (24 episodes/condition, held-out seeds)

| Controller | Save rate | Left | Center | Right |
| --- | ---: | ---: | ---: | ---: |
| passive | 37.5 % | — | — | — |
| random | 37.5 % | — | — | — |
| heuristic (ceiling) | 91.7 % | — | — | — |
| fixed MaleCNS | 37.5 % | — | — | — |
| **trained MaleCNS / normal** | **37.5 %** | 0 % | 100 % | 0 % |
| trained / blind | 37.5 % | 0 % | 100 % | 0 % |
| trained / mirrored | 37.5 % | 0 % | 100 % | 0 % |
| trained / shuffled | 37.5 % | 0 % | 100 % | 0 % |
| frozen-baseline recheck | 37.5 % | 0 % | 100 % | 0 % |

The trained decoder (plasticity off, deterministic) sits exactly at the passive
floor and is identical across all sensory controls — its only saves are
centre-by-standing.

## Controls

- **Blind / mirrored / shuffled = normal (all 37.5 %):** the behavior is
  completely vision-independent, so no genuine visually-driven sensorimotor
  behavior was learned. (A real learned controller would differ under at least
  blind and mirrored.)
- **Frozen-baseline recheck = 37.5 %:** resetting the masked weights to baseline
  reproduces the passive-like floor, confirming reproducibility and that nothing
  outside the mask changed.

## Ablation

Because trained == passive across every condition, the learned edges carry no
functional role — there is nothing causal to ablate away. The near-zero opponent
asymmetry confirms this directly, so a "reset the strongest learned synapses"
ablation cannot reduce a save rate that is already at the floor.

## Scientific status

The learning rule is **EXPERIMENTAL and unvalidated**; it is not a claim about
real fruit-fly plasticity. The fixed-weight CPU reference remains the scientific
ground truth and is preserved (tags `embodied-flykeeper-baseline`,
`neural-diagnostics-complete`, `pathway-audit-complete`; branch
`targeted-visual-motor-plasticity` isolates all learning work). Training used the
CPU reference backend only.

## Recommendation

**PLASTICITY FAILED on this pathway — do not keep tuning reward learning here.**
The bottleneck is representational, not a matter of learning rate, episode count,
or compute (a GPU backend would not help; it would run a common-mode-strengthening
rule faster). Two concrete next directions, in order of scientific cleanliness:

1. **Re-scope the plastic pathway to include a direction-preserving stage.** The
   directional signal is strong and clean at the retina/optic-lobe (100 %
   decodable) but is already ~gone at the specific intermediates on the current
   route. Before learning, search for intermediate populations that *retain* the
   left/right contrast (e.g. motion/looming-selective LPLC/LC types that appeared
   among the source→mid targets) and build the plastic mask onto *those*, so the
   eligibility itself is direction-differentiated and a reward rule has real
   structure to amplify.

2. **Engineer a trainable readout** (a small, explicit, interpretable linear map
   from a direction-carrying neural population — optic-lobe/visual-projection,
   where direction demonstrably exists — to the locomotion command), trained by
   the same reward, and reported transparently as an engineered decoder rather
   than connectome plasticity. This sidesteps the fixed connectome's failure to
   route direction to the descending neurons while keeping the sensory
   representation biological.

Whole-brain plasticity remains explicitly **not** recommended.

---

### Files
- `workspace/experiments/plasticity/{build_mask, engine, train, sanity, diagnose_eligibility, evaluate, analyze_weights}.py`
- Checkpoint: `workspace/checkpoints/run_directional/learned.npz` (masked weights only)
- Artifacts: `workspace/outputs/plasticity/*.json`, `pathway_table.json`
