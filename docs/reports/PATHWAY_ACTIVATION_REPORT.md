# Pathway activation audit — can the fixed network transmit visual direction to the descending neurons?

**Question:** Can the *fixed* MaleCNS network (no synaptic weight changes) be made
to transmit visual left/right shot direction from the optic lobe, through the
visual-projection / intermediate populations, to the descending neurons — using
only its existing connectivity and dynamics (sensory gain, integration time,
readout)? And if not, where exactly does the signal fail?

**Answer:** No. Direction is strong and clean in the retina and optic lobe, but
the **left/right directional contrast collapses at the optic-lobe →
visual-projection transition**, and the visual-projection population is
near-silent regardless of sensory gain or integration time. The descending
neurons receive only a **common-mode** ("something is out there") depolarization,
never a directional one. No fixed-network knob rescues it. **Decision-tree
outcome: 4** (signal dies at the visual-projection stage), with the precise
mechanism being loss of the directional component — not a missing wire and not a
simple global threshold problem.

The MaleCNS scientific core, the embodied environment, and the frozen baseline
were not modified. State was read-only; the only intervention was bounded
external current (≤ 30 mV) for causal stimulation. Frozen at tags
`neural-diagnostics-complete` and `embodied-flykeeper-baseline`.

---

## Candidate pathways

From connectivity tracing combined with per-neuron directional strength
(`rank_pathways.py` → `candidate_pathways.json`):

- **663** strongly direction-selective visual source neurons (left-vs-right
  `|AUC−0.5|·2 ≥ 0.8`) reach **664** distinct descending neurons via **13,420**
  two-hop `source → intermediate → DN` routes. **49,308** neurons project
  directly onto some DN.
- The top routes are anatomically sensible lamina→medulla→descending chains,
  e.g. `L2 → Tm4 → DNp11`, `L3 → Mi1 → DNp30`, `L5 → MeVP9 → DNp22`, with
  synaptic contact weights of ~3–13.
- The 4 visually responsive DNs (DNp20 L/R = 10162/10059; DNpe017 L/R =
  10527/555871) each have 43–62 such routes.

So the **anatomical route exists** — this is not a "no pathway" case.

## Subthreshold analysis (the key measurement)

`subthreshold.py` → `subthreshold.json`, reading membrane `v` (rest −52 mV,
threshold −45 mV), conductance `g`, and distance-to-threshold for candidate
neurons under moving left/right/centre/blind shots:

- **Visual input does reach the descending neurons subthreshold.** Mean membrane
  vs the blind baseline: DNp20_L **+0.89 mV**, DNpe017_L **+0.65 mV**,
  DNpe017_R **+0.64 mV**. The DNs depolarize when the fly can see.
- **But the left/right (directional) component is essentially absent at the DN
  membrane:** `v(right) − v(left)` = +0.0006, 0.0000, −0.15, −0.23 mV for the
  four DNs. That is 20–600× smaller than the 4.3–6.5 mV that still separates them
  from threshold.

So the DNs receive a **common-mode** visual depolarization but not a
**directional** one. This is not merely "signal below threshold waiting for more
gain" — the directional part is gone before it arrives.

## Causal stimulation (which transition fails)

`causal_stim.py` → `causal_stim.json`, driving one stage with +25 mV external
current and measuring the next:

| Transition | downstream spiking | mean depol | interpretation |
| --- | --- | --- | --- |
| source → intermediate | 8 % → 15 % | +1.27 mV | transmits weakly |
| intermediate → descending | 3 % → 18 % | +6.0 mV (max) | transmits when driven hard |
| **source → descending (2-hop end-to-end)** | 3 % → 6 % | **+0.25 mV** | **severe attenuation** |
| descending self-drive (ref) | 3 % → 91 % | +2.2 mV | DNs fire when driven |

Each single hop can transmit when its input is strongly forced, but the two-hop
chain end-to-end barely propagates (+0.25 mV). The route conducts common-mode
activity but attenuates it, and — combined with the subthreshold result — does
not preserve the directional contrast.

## Gain sweep

`gain_sweep.py` → `gain_sweep.json`, scaling retinal luminance 0.25×–8× and
measuring the directional signal `|R−L|` (total spikes) per stage:

| Stage | directional signal across all gains | notes |
| --- | --- | --- |
| retina | 236–2295 (huge) | perfect decode at every gain |
| optic lobe | ±6 to ±326 (tiny; falls as gain rises) | blind common-mode (38,216) exceeds visual |
| **visual projection** | **0 at every gain; ~28–32 total spikes** | near-silent, gain does not activate it |
| descending | 0 (or ±1 noise) | no gain produces a directional DN response |

**No sensory-gain regime produces a directional descending response.** More gain
only adds common-mode drive; it never restores the lost direction. (The
occasional "decode = 1.0" at central/descending is a nearest-centroid artifact
fitting ±1-spike noise, not a real signal.)

## Temporal sweep

`temporal_sweep.py` → `temporal_sweep.json`, directional decode vs cumulative
integration window (5–120 ms):

- Retina and optic lobe: 100 % at every window.
- **Visual projection: chance (0.50) at every window** — more integration time
  does not help a silent population.
- Central/descending: apparent 1.0 only at ≥50–100 ms, but with directional
  magnitude `|R−L|` = 0.00 to −1.00 spikes — again a noise artifact, not signal.

Longer integration does not restore genuine deep directional transmission.

## Baseline / saturation / common-mode

`common_mode.py` → `common_mode.json` (moving shots), directional `R−L` vs
common-mode `(R+L)/2` per stage:

| Stage | directional (R−L) | common-mode | dir/common | directional neurons |
| --- | ---: | ---: | ---: | ---: |
| retina | 1203 | 44,096 | 0.027 | 309 |
| optic lobe | 6.7 | 28,821 | 0.0002 | 31 |
| visual projection | 0.0 | 30 | 0.0 | 0 |
| central | −1.0 | 1.5 | — | 0 |
| descending | 0.0 | 16 | 0.0 | 0 |

The directional-to-common-mode ratio drops **100× from retina to optic lobe**
(0.027 → 0.0002) and to **zero** by the visual-projection stage, while
common-mode activity stays large. The tonic lamina drive is so strong that the
optic lobe's *blind* common-mode (38,216) exceeds its visual common-mode — a
large shared drive that carries no direction and swamps the vanishing L/R
contrast.

## Laterality

`laterality.py` → `laterality.json`:

- **Natural vision:** the responsive-DN right-minus-left preference is **0.00 for
  left, right, mirrored-left, and mirrored-right** shots — completely flat, with
  no mirror reversal. (Blind gives +1, i.e. noise.)
- **Causal:** driving the left-hemisphere directional visual sources vs the
  right-hemisphere ones yields a DN L/R preference of −1 / 0 — negligible even
  under direct, strong hemisphere stimulation.

The opponent left/right routing to the descending stage is not functionally
expressed under the fixed dynamics.

## Exact failure point

**The directional signal is lost at the optic-lobe → visual-projection
transition, and the visual-projection population is near-silent (~30 spikes at
any gain).** Consequently:

1. Retina / optic lobe: direction present and strong.
2. Visual projection: **near-silent, zero directional signal — the primary
   defect.**
3. Central / descending: receive only common-mode visual drive; zero directional
   signal; sit 4–6.5 mV below threshold with directional modulation ~0.01–0.2 mV.

This is Outcome 4 (die at visual projection), with the added detail that the
descending neurons are reached by common-mode but not directional input.

## Fixed-network verdict

Every no-learning lever has now been exhausted across this and the previous
phase: readout neuron choice, total-count vs timing readout, sensory gain
(0.25–8×), integration window (5–200 ms), and causal stage stimulation. **None
produce a genuine, vision-dependent directional descending response.** The fixed
network cannot be made to transmit visual direction to the descending neurons
with its existing weights.

## Recommendation

**TARGETED PLASTICITY JUSTIFIED — at the optic-lobe → visual-projection →
descending pathway**, scoped narrowly, not whole-brain.

Rationale, grounded in the audit:
- Direction is present and clean upstream (retina/optic lobe) — the information
  exists.
- An anatomical 2-hop route to the DNs exists (663 sources → 664 DNs).
- The DNs can drive the body correctly and symmetrically (established last phase;
  self-drive fires 91 % here).
- The single, specific defect is that the fixed weights do not route the
  optic-lobe directional contrast onto opponent descending populations, and the
  intervening visual-projection population is barely recruited.

So the minimal, appropriate intervention is a reward-modulated plasticity rule
restricted to the synapses on the identified `optic-lobe → visual-projection →
descending` routes (candidate DNp20/DNpe017 opponent pairs and their 2-hop
predecessors), with SAVE → positive and GOAL → negative modulation. Keep the
retinal pathway, the LIF core, and the body fixed.

Guardrails for that next phase (not started here):
- The DoomFly plasticity implementation is **EXPERIMENTAL and scientifically
  unvalidated**; if reused it must be labelled as such and must not be described
  as validated fruit-fly learning biology.
- **Preserve the fixed-weight CPU reference permanently** (tags
  `embodied-flykeeper-baseline`, `neural-diagnostics-complete`).
- The first learning experiment should be small, reversible, restricted to the
  named pathway, and explicitly experimental.

Do **not** enable whole-brain plasticity. Do not begin plasticity until this
scoped design is set up deliberately.

---

### Files
- Experiments: `workspace/experiments/pathway_audit/{rank_pathways, subthreshold, causal_stim, gain_sweep, common_mode, temporal_sweep, laterality}.py`
- Artifacts: `workspace/outputs/pathway_audit/*.json`
