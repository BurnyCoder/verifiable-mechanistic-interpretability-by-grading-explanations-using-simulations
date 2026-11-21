# Summary — grading mechanistic explanations by weights-blind simulation

*Libor "Burny" Burian, [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en) — 2026-11-21.
Developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic interpretability
explanations, co-organized by the authors in the [mechinterp.com](https://mechinterp.com/) community. Every number
below is generated from `results/` by `vmi analyse`.*

## Question

Can an explanation of a neural network be graded automatically by giving it to an agent that has never seen the
network, letting the agent write the network's weights, and comparing the rebuilt network with the original? Is
that score a valid measure of explanation quality?

## Setup in five lines

1. Target: one-layer modular-addition transformers ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)); the main
   target T0 and the seed-sensitivity targets are unpublished, T-new was trained after the protocol freeze.
2. Explanations rendered by script at five levels of true detail (E0–E4) plus length-matched wrong ones (W-*).
3. Simulator: Claude Haiku 5.5 in Claude Code, offline, no path to the weights; local Qwen models were tried and
   stopped after every run built a constant-output network (chance).
4. Scores: output KL (behaviour) and cross-validated hidden-state similarity HS (mechanism, linear alignment).
5. Calibration: an agent-free reference compiler, retrains, distilled students, random init, wrong targets, a planted
   mechanism, the training trajectory; hypotheses preregistered.

## Results so far

<!-- BEGIN key-results -->
*Round 1, N = 1 run(s) per condition — not yet N=3 or N=10*

- **Model-specific detail raises hidden-state similarity:** HS on T0 is 0.05 (E1, algorithm only), 0.90 (E2), 0.94 (E3) and 0.91 (E4); with no explanation (E0) it is 0.00. Not monotone: E4 < E3.
- **Correct behaviour, wrong mechanism is caught by HS, not by output KL:** the wrong-frequency replacement reaches KL 9.31e-04 nats (true E3: 0.023) but HS 0.06.
- **Copying behaviour does not copy the mechanism:** independent retrains (HS 0.12) and logit-distilled students (HS 0.23, KL 2.38e-06) stay at the random-initialisation floor (HS 0.24).
- **Negative result:** the primary HS does not separate the Pizza explanation (W-alg, HS 0.89) or the scrambled neuron table (W-scramble, HS 0.92) from the true explanations; the secondary matched-neuron correlation separates W-scramble (0.62) from E4 (0.89).
- **Memorisation (post hoc):** when the explanation gives no frequencies (E0, E1), the simulator's 2 replacement(s) on unpublished targets use the paper's published frequencies (mean overlap 0.80 with {14, 35, 41, 42, 52}, 0.00 with the target's own). Grading the same weights against the published checkpoint (target swap): E0: HS 0.00 against T0, 0.63 against the published checkpoint; E1: HS 0.05 against T0, 0.95 against the published checkpoint. A separate algorithm-only run on the published checkpoint scores HS 0.92: scores on published models measure memory, not the explanation.
- **Cost:** all Claude Haiku 5.5 runs so far cost $1.13 in total.
<!-- END key-results -->

## Hypotheses: predicted vs. observed

<!-- BEGIN predictions -->
| Hypothesis | Predicted (preregistration §5) | Measured with the main simulator (T0) | Direction matches? | Formal verdict |
|---|-------|---------|-----|------|
| H1 | HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1 | HS E0 0.004, E1 0.048, E2 0.899, E3 0.939, E4 0.912; KL E1 1.05e-04, E2 1.20e-05, E3 0.023, E4 0.087 | partly (HS not monotone E1→E4; KL does not fall E1→E4) | not supported · at N=1 — not yet N=3 or N=10 |
| H2 | true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble | HS difference E3>W-freq +0.884; E3>W-alg +0.051; E3>W-role +0.771; E4>W-scramble -0.006 | partly (E4>W-scramble reversed) | E3>W-freq: cannot tell at this N; E3>W-alg: cannot tell at this N; E3>W-role: cannot tell at this N; E4>W-scramble: not supported · at N=1 — not yet N=3 or N=10 |
| H3 | W-freq: argmax agreement ≥ 0.99, KL not above the true E2–E4 range, HS below each of E2–E4 | W-freq: agreement 1.000, KL 9.31e-04 (true 1.20e-05–0.087), HS 0.055 (true 0.899–0.939) | yes | supported · at N=1 — not yet N=3 or N=10 |
| H4 | E3 and E4 beat independent retrains on HS (vs. distilled students: open question, no direction) | E3 0.939, E4 0.912; 9 retrains mean 0.119 (max 0.329); 3 students mean 0.233 (max 0.282) | yes | E3: cannot tell at this N; E4: cannot tell at this N · at N=1 — not yet N=3 or N=10 |
| H4b | planted model: HS = 1 and KL = 0 at step 0, then HS strictly falls and KL strictly rises | HS 1.000 → 0.998 → 0.957 → 0.888; KL 0.000 → 0.371 → 0.040 → 4.25e-04 | partly (KL not strictly rising) | not supported · deterministic |
| H5 | agent ≤ reference compiler on every explanation; the gap shrinks with simulator capability | agent − compiler on T0: E2 +0.005, E3 +0.013, E4 -0.040, W-freq -0.039; agent above compiler in 7 of 15 target–explanation pairs (largest +0.161, T2 E3); 5 local runs, 0 above chance | no (agent above compiler somewhere; capability trend not testable) | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H6 | the local simulators rank E2, E3, E4, W-freq the same way | 5 local runs, 0 above chance: nothing to rank | not testable | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H7 | a T0 replacement fits T0 better than any other trained or no-mechanism model (HS) | E3: HS 0.939 vs best other 0.274 (T2); E4: HS 0.912 vs best other 0.226 (T2) | yes | cannot tell at this N · 2 E3/E4 replacements (at N=1 — not yet N=3 or N=10) |
| H8 | HS against T0's checkpoints rises as restricted loss falls (circuit formation) | Spearman ρ(HS, −log restricted loss) = 0.842 | yes | supported · one replacement, all checkpoints |
<!-- END predictions -->

Replications on the seed-sensitivity targets (preregistration §5: H1–H3 re-tested per target with the same rules):

<!-- BEGIN replications -->
| Target | HS per true explanation | H1: HS rises E2→E4? | H2: HS(E3) − HS(W-freq) | H3 observed (W-freq) | H3 holds? |
|-----|------|---|---|-------|-----|
| T1 | E2 0.955, E3 0.917, E4 0.955 | no | +0.883 | agreement 0.806, KL 0.669 (true max 0.773), HS 0.034 | partly (agreement < 0.99) |
| T2 | E2 0.949, E3 0.944, E4 0.934 | no | +0.855 | agreement 1.000, KL 0.005 (true max 0.112), HS 0.089 | yes |
| T-new | E3 0.846, E4 0.984 | yes | +0.759 | agreement 0.897, KL 0.378 (true max 0.344), HS 0.087 | partly (agreement < 0.99; KL above the true range) |
| T-paper (flagged: memorisation) | E1 0.925, E3 0.944, E4 0.929 | no | +0.852 | agreement 1.000, KL 9.56e-05 (true max 0.518), HS 0.091 | yes |
<!-- END replications -->

**Reading.** Where a prediction concerns the frequency level, the main simulator's results go the predicted way (H3, H4, H7, H8, and H2 for W-freq and W-role). The predictions fail where they assumed that every added level of detail helps (H1: HS gains nothing after E2 and output KL rises), that HS sees below the frequency level (H2: W-alg nearly ties, W-scramble is reversed), that the reference compiler is a ceiling (H5), or that output KL grows smoothly as a planted model drifts (H4b: KL spikes after the first optimiser steps, then falls). "Direction matches" compares directions and preregistered thresholds only; the formal verdict adds the preregistered test, which at small N mostly cannot reach significance. On the seed-sensitivity targets the gap between E3 and the wrong-frequency explanation replicates on every target and H1's monotone rise does not; where H3 fails there, the wrong-frequency replacement itself is broken (argmax agreement below the preregistered 0.99), an implementation error of the simulator rather than a property of the explanation.

## What this means

- **Use hidden-state similarity, not output KL**, to grade explanations of this kind of model: behaviour can be copied
  without the mechanism, and KL rewards that.
- **HS works at the level of frequencies, not below it.** It cannot tell the Clock algorithm from the Pizza algorithm
  with the same frequencies, nor a correct neuron table from a scrambled one; a neuron-level or causal second metric
  is needed for finer explanations.
- **Grade on unpublished models.** The simulator fills gaps in an explanation with what it remembers from the
  literature, which makes scores on famous models meaningless.

## Status and next steps

Round 1 (N = 1 per condition) is complete. The local Qwen capability ablation was stopped after five runs, all at
chance (user decision; preregistration deviation 5), so H5 and H6 cannot be tested. Round 2 (N = 3)
runs the 32 Claude Haiku conditions; round 3 (N = 10) is postponed to a later day. Open: a second metric for
sub-frequency structure. Details:
[paper](paper.pdf), [round-1 report](experiments/01-round1/experiment.md), [self-review](reviews/round1-review.md),
[results explorer](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/).
