# Summary — grading mechanistic explanations by weights-blind simulation

*Libor "Burny" Burian, [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en) — 2026-11-22.
Developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic interpretability
explanations, co-organized by the authors in the [mechinterp.com](https://mechinterp.com/) community. Every result
number below is generated from `results/` by `vmi analyse`.*

## Question

Can an explanation of a neural network be graded automatically by giving it to an agent that has never seen the
network, letting the agent write the network's weights, and comparing the rebuilt network with the original? Is
that score a valid measure of explanation quality?

## Setup in five lines

1. Target: one-layer modular-addition transformers ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)); the main
   target T0 and the seed targets T1, T2 are analysed in no paper (weights and key frequencies public on Hugging Face
   since 2026-10-21, after the simulator's June 2026 training-data cutoff; no internet in the sandbox), T-new was trained after the protocol freeze, and T-paper is the
   published checkpoint (flagged for memorisation).
2. Explanations rendered by script at five levels of true detail (E0–E4) plus wrong ones (W-*), length-matched except the Pizza explanation W-alg (about <!-- v:walg-shorter -->40<!-- /v --> % shorter).
3. Simulator: Claude Haiku 5.5 in Claude Code, no network except its model endpoint, no path to the weights; local Qwen models were tried and
   stopped after every run built a constant-output network (chance).
4. Scores: output KL (behaviour) and cross-validated hidden-state similarity HS (mechanism, linear alignment).
5. Calibration: an agent-free reference compiler, retrains, distilled students, random init, wrong targets, a planted
   mechanism, the training trajectory; hypotheses preregistered.

## Results so far

<!-- BEGIN key-results -->
*Round 2, N = 3 run(s) per condition — not yet N=10*

- **Model-specific detail raises hidden-state similarity:** HS on T0 is 0.05 (E1, algorithm only), 0.89 (E2), 0.93 (E3) and 0.94 (E4); with no explanation (E0) it is 0.13. Monotone from E1 to E4; E0 lies above E1, against the preregistered prediction.
- **Correct behaviour, wrong mechanism is caught by HS, not by output KL:** the wrong-frequency replacement reaches KL 0.003 nats (true E3: 0.010) but HS 0.07.
- **Copying behaviour does not copy the mechanism:** independent retrains (HS 0.12) and logit-distilled students (HS 0.23, KL 2.38e-06) score no higher than randomly initialised networks (HS 0.24). HS's scale is therefore not anchored at 0 for unrelated networks: it measures overlap of internal features, and working networks built on other frequencies score below random ones.
- **Negative result:** the primary HS does not separate the Pizza explanation (W-alg, HS 0.91) or the scrambled neuron table (W-scramble, HS 0.94) from the true explanations; the secondary matched-neuron correlation separates W-scramble (0.65) from E4 (0.91).
- **Memorisation (post hoc):** E0 and E1 contain nothing target-specific (identical text on every target, and the sandbox does not reveal the target), so their runs on T0 and on the published checkpoint are replicates of one condition. When the explanation gives no frequencies, the simulator often builds most of the published paper's frequencies {14, 35, 41, 42, 52} into its replacement. E1 (6 runs on T0 and T-paper): 5 of 6 build scripts list at least four of the paper's five frequencies (3 of 6 replacements build at least four by the key-frequency rule, a lower bound); on average 68% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.08 against T0 but 0.70 against the published checkpoint (target swap: same weights, other target; range 0.24–0.95); E0 (3 runs on T0): 1 of 3 build scripts list at least four of the paper's five frequencies (1 of 3 replacements build at least four by the key-frequency rule, a lower bound); on average 36% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.13 against T0 but 0.28 against the published checkpoint (target swap: same weights, other target; range 0.03–0.63). On a published model, a high score can therefore come from the simulator's memory rather than the explanation (one published model and one simulator tested).
- **Cost:** the 96 reported Claude Haiku 5.5 runs cost $2.20 ($2.33 including development runs and plumbing checks).
<!-- END key-results -->

## Hypotheses: predicted vs. observed

<!-- BEGIN predictions -->
| Hypothesis | Predicted (preregistration §5) | Measured with the main simulator (T0) | Direction matches? | Formal verdict |
|---|-------|---------|-----|------|
| H1 | HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1 | HS E0 0.130, E1 0.046, E2 0.894, E3 0.929, E4 0.943; KL E1 0.097, E2 2.12e-04, E3 0.010, E4 0.060 | partly (KL does not fall E1→E4; E0 not below E1) | HS: supported; KL_out: not supported · at N=3 — not yet N=10 |
| H2 | true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble | HS difference E3>W-freq +0.859 [+0.835, +0.880]; E3>W-alg +0.020 [-0.008, +0.045]; E3>W-role +0.725 [+0.661, +0.768]; E4>W-scramble +0.007 [-0.025, +0.037] | partly (E3>W-alg tie (CI includes 0); E4>W-scramble tie (CI includes 0)) | E3>W-freq: cannot tell at this N; E3>W-alg: cannot tell at this N; E3>W-role: cannot tell at this N; E4>W-scramble: cannot tell at this N · at N=3 — not yet N=10 |
| H3 | W-freq: argmax agreement ≥ 0.99, KL not above the true E2–E4 range, HS below each of E2–E4 | W-freq: agreement 1.000, KL 0.003 (true 2.12e-04–0.060), HS 0.070 (true 0.894–0.943); HS − E2 -0.824 [-0.840, -0.801], E3 -0.859 [-0.880, -0.835], E4 -0.873 [-0.904, -0.841] | yes | supported · at N=3 — not yet N=10 |
| H4 | E3 and E4 beat independent retrains on HS (vs. distilled students: open question, no direction) | E3 0.929, E4 0.943; 9 retrains mean 0.119 (max 0.329); 3 students mean 0.233 (max 0.282); vs students, two-sided: E3 cannot tell at this N, E4 cannot tell at this N | yes | E3: supported; E4: supported · at N=3 — not yet N=10 |
| H4b | planted model: HS = 1 and KL = 0 at step 0, then HS strictly falls and KL strictly rises | HS 1.000 → 0.998 → 0.957 → 0.888; KL 0.000 → 0.371 → 0.040 → 4.25e-04 | partly (KL not strictly rising) | not supported · deterministic |
| H5 | agent ≤ reference compiler on every explanation; the gap shrinks with simulator capability | T0 (preregistered): agent − compiler E2 -0.001, E3 +0.002, E4 -0.009, W-freq -0.024, none of the four explanations beyond the bootstrap CI; seed targets T1, T2, T-new (extension, post hoc): agent above compiler in six of eleven pairs by sign, four of eleven target–explanation pairs on three targets beyond the bootstrap CI (T1 E2, T2 E3, T2 E4, T-new E4; largest +0.179 (T2 E3)); T-paper (memorisation-flagged, not counted): E3 +0.073 beyond the CI, E4 -0.046, W-freq -0.004; 5 local runs, 0 above chance | partly (seed targets: agent above compiler beyond its CI; capability trend not testable) | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H6 | the local simulators rank E2, E3, E4, W-freq the same way | 5 local runs, 0 above chance: nothing to rank | not testable | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H7 | a T0 replacement fits T0 better than any other trained or no-mechanism model (HS) | E3: HS 0.939 vs best other 0.274 (T2); E3: HS 0.932 vs best other 0.240 (T2); E3: HS 0.915 vs best other 0.290 (T2); E4: HS 0.912 vs best other 0.226 (T2); E4: HS 0.953 vs best other 0.227 (T2); E4: HS 0.965 vs best other 0.258 (T2) | yes | supported · 6 E3/E4 replacements (at N=3 — not yet N=10) |
| H8 | HS against T0's checkpoints rises as restricted loss falls (circuit formation) | Spearman ρ(HS, −log restricted loss) per replacement: median 0.774, min 0.751 (6 E3/E4 replacements × 401 checkpoints); negative controls (post hoc), ρ per replacement: E0 -0.83, -0.45, +0.14; W-freq -0.40, -0.24, -0.38; E1 -0.58, -0.58, -0.57 | yes | supported · 6 E3/E4 replacements × 401 checkpoints |
<!-- END predictions -->

Replications on the seed-sensitivity targets (preregistration §5: H1–H3 re-tested per target with the preregistered tests, deviation 8; first row: T0 restricted to E2–E4 for comparison, post hoc):

<!-- BEGIN replications -->
| Target | HS per true explanation | H1: HS rises with detail? (trend test) | H2: HS(E3) − HS(W-freq); Mann–Whitney | H3 observed (W-freq) | H3 holds? |
|-----|------|---|---|-------|-----|
| T0 (E2–E4 only; post hoc, for comparison) | E2 0.894, E3 0.929, E4 0.943 | means rise E2→E3→E4; supported (JT p 0.011, ρ 0.79) | +0.859; cannot tell at this N (p 0.050) | agreement 1.000, KL 0.003 (true max 0.060), HS 0.070 | yes |
| T1 | E2 0.957, E3 0.936, E4 0.960 | means do not rise E2→E3→E4; not supported (JT p 0.500, ρ 0.05) | +0.885; cannot tell at this N (p 0.050) | agreement 0.854, KL 0.537 (true max 0.498), HS 0.051 | partly (agreement < 0.99; KL above the true range) |
| T2 | E2 0.900, E3 0.962, E4 0.951 | means do not rise E2→E3→E4; not supported (JT p 0.500, ρ 0.05) | +0.864; cannot tell at this N (p 0.050) | agreement 0.999, KL 0.006 (true max 0.049), HS 0.098 | yes |
| T-new | E3 0.910, E4 0.984 | means rise E3→E4; cannot tell at this N (JT p 0.050, ρ 0.88) | +0.816; cannot tell at this N (p 0.050) | agreement 0.907, KL 0.432 (true max 0.328), HS 0.094 | partly (agreement < 0.99; KL above the true range) |
| T-paper (flagged: memorisation) | E1 0.590, E3 0.940, E4 0.937 | means do not rise E3→E4; cannot tell at this N (JT p 0.650, ρ -0.10) | +0.823; cannot tell at this N (p 0.050) | agreement 0.998, KL 0.007 (true max 0.413), HS 0.117 | yes |
<!-- END replications -->

**Reading.** At N = 3 the preregistered tests support H1's HS part, H3, H4, H7 and H8: HS rises with true detail (the big step is E1 → E2; <!-- v:e3-vs-e2 -->every E3 run is above every E2 run<!-- /v --> under the preregistered HS, though E2 and E3 tie under the post-hoc whitened check; E3 → E4 is mixed), a wrong-frequency explanation keeps correct answers while its HS falls below every true explanation that names the frequencies (E2–E4), E3 and E4 replacements beat independent retrains and fit T0 better than any other model, and HS tracks circuit formation (descriptively; replacements without T0's mechanism do not, the paper's §4.3). The predictions fail where they assumed that output KL improves with detail (H1's KL part: it is lowest at E2 and rises at E3 and E4), that HS sees below the frequency level (H2: W-alg and W-scramble tie with the true explanations, their bootstrap CIs include 0), that the reference compiler is a ceiling (H5: the agent stays within its level on T0, the preregistered comparison, but exceeds it on the seed targets in <!-- v:h5-seed -->four of eleven target–explanation pairs on three targets<!-- /v -->), or that output KL grows smoothly as a planted model drifts (H4b). "Direction matches" compares directions and preregistered thresholds only (a difference whose bootstrap CI includes 0 counts as a tie; with three runs per group a percentile bootstrap has only ten distinct resamples per group, so these intervals are too narrow and a tie is the safe reading, an "ahead" is not significance); the formal verdict adds the preregistered test, and 3 vs 3 runs cannot reach significance for H2 (smallest attainable p = 0.05). On the seed-sensitivity targets the gap between E3 and the wrong-frequency explanation replicates on every target (its rank test cannot decide at N = 3), while the fine E2 → E4 rise does not replicate (its trend test is not supported on T1 and T2, although on T0 the same E2–E4 test gives <!-- v:t0-e24 -->supported (JT p 0.011, ρ 0.79)<!-- /v -->; on T-new, with only E3 and E4, the means rise but the test cannot decide at N = 3; the large E1 → E2 step was not run on these targets); where H3 fails there, argmax agreement is below the preregistered 0.99 for the wrong-frequency replacement and also for the true E3 and E4 replacements, so the explanation is not shown to be the cause (not diagnosed).

## What this means

- **Use hidden-state similarity, not output KL**, to grade explanations of this kind of model (evidence: one task
  family, one simulator, N = 3): behaviour can be copied without the mechanism, and KL rewards that. HS measures
  overlap of internal features: its scale is not anchored at 0 (random networks score above working networks that use
  other frequencies).
- **HS works at the level of frequencies, not below it.** It cannot tell the Clock algorithm from the Pizza algorithm
  with the same frequencies, nor a correct neuron table from a scrambled one; a neuron-level or causal second metric
  is needed for finer explanations.
- **Grade on models whose mechanism the simulator cannot have seen.** The simulator fills gaps in an explanation with what it remembers
  from the literature, which confounds scores of underspecified explanations on famous models (full explanations
  still scored as expected on the published checkpoint).

## Status and next steps

Rounds 1 and 2 are complete (N = 3 runs per condition). The local Qwen capability ablation was stopped after <!-- v:n-local -->five<!-- /v -->
runs, all at chance (user decision; preregistration deviation 5), so H5's capability trend and H6 cannot be tested. Round 2 (N = 3)
runs the 32 Claude Haiku conditions; round 3 (N = 10) is postponed to a later day. Open: a second metric for
sub-frequency structure. Details:
[paper](paper.pdf), [round-1 report](experiments/01-round1/experiment.md), [round-2 report](experiments/02-round2/experiment.md),
[reviews](reviews/round2-review.md), [independent audit](reviews/independent-audit-2026-11-22.md), [second audit](reviews/audit-2-2026-11-22.md),
[results explorer](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/).
