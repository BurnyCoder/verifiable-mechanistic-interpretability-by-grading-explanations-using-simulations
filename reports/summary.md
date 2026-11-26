# Summary — grading mechanistic explanations by weights-blind simulation

*Libor "Burny" Burian, [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en) — 2026-11-23.
Developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic interpretability
explanations, co-organized by the authors in the [mechinterp.com](https://mechinterp.com/) community. Every result
number below is generated from `results/` by `vmi analyse`.*

## Question

Can an agent reconstruct a network's activity when given its explanation, architecture, task and self-test, with target weights withheld? We compared the replacement's output probabilities and hidden states with the target on the finite modular-addition grid.

## Setup in five lines

1. Target: one-layer modular-addition transformers ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)); the main
   target T0 and the seed targets T1, T2 differ from the published checkpoint (weights and key frequencies uploaded to Hugging Face
   on 2026-10-21, after the simulator's June 2026 training-data cutoff and its 2026-10-07 release; no internet in the sandbox), T-new was trained after the protocol freeze, and T-paper is the
   published checkpoint (flagged for memorisation).
2. Explanations rendered by script at five levels of true detail (E0–E4; true at the level of frequencies and roles, except that T0's describe a read-out-only frequency as used everywhere) plus wrong ones (W-*), length-matched except the Pizza explanation W-alg (about <!-- v:walg-shorter -->40<!-- /v --> % shorter).
3. Simulator: Claude Haiku 5.5 in Claude Code, no network except its model endpoint, no path to the weights; local Qwen models were tried and
   stopped after every finished round-1 run built a constant-output network (chance).
4. Scores: output KL (behaviour: output probabilities) and cross-validated hidden-state similarity HS (held-out bidirectional linear prediction).
5. Calibration: an agent-free reference compiler, retrains, distilled students, random init, wrong targets, a planted
   mechanism, the training trajectory; hypotheses preregistered.

## Results so far

<!-- BEGIN key-results -->
*Round 2, N = 3 run(s) per condition — not yet N=10*

- **Model-specific detail raises hidden-state similarity:** HS on T0 is 0.05 (E1, algorithm only), 0.89 (E2), 0.93 (E3) and 0.94 (E4); with no explanation (E0) it is 0.13. Monotone from E1 to E4; E0 lies above E1, against the preregistered prediction.
- **Correct behaviour from the wrong key frequencies is caught by HS, not by output KL:** the wrong-frequency replacement reaches KL 0.003 nats (true E3: 0.010) but HS 0.07. The output logits do show it (post hoc, audit 9): after one fitted scale, the centred-logit FVU is 1.00 for W-freq against 0.12–0.21 for E3; KL misses it because a grokked model's output probabilities are nearly one-hot.
- **Small output KL coexists with low hidden-state similarity:** independent retrains (HS 0.12) and students trained to match T0's output probabilities (KL at temperature 1; HS 0.23, KL 2.38e-06) score no higher than randomly initialised networks (HS 0.24). HS's scale is therefore not anchored at 0 for unrelated networks: it measures overlap of internal features, and working networks built on other frequencies score below random ones. Neither copies T0's logits (centred-logit FVU: students 0.94–1.37, retrains 1.38–5.39; E3 replacements 0.12–0.21; post hoc, audit 9); students trained to match T0's centred logits (post hoc, pre-specified; FVU 0.023–0.165) reach HS 0.778–0.849: mean HS 0.849 lies between those ranges (below 0.85, above 0.35), which the pre-specified reading calls partial (1 of 3 met the training criterion; the others are not interpreted).
- **Negative result:** at N = 3 the primary HS does not separate the Pizza explanation (W-alg, HS 0.91; E3 − W-alg +0.020, bootstrap CI [-0.008, +0.045]) or the scrambled neuron table (W-scramble, HS 0.94; E4 − W-scramble +0.007, CI [-0.025, +0.037]) from the true explanations (a failure to separate, not a demonstrated equivalence); the secondary matched-neuron correlation separates W-scramble (0.65) from E4 (0.91). The agent-free compiler's literal build of the scrambled table does score 0.09 below its E4 build on T0, mostly at resid_post: 0.69 vs 0.95; elsewhere the gap is within ±0.007 on T1, T-new, T-paper; on T2 the scrambled build scores 0.05 higher. Post hoc, T0's distance irrelevance is in Zhong et al.'s Clock-typical range (0.75) and the literal Pizza build is in the Pizza-typical range (0.003); the agents' replacements from the true E1–E3 are Pizza-like too (means 0.13–0.18), and E4's are split (mean 0.41). Among the builds that carry T0's frequencies, the Clock-side ones (two of three E4 builds, q 0.46, and all three W-scramble builds, q 0.50–0.66) score HS 0.91–0.95 and the Pizza-side ones (E2, E3, E4, W-alg) 0.89–0.97: at this N, HS did not separate Clock-side from Pizza-side builds that share the key frequencies (descriptive; side and explanation are confounded).
- **Memorisation (post hoc):** E0 and E1 contain nothing target-specific (identical text on every target, and the sandbox does not reveal the target), so their runs on T0 and on the published checkpoint are replicates of one condition. When the explanation gives no frequencies, the simulator often builds most of the published paper's frequencies {14, 35, 41, 42, 52} into its replacement (possible cues include the sandbox description of the paper's architecture and, for E1, the paper's own algorithm; only E0's text also outlines the training setup; no sandbox file names a frequency or the paper; audit 8). E1 (6 runs on T0 and T-paper): 5 of 6 build scripts list at least four of the paper's five frequencies (3 of 6 replacements build at least four by the key-frequency rule, a lower bound); on average 68% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.08 against T0 but 0.70 against the published checkpoint (target swap: same weights, other target; range 0.24–0.95); E0 (3 runs on T0): 1 of 3 build scripts list at least four of the paper's five frequencies (1 of 3 replacements build at least four by the key-frequency rule, a lower bound); on average 36% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.13 against T0 but 0.28 against the published checkpoint (target swap: same weights, other target; range 0.03–0.63). On a published model, a high score can therefore come from the simulator's memory rather than the explanation. The origin of the simulator's prior knowledge is unknown (one published model and one simulator tested).
- **Exploratory logit comparison:** all 3 E3 builds had lower scale-adjusted logit error (0.123–0.210) than all 3 W-alg builds (0.232–0.232). E4 and W-scramble overlapped. These within-sample ranges add a descriptive result alongside the inconclusive registered HS comparisons.
- **Cost:** the 96 reported Claude Haiku 5.5 runs cost $2.20 ($2.33 including development runs and plumbing checks).
<!-- END key-results -->

## Hypotheses: predicted vs. observed

<!-- BEGIN predictions -->
| Hypothesis | Predicted (preregistration §5) | Measured with the main simulator (T0) | Direction matches? | Formal verdict |
|---|-------|---------|-----|------|
| H1 | HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1 | HS E0 0.130, E1 0.046, E2 0.894, E3 0.929, E4 0.943; KL E1 0.097, E2 2.12e-04, E3 0.010, E4 0.060 | partly (KL does not fall E1→E4; E0 not below E1) | HS trend E1→E4: supported; E0 below E1: not observed (E0 above E1; direction only, no test preregistered); KL_out: not supported · at N=3 — not yet N=10 |
| H2 | true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble | HS difference E3>W-freq +0.859 [+0.835, +0.880]; E3>W-alg +0.020 [-0.008, +0.045]; E3>W-role +0.725 [+0.661, +0.768]; E4>W-scramble +0.007 [-0.025, +0.037] | partly (E3>W-alg direction uncertain (CI includes 0); E4>W-scramble direction uncertain (CI includes 0)) | E3>W-freq: not supported (Holm p 0.20); E3>W-alg: not supported (Holm p 0.70); E3>W-role: not supported (Holm p 0.20); E4>W-scramble: not supported (Holm p 0.70) — preregistered §5 criterion; at N = 3 even complete separation gives Holm p 0.20, so these tests could not reach α. Post-hoc rule of deviation 1 (applied at this N by deviation 7): all read "cannot tell at this N" · at N=3 — not yet N=10 |
| H3 | W-freq: argmax agreement ≥ 0.99, KL within the E2–E4 mean range, HS below each true mean; the falsifier tests only the upper KL bound | W-freq: agreement 1.000, KL 0.003 (true 2.12e-04–0.060), HS 0.070 (true 0.894–0.943); HS − E2 -0.824 [-0.840, -0.801], E3 -0.859 [-0.880, -0.835], E4 -0.873 [-0.904, -0.841] | yes | supported · at N=3 — not yet N=10 |
| H4 | E3 and E4 beat independent retrains on HS (vs. distilled students: open question, no direction) | E3 0.929, E4 0.943; 9 retrains mean 0.119 (max 0.329); 3 students mean 0.233 (max 0.282); vs students, two-sided: E3 not significant (Holm p 0.20; smallest attainable p 0.10; every agent run above every student; post-hoc rule: cannot tell at this N), E4 not significant (Holm p 0.20; smallest attainable p 0.10; every agent run above every student; post-hoc rule: cannot tell at this N) | yes | E3: supported; E4: supported · at N=3 — not yet N=10 |
| H4b | planted model: HS = 1 and KL = 0 at step 0, then HS strictly falls and KL strictly rises | HS 1.000 → 0.998 → 0.957 → 0.888 (step 0 unrounded: 0.999999966590); KL 0.000 → 0.371 → 0.040 → 4.25e-04 | partly (step-zero HS differs from exact identity; KL not strictly rising) | not supported · deterministic |
| H5 | agent ≤ reference compiler on every explanation; the gap shrinks with simulator capability | T0 (preregistered): agent − compiler E2 -0.001, E3 +0.002, E4 -0.009, W-freq -0.024, none of the four explanations beyond the bootstrap CI; seed targets T1, T2, T-new (extension, post hoc): agent above compiler in six of eleven pairs by sign, four of eleven target–explanation pairs on three targets beyond the bootstrap CI (T1 E2, T2 E3, T2 E4, T-new E4; largest +0.179 (T2 E3); margins under 0.01: T1 E2 +0.007, T-new E4 +0.001); T-paper (memorisation-flagged, not counted): E3 +0.073 beyond the CI, E4 -0.046, W-freq -0.004; 5 local runs, 0 above chance | partly (seed targets: agent above compiler beyond its CI; capability trend not testable) | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H6 | the local simulators rank E2, E3, E4, W-freq the same way | 5 local runs, 0 above uniform-guess accuracy. The three-model ranking matrix is incomplete | not testable | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H7 | a T0 replacement fits T0 better than any other trained or no-mechanism model (HS) | E3: HS 0.939 vs best other 0.274 (T2); E3: HS 0.932 vs best other 0.240 (T2); E3: HS 0.915 vs best other 0.290 (T2); E4: HS 0.912 vs best other 0.226 (T2); E4: HS 0.953 vs best other 0.227 (T2); E4: HS 0.965 vs best other 0.258 (T2) | yes | supported · 6 E3/E4 replacements (at N=3 — not yet N=10) |
| H8 | HS against T0's checkpoints rises as restricted loss falls (circuit formation) | Historical round 2: one selected replacement, 41 checkpoints, ρ 0.856. Retrospective full-checkpoint analysis: Spearman ρ(HS, −log restricted loss) per replacement: median 0.774, min 0.751 (6 E3/E4 replacements × 401 checkpoints); negative controls (post hoc), ρ per replacement: E0 -0.83, -0.45, +0.14; W-freq -0.40, -0.24, -0.38; E1 -0.58, -0.58, -0.57; reference networks (post hoc, audit 4): final T0 itself +0.59, Fourier-restricted T0 +0.99, compiler E3 +0.88, a retrain -0.53; during circuit formation (steps ≤ 14,900) every network with T0's frequencies has ρ ≥ 0.996 | yes | supported · 6 E3/E4 replacements × 401 checkpoints |
<!-- END predictions -->

Replications on the seed-sensitivity targets (preregistration §5: H1–H3 re-tested per target with the preregistered tests, deviation 8; first row: T0 restricted to E2–E4 for comparison, post hoc):

<!-- BEGIN replications -->
| Target | Mean HS by true level | H1 increasing HS (p, ρ) | H1 decreasing KL (p, ρ) | H2 E3 − W-freq (Δ HS, p) |
|----------|-------------------------|--------------------|--------------------|--------------------|
| T0 | E2 0.894, E3 0.929, E4 0.943 | 0.011, +0.79 (supported) | 0.999, -0.90 | +0.859, 0.050 |
| T1 | E2 0.957, E3 0.936, E4 0.960 | 0.500, +0.05 | 0.995, -0.79 | +0.885, 0.050 |
| T2 | E2 0.900, E3 0.962, E4 0.951 | 0.500, +0.05 | 1.000, -0.95 | +0.864, 0.050 |
| T-new | E3 0.910, E4 0.984 | 0.050, +0.88 | 0.050, +0.88 | +0.816, 0.050 |
| T-paper | E1 0.590, E3 0.940, E4 0.937 | 0.061, +0.63 | 0.666, -0.11 | +0.823, 0.050 |

H1 uses the available levels. The T0 row restricts the ladder to E2–E4 as a post-hoc comparison. Unmarked H1 cells miss the registered criterion. H2 uses the single displayed contrast on seed targets. The main T0 H2 verdict uses all four contrasts with Holm correction. T-paper is contamination-flagged.

| Target | H3 wrong-frequency agreement, KL, HS | H3 interval and upper-bound readings |
|----------|------------------------------------|-------------------------------------------|
| T0 | agreement 1.000, KL 0.003 (true 2.12e-04–0.060), HS 0.070 | interval passes, upper-bound reading passes |
| T1 | agreement 0.854, KL 0.537 (true 0.001–0.498), HS 0.051 | interval fails, upper-bound reading fails |
| T2 | agreement 0.999, KL 0.006 (true 3.41e-05–0.049), HS 0.098 | interval passes, upper-bound reading passes |
| T-new | agreement 0.907, KL 0.432 (true 0.012–0.328), HS 0.094 | interval fails, upper-bound reading fails (E2 missing; full comparison unavailable) |
| T-paper | agreement 0.998, KL 0.007 (true 0.141–0.413), HS 0.117 | interval fails, upper-bound reading passes (E2 missing; full comparison unavailable) |
<!-- END replications -->

**Reading.** On T0 the registered HS trend passed and the decreasing-KL trend failed. E0 exceeded E1 on HS (<!-- v:e0-vs-e1 -->E0 0.130 vs E1 0.046<!-- /v -->). All four H2 comparisons missed the Holm criterion. For E3 versus W-alg and E4 versus W-scramble the mean-difference intervals span zero, leaving their directions uncertain. H3 met both KL readings on T0. H4 passed against retrains and H7 passed against the fixed wrong-target panel. H4b failed the joint monotonicity check. The stopped simulator grid leaves H5 and H6 unresolved. H8 met its descriptive correlation criterion on a single training trajectory. The published checkpoint satisfies only the upper-bound H3 reading on observed levels. [Dated audit and target-level results](reviews/scientific-audit-2026-11-26.md).

## What this means

- **Output and hidden-state metrics answered different questions.** On T0, students distilled with temperature-one KL attained small KL and low HS. Exact softmax probabilities determine centred logits. A squared-error comparison gives more weight to wrong-class logit discrepancies in this setting. All observed E3 builds had lower scale-adjusted logit error than W-alg (<!-- v:au-fvu -->E3 0.123–0.210, W-alg 0.232–0.232<!-- /v -->). This was a post-hoc comparison. [Hinton et al., §2](https://arxiv.org/pdf/1503.02531).
- **H2 is not supported at N = 3 by its preregistered criterion** (3-vs-3 tests cannot reach α after Holm; a post-hoc
  rule reads "cannot tell at this N"), and **H4b is falsified** (HS fell as predicted; KL followed the planted model's
  dip and recovery in accuracy). Specificity (H7) was shown only against networks with other key frequencies.
- **HS works at the level of frequencies, not below it, and there a weights-only spectrum comparison and a logit-level
  output comparison do about as well.** At N = 3 it did not separate the agents' rebuilds of the
  Clock and the Pizza algorithm with the same frequencies, nor of a correct neuron table and a scrambled one (a literal,
  agent-free rebuild of the scrambled table does lose <!-- v:a5-scr-t0 -->0.09<!-- /v --> on T0, not on the other targets); a neuron-level or
  causal second metric is needed for finer explanations. Post hoc, Zhong et al.'s Clock/Pizza diagnostic shows the blind spot inside the
  true explanations too: T0 has Clock-typical distance irrelevance (<!-- v:cp-t0 -->0.75<!-- /v -->), its E1–E3 rebuilds are Pizza-like
  (<!-- v:cp-e13 -->0.13–0.18<!-- /v -->), and HS still rates them high; the rebuilds that did land on the Clock side
  (<!-- v:a9-cp-clock-builds -->two of three E4 builds, q 0.46, and all three W-scramble builds, q 0.50–0.66<!-- /v -->) score no higher
  (<!-- v:a9-cp-clock-hs -->0.91–0.95<!-- /v --> vs <!-- v:a9-cp-pizza-hs -->0.89–0.97<!-- /v -->; post hoc, audit 9).
- **The agent's repair is part of the score.** Every run checked task accuracy, through the self-test or its own code
  (<!-- v:any-label -->all 96<!-- /v -->); removing the self-test did not remove that feedback, so only the agent-free
  compiler measures an explanation without it.
- **Grade on models whose mechanism the simulator cannot have seen.** The simulator fills gaps in an explanation with what it remembers of the published mechanism
  (cued by the sandbox's description of the paper's exact architecture and, for the algorithm-only explanation, by the paper's own algorithm), which confounds scores of underspecified explanations on famous models (full explanations
  still scored as expected on the published checkpoint).

## Status and next steps

Rounds 1 and 2 are complete (N = 3 runs per condition). The local Qwen capability ablation was stopped after <!-- v:n-local -->five<!-- /v -->
runs, all at chance (user decision; preregistration deviation 5), so H5's capability trend and H6 cannot be tested. Round 2 (N = 3)
ran the 32 Claude Haiku conditions; round 3 (N = 10) is postponed to a later day. Open: a second metric for
sub-frequency structure; ideas for a next iteration (not scheduled) are in [next-steps.md](next-steps.md). Details:
[paper](paper.pdf), [round-1 report](experiments/01-round1/experiment.md), [round-2 report](experiments/02-round2/experiment.md),
[reviews](reviews/round2-review.md), [independent audit](reviews/independent-audit-2026-11-22.md), [second audit](reviews/audit-2-2026-11-22.md), [third audit](reviews/audit-3-2026-11-22.md), [fourth audit](reviews/audit-4-2026-11-22.md), [fifth audit](reviews/audit-5-2026-11-22.md), [sixth audit](reviews/audit-6-2026-11-22.md), [seventh audit](reviews/audit-7-2026-11-22.md), [eighth audit](reviews/audit-8-2026-11-23.md),
[results explorer](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/).
