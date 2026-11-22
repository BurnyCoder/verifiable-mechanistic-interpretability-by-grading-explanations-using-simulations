# Experiment 02 — Round 2 (N = 3 per condition)

*Status: done. Every table below is generated from `results/` by `vmi analyse 2` and shows rounds 1–2 together
(three runs per condition: replicates 0, 1, 2). Round 3 (N = 10) is postponed (preregistration deviation 6); every
verdict is labelled "at N=3 — not yet N=10". Protocol and deviations: [preregistration](../../preregistration.md).*

## What changed from round 1

- **Runs:** 64 more Claude Haiku 5.5 runs (2 per condition, 32 conditions), no failures; no local-model runs
  (deviation 5). The wrong-target control used all six T0 E3/E4 replacements; the H8 trajectory used one
  (`main__T0__E4…s2`, the highest-HS one) against every 10th checkpoint (41 of 401), not every replacement × all 401
  checkpoints as preregistered. After the round it was re-run as preregistered (deviation 8): all six replacements × 401
  checkpoints, every ρ > 0, so H8 stays supported; the H8 row below shows the re-run. The grader sensitivity analysis covers
  all T0 main and control runs.
- **Analysis corrections (deviation 7):** H1 now reports its KL part separately, as §5 preregisters; H2 applies the
  §6 power rule at N = 3. Neither correction turns a verdict into "supported".
- **Not replicated from round 1:** the low-effort E3 run's accuracy collapse (one of three runs); the low-effort
  W-freq run's accuracy collapse (s0 0.770; s1, s2 1.000; `results/runs/hyper__T0__W-freq-effort-low__*`); the E0 run's use of
  the published frequencies (one of three runs); E4 < E3 on T0 (at N = 3 the means are ordered E1 < E2 < E3 < E4).

## Key results

<!-- BEGIN key-results -->
*Round 2, N = 3 run(s) per condition — not yet N=10*

- **Model-specific detail raises hidden-state similarity:** HS on T0 is 0.05 (E1, algorithm only), 0.89 (E2), 0.93 (E3) and 0.94 (E4); with no explanation (E0) it is 0.13. Monotone from E1 to E4; E0 lies above E1, against the preregistered prediction.
- **Correct behaviour from the wrong key frequencies is caught by HS, not by output KL:** the wrong-frequency replacement reaches KL 0.003 nats (true E3: 0.010) but HS 0.07.
- **Copying behaviour does not copy the mechanism:** independent retrains (HS 0.12) and logit-distilled students (HS 0.23, KL 2.38e-06) score no higher than randomly initialised networks (HS 0.24). HS's scale is therefore not anchored at 0 for unrelated networks: it measures overlap of internal features, and working networks built on other frequencies score below random ones.
- **Negative result:** the primary HS does not separate the Pizza explanation (W-alg, HS 0.91) or the scrambled neuron table (W-scramble, HS 0.94) from the true explanations; the secondary matched-neuron correlation separates W-scramble (0.65) from E4 (0.91). Post hoc, by Zhong et al.'s distance irrelevance (≥ 0.4 Clock-like, < 0.4 Pizza-like) T0 is a Clock (0.75) and the literal Pizza build a Pizza (0.003), but the agents' replacements from the true E1–E3 are Pizza-like too (means 0.13–0.18; E4 0.41), and HS still rates the E2–E4 ones 0.89–0.94: HS does not see the Clock/Pizza distinction, even among the true explanations.
- **Memorisation (post hoc):** E0 and E1 contain nothing target-specific (identical text on every target, and the sandbox does not reveal the target), so their runs on T0 and on the published checkpoint are replicates of one condition. When the explanation gives no frequencies, the simulator often builds most of the published paper's frequencies {14, 35, 41, 42, 52} into its replacement. E1 (6 runs on T0 and T-paper): 5 of 6 build scripts list at least four of the paper's five frequencies (3 of 6 replacements build at least four by the key-frequency rule, a lower bound); on average 68% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.08 against T0 but 0.70 against the published checkpoint (target swap: same weights, other target; range 0.24–0.95); E0 (3 runs on T0): 1 of 3 build scripts list at least four of the paper's five frequencies (1 of 3 replacements build at least four by the key-frequency rule, a lower bound); on average 36% of the replacements' W_L spectrum lies on those five frequencies (chance 9%); mean HS 0.13 against T0 but 0.28 against the published checkpoint (target swap: same weights, other target; range 0.03–0.63). On a published model, a high score can therefore come from the simulator's memory rather than the explanation (one published model and one simulator tested).
- **Cost:** the 96 reported Claude Haiku 5.5 runs cost $2.20 ($2.33 including development runs and plumbing checks).
<!-- END key-results -->

## Hypotheses: predicted vs. observed (N = 3)

<!-- BEGIN predictions -->
| Hypothesis | Predicted (preregistration §5) | Measured with the main simulator (T0) | Direction matches? | Formal verdict |
|---|-------|---------|-----|------|
| H1 | HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1 | HS E0 0.130, E1 0.046, E2 0.894, E3 0.929, E4 0.943; KL E1 0.097, E2 2.12e-04, E3 0.010, E4 0.060 | partly (KL does not fall E1→E4; E0 not below E1) | HS trend E1→E4: supported; E0 below E1: not observed (E0 above E1; direction only, no test preregistered); KL_out: not supported · at N=3 — not yet N=10 |
| H2 | true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble | HS difference E3>W-freq +0.859 [+0.835, +0.880]; E3>W-alg +0.020 [-0.008, +0.045]; E3>W-role +0.725 [+0.661, +0.768]; E4>W-scramble +0.007 [-0.025, +0.037] | partly (E3>W-alg tie (CI includes 0); E4>W-scramble tie (CI includes 0)) | E3>W-freq: cannot tell at this N; E3>W-alg: cannot tell at this N; E3>W-role: cannot tell at this N; E4>W-scramble: cannot tell at this N · at N=3 — not yet N=10 |
| H3 | W-freq: argmax agreement ≥ 0.99, KL not above the true E2–E4 range, HS below each of E2–E4 | W-freq: agreement 1.000, KL 0.003 (true 2.12e-04–0.060), HS 0.070 (true 0.894–0.943); HS − E2 -0.824 [-0.840, -0.801], E3 -0.859 [-0.880, -0.835], E4 -0.873 [-0.904, -0.841] | yes | supported · at N=3 — not yet N=10 |
| H4 | E3 and E4 beat independent retrains on HS (vs. distilled students: open question, no direction) | E3 0.929, E4 0.943; 9 retrains mean 0.119 (max 0.329); 3 students mean 0.233 (max 0.282); vs students, two-sided: E3 cannot tell at this N, E4 cannot tell at this N | yes | E3: supported; E4: supported · at N=3 — not yet N=10 |
| H4b | planted model: HS = 1 and KL = 0 at step 0, then HS strictly falls and KL strictly rises | HS 1.000 → 0.998 → 0.957 → 0.888; KL 0.000 → 0.371 → 0.040 → 4.25e-04 | partly (KL not strictly rising) | not supported · deterministic |
| H5 | agent ≤ reference compiler on every explanation; the gap shrinks with simulator capability | T0 (preregistered): agent − compiler E2 -0.001, E3 +0.002, E4 -0.009, W-freq -0.024, none of the four explanations beyond the bootstrap CI; seed targets T1, T2, T-new (extension, post hoc): agent above compiler in six of eleven pairs by sign, four of eleven target–explanation pairs on three targets beyond the bootstrap CI (T1 E2, T2 E3, T2 E4, T-new E4; largest +0.179 (T2 E3)); T-paper (memorisation-flagged, not counted): E3 +0.073 beyond the CI, E4 -0.046, W-freq -0.004; 5 local runs, 0 above chance | partly (seed targets: agent above compiler beyond its CI; capability trend not testable) | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H6 | the local simulators rank E2, E3, E4, W-freq the same way | 5 local runs, 0 above chance: nothing to rank | not testable | not evaluable (local simulators stopped at chance, deviation 5) · 5 local runs, qwen3.5-4b: 0/4 above chance, qwen3.5-9b: 0/1 above chance |
| H7 | a T0 replacement fits T0 better than any other trained or no-mechanism model (HS) | E3: HS 0.939 vs best other 0.274 (T2); E3: HS 0.932 vs best other 0.240 (T2); E3: HS 0.915 vs best other 0.290 (T2); E4: HS 0.912 vs best other 0.226 (T2); E4: HS 0.953 vs best other 0.227 (T2); E4: HS 0.965 vs best other 0.258 (T2) | yes | supported · 6 E3/E4 replacements (at N=3 — not yet N=10) |
| H8 | HS against T0's checkpoints rises as restricted loss falls (circuit formation) | Spearman ρ(HS, −log restricted loss) per replacement: median 0.774, min 0.751 (6 E3/E4 replacements × 401 checkpoints); negative controls (post hoc), ρ per replacement: E0 -0.83, -0.45, +0.14; W-freq -0.40, -0.24, -0.38; E1 -0.58, -0.58, -0.57 | yes | supported · 6 E3/E4 replacements × 401 checkpoints |
<!-- END predictions -->

Replications on the seed-sensitivity targets:

<!-- BEGIN replications -->
| Target | HS per true explanation | H1: HS rises with detail? (trend test) | H2: HS(E3) − HS(W-freq); Mann–Whitney | H3 observed (W-freq) | H3 holds? |
|-----|------|---|---|-------|-----|
| T0 (E2–E4 only; post hoc, for comparison) | E2 0.894, E3 0.929, E4 0.943 | means rise E2→E3→E4; supported (JT p 0.011, ρ 0.79) | +0.859; cannot tell at this N (p 0.050) | agreement 1.000, KL 0.003 (true max 0.060), HS 0.070 | yes |
| T1 | E2 0.957, E3 0.936, E4 0.960 | means do not rise E2→E3→E4; not supported (JT p 0.500, ρ 0.05) | +0.885; cannot tell at this N (p 0.050) | agreement 0.854, KL 0.537 (true max 0.498), HS 0.051 | partly (agreement < 0.99; KL above the true range) |
| T2 | E2 0.900, E3 0.962, E4 0.951 | means do not rise E2→E3→E4; not supported (JT p 0.500, ρ 0.05) | +0.864; cannot tell at this N (p 0.050) | agreement 0.999, KL 0.006 (true max 0.049), HS 0.098 | yes |
| T-new | E3 0.910, E4 0.984 | means rise E3→E4; cannot tell at this N (JT p 0.050, ρ 0.88) | +0.816; cannot tell at this N (p 0.050) | agreement 0.907, KL 0.432 (true max 0.328), HS 0.094 | partly (agreement < 0.99; KL above the true range) |
| T-paper (flagged: memorisation) | E1 0.590, E3 0.940, E4 0.937 | means do not rise E1→E3→E4; not supported (JT p 0.061, ρ 0.63) | +0.823; cannot tell at this N (p 0.050) | agreement 0.998, KL 0.007 (true max 0.413), HS 0.117 | yes |
<!-- END replications -->

**Reading.** With three runs per condition the preregistered tests decide five hypotheses. HS rises with true detail
(H1, HS part: supported); the large step is E1 → E2, E2 → E3 adds a few hundredths in every run (preregistered HS; E2 and E3 tie under the post-hoc whitened check of the
audit) and E3 → E4 is mixed. Output KL does not fall with detail (H1, KL part: not supported) — it is lowest at E2 and rises at E3 and E4.
The wrong-frequency explanation keeps correct behaviour while HS falls below every true explanation that names the frequencies (E2–E4; H3: supported).
E3 and E4 beat independent retrains (H4: supported) and fit T0 better than any other model (H7: supported); HS tracks
circuit formation (H8: supported, descriptive, in its preregistered form over all six
E3/E4 replacements and 401 checkpoints after deviation 8; replacements without T0's mechanism do not, post hoc,
deviation 9). H2 cannot be decided with 3 vs 3 runs (smallest attainable
p = 0.05), but its bootstrap intervals already separate two arms from two others: E3 beats W-freq and W-role by wide
margins, while W-alg (Pizza) and W-scramble tie with the true explanations. W-alg is not length-matched (about 60 % of
E3's characters on every target, `results/explanations/*/W-alg/length.json`), unlike the H2 row's wording. H4b's KL half fails deterministically, as in round 1.

## Main method and agent controls on T0

<!-- BEGIN scoreboard-T0 -->
| Explanation | role | chars | N | agent accuracy | agent KL_out ↓ | agent HS ↑ | agent matched-neuron corr ↑ | compiler KL_out ↓ | compiler HS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| E0 | control: no mechanism | 279 | 3 | 1.000 ± 0.000 | 0.005 ± 0.004 | 0.130 ± 0.109 | 0.056 ± 0.031 | 1.64e-04 | 0.051 |
| E1 | method | 1556 | 3 | 1.000 ± 0.000 | 0.097 ± 0.152 | 0.046 ± 0.002 | 0.028 ± 1.59e-04 | 1.64e-04 | 0.051 |
| E2 | method | 1788 | 3 | 1.000 ± 0.000 | 2.12e-04 ± 1.78e-04 | 0.894 ± 0.005 | 0.669 ± 0.005 | 1.30e-05 | 0.895 |
| E3 | method | 3712 | 3 | 1.000 ± 4.70e-04 | 0.010 ± 0.012 | 0.929 ± 0.012 | 0.759 ± 0.060 | 4.25e-04 | 0.927 |
| E4 | method | 51329 | 3 | 0.986 ± 0.012 | 0.060 ± 0.045 | 0.943 ± 0.028 | 0.913 ± 0.044 | 0.010 | 0.952 |
| W-freq | control: wrong frequencies | 3724 | 3 | 1.000 ± 9.04e-05 | 0.003 ± 0.002 | 0.070 ± 0.021 | 0.030 ± 1.64e-04 | 0.001 | 0.095 |
| W-alg | control: wrong algorithm (Pizza) | 2224 | 3 | 1.000 ± 0.000 | 0.024 ± 0.017 | 0.909 ± 0.027 | 0.737 ± 0.061 | 0.040 | 0.921 |
| W-role | control: wrong component roles | 3777 | 3 | 1.000 ± 2.71e-04 | 9.80e-04 ± 7.60e-04 | 0.204 ± 0.060 | 0.013 ± 0.002 | 4.727 | 0.500 |
| W-scramble | control: scrambled neurons | 51590 | 3 | 0.998 ± 0.004 | 0.006 ± 0.008 | 0.936 ± 0.018 | 0.653 ± 0.027 | 2.02e-05 | 0.859 |
<!-- END scoreboard-T0 -->

## No-agent controls

<!-- BEGIN controls-T0 -->
| Planted model trained further (steps) | HS ↑ | KL_out | snapshot task accuracy |
|---|---|---|---|
| 0 | 1.000 | 0.000 | 1.0000 |
| 100 | 0.998 | 0.371 | 0.9774 |
| 1000 | 0.957 | 0.040 | 0.9958 |
| 10000 | 0.888 | 4.25e-04 | 1.0000 |

| Replacement | HS vs T0 | best wrong target (HS) | d = HS(T0) − max HS(wrong) |
|---|---|---|---|
| E3 | 0.939 | T2 (0.274) | 0.665 |
| E3 | 0.932 | T2 (0.240) | 0.693 |
| E3 | 0.915 | T2 (0.290) | 0.626 |
| E4 | 0.912 | T2 (0.226) | 0.685 |
| E4 | 0.953 | T2 (0.227) | 0.726 |
| E4 | 0.965 | T2 (0.258) | 0.708 |
<!-- END controls-T0 -->

## Baselines (no agent)

<!-- BEGIN baselines-T0 -->
| Reference point | what it has | n | task accuracy | KL_out ↓ | HS ↑ |
|---|---|---|---|---|---|
| Random initialisation | nothing (floor) | 5 | 0.009 ± 0.001 | 4.766 ± 0.006 | 0.236 ± 0.006 |
| Independent retrains | same recipe, other seeds | 9 | 1.000 ± 0.000 | 6.90e-06 ± 3.96e-06 | 0.119 ± 0.097 |
| Logit-distilled students | M's full output logits, no explanation | 3 | 1.000 ± 0.000 | 2.38e-06 ± 8.25e-07 | 0.233 ± 0.058 |
| Fourier-restricted M | M's weights, key frequencies only in W_E/W_U | 1 | 1.000 | 0.002 | 0.992 |
| Weights dump | all 226,816 weights | 1 | 1.000 | 0.000 | 1.000 |
<!-- END baselines-T0 -->

## Ablations and hyperparameter variants

<!-- BEGIN ablations-T0 -->
| Change | Explanation | Simulator | N | task accuracy | KL_out ↓ | HS ↑ | matched-neuron corr ↑ |
|---|---|---|---|---|---|---|---|
| none (main condition) | E3 | claude-haiku-5-5 | 3 | 1.000 ± 4.70e-04 | 0.010 ± 0.012 | 0.929 ± 0.012 | 0.759 ± 0.060 |
| explanation as maths | E3 | claude-haiku-5-5 | 3 | 0.999 ± 0.002 | 0.003 ± 0.004 | 0.915 ± 0.003 | 0.771 ± 0.036 |
| explanation as code | E3 | claude-haiku-5-5 | 3 | 1.000 ± 9.04e-05 | 0.002 ± 3.96e-04 | 0.897 ± 0.030 | 0.805 ± 2.48e-04 |
| self-test removed | E3 | claude-haiku-5-5 | 3 | 1.000 ± 3.62e-04 | 0.004 ± 0.002 | 0.942 ± 0.025 | 0.765 ± 0.043 |
| effort low | E3 | claude-haiku-5-5 | 3 | 0.853 ± 0.252 | 0.661 ± 1.137 | 0.921 ± 0.003 | 0.751 ± 0.027 |
| effort high | E3 | claude-haiku-5-5 | 3 | 1.000 ± 0.000 | 0.001 ± 7.17e-04 | 0.936 ± 0.034 | 0.791 ± 0.025 |
| none (main condition) | W-freq | claude-haiku-5-5 | 3 | 1.000 ± 9.04e-05 | 0.003 ± 0.002 | 0.070 ± 0.021 | 0.030 ± 1.64e-04 |
| effort low | W-freq | claude-haiku-5-5 | 3 | 0.923 ± 0.133 | 0.344 ± 0.594 | 0.088 ± 0.025 | 0.061 ± 0.031 |
| effort high | W-freq | claude-haiku-5-5 | 3 | 1.000 ± 0.000 | 0.001 ± 6.05e-04 | 0.093 ± 0.005 | 0.032 ± 0.003 |
| simulator qwen3.5-4b | E2 | qwen3.5-4b | 1 | 0.009 | 4.727 | 0.000 | 0.000 |
| simulator qwen3.5-4b | E3 | qwen3.5-4b | 1 | 0.009 | 4.727 | 0.000 | 0.000 |
| simulator qwen3.5-4b | E4 | qwen3.5-4b | 1 | 0.009 | 4.727 | 0.000 | 0.000 |
| simulator qwen3.5-4b | W-freq | qwen3.5-4b | 1 | 0.009 | 6.066 | 0.000 | 1.41e-16 |
| simulator qwen3.5-9b | E2 | qwen3.5-9b | 1 | 0.009 | 7.628 | 0.000 | 0.000 |
<!-- END ablations-T0 -->

**Reading.** Format (maths, code), removing the self-test and effort level move HS by a few hundredths at most.
All three runs without the self-test measured task accuracy with their own code (audit 3), so that ablation removed
the script and the instruction to raise accuracy, not the label feedback. The
round-1 low-effort collapse was one run in three, and its cause is a layout bug in that run's own script (one
head's output written onto the readout dimensions; fixing that one number restores full accuracy, `vmi diagnose`).

## Memorisation check (post hoc)

<!-- BEGIN frequencies -->
| Target | Explanation | frequencies built (key-frequency rule) | frequencies the script lists | target's key frequencies | W_L norm share: target's / paper's frequencies | HS | E0/E1, same weights: HS vs T0 / vs published checkpoint |
|---|---|---|---|---|---|---|---|
| T0 | E0 | 14, 35, 41, 52 | 14, 35, 41, 52 | 2, 3, 23, 37, 39 | 0.00 / 1.00 | 0.004 | 0.004 / 0.626 |
| T0 | E0 | 9, 10, 11, 12, 13, 14, 15, 16 | computed | 2, 3, 23, 37, 39 | 0.01 / 0.07 | 0.191 | 0.191 / 0.189 |
| T0 | E0 | 1, 2, 3, 4, 5, 6, 7, 8 | computed | 2, 3, 23, 37, 39 | 0.25 / 0.00 | 0.194 | 0.194 / 0.027 |
| T0 | E1 | 14, 35, 41, 52 | 14, 35, 41, 42, 52 | 2, 3, 23, 37, 39 | 0.00 / 1.00 | 0.048 | 0.048 / 0.951 |
| T0 | E1 | 14, 35, 40, 41, 52 | 14, 35, 40, 41, 52 | 2, 3, 23, 37, 39 | 0.00 / 0.79 | 0.045 | 0.045 / 0.751 |
| T0 | E1 | 14, 35, 40, 41, 52 | 14, 35, 40, 41, 52 | 2, 3, 23, 37, 39 | 0.00 / 0.78 | 0.046 | 0.046 / 0.755 |
| T0 | E2 | 3, 23, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.899 | – |
| T0 | E2 | 3, 23, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.895 | – |
| T0 | E2 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.889 | – |
| T0 | E3 | 3, 23 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.939 | – |
| T0 | E3 | 3, 23, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.932 | – |
| T0 | E3 | 3, 23 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.915 | – |
| T0 | E4 | 3, 23, 39 | computed | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.912 | – |
| T0 | E4 | 3, 23, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.953 | – |
| T0 | E4 | 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.965 | – |
| T0 | W-freq | 31, 42, 52, 55 | 22, 31, 42, 52, 55 | 2, 3, 23, 37, 39 | 0.00 / 0.48 | 0.055 | – |
| T0 | W-freq | 31, 42, 55 | 22, 31, 42, 52, 55 | 2, 3, 23, 37, 39 | 0.00 / 0.58 | 0.061 | – |
| T0 | W-freq | 31, 42, 55 | 22, 31, 42, 52, 55 | 2, 3, 23, 37, 39 | 0.00 / 0.33 | 0.094 | – |
| T0 | W-alg | 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.889 | – |
| T0 | W-alg | 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.899 | – |
| T0 | W-alg | 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.940 | – |
| T0 | W-role | 32, 34, 41, 54 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 0.00 / 0.13 | 0.168 | – |
| T0 | W-role | 32, 34, 41, 54 | computed | 2, 3, 23, 37, 39 | 0.00 / 0.08 | 0.171 | – |
| T0 | W-role | 32, 34, 41, 54 | 2, 3, 22, 23, 32, 34, 37, 39, 41, 54 | 2, 3, 23, 37, 39 | 0.00 / 0.03 | 0.273 | – |
| T0 | W-scramble | 3, 23, 39 | 2, 3, 23, 37, 39 | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.918 | – |
| T0 | W-scramble | 3, 23, 39 | computed | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.954 | – |
| T0 | W-scramble | 3, 23, 39 | computed | 2, 3, 23, 37, 39 | 1.00 / 0.00 | 0.938 | – |
| T1 | E2 | 11, 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.955 | – |
| T1 | E2 | 11, 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.961 | – |
| T1 | E2 | 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.956 | – |
| T1 | E3 | 11, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.917 | – |
| T1 | E3 | 11, 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.943 | – |
| T1 | E3 | 11, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.948 | – |
| T1 | E4 | 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.955 | – |
| T1 | E4 | 25, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.966 | – |
| T1 | E4 | 11, 49 | 11, 25, 49 | 11, 25, 49 | 1.00 / 0.00 | 0.960 | – |
| T1 | W-freq | 38, 52 | 15, 38, 52 | 11, 25, 49 | 0.00 / 0.66 | 0.034 | – |
| T1 | W-freq | 15, 52 | 15, 38, 52 | 11, 25, 49 | 0.00 / 0.18 | 0.079 | – |
| T1 | W-freq | 15, 38, 52 | 15, 38, 52 | 11, 25, 49 | 0.00 / 0.73 | 0.039 | – |
| T2 | E2 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.949 | – |
| T2 | E2 | 37, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.789 | – |
| T2 | E2 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.963 | – |
| T2 | E3 | 37, 45, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.944 | – |
| T2 | E3 | 45, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.970 | – |
| T2 | E3 | 45, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.970 | – |
| T2 | E4 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.934 | – |
| T2 | E4 | 46, 47 | computed | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.955 | – |
| T2 | E4 | 46 | 37, 44, 45, 46, 47, 51 | 37, 44, 45, 46, 47, 51 | 1.00 / 0.00 | 0.962 | – |
| T2 | W-freq | 32 | computed | 37, 44, 45, 46, 47, 51 | 0.00 / 0.00 | 0.089 | – |
| T2 | W-freq | 9, 34 | computed | 37, 44, 45, 46, 47, 51 | 0.00 / 0.00 | 0.093 | – |
| T2 | W-freq | 9, 34 | 4, 8, 9, 11, 32, 34 | 37, 44, 45, 46, 47, 51 | 0.00 / 0.00 | 0.111 | – |
| T-new | E3 | 22, 41 | 22, 37, 41, 51 | 22, 37, 41, 51 | 1.00 / 0.85 | 0.846 | – |
| T-new | E3 | 22, 37, 41, 51 | 22, 37, 41, 51 | 22, 37, 41, 51 | 1.00 / 0.24 | 0.916 | – |
| T-new | E3 | 22, 37, 41, 51 | 22, 37, 41, 51 | 22, 37, 41, 51 | 1.00 / 0.29 | 0.968 | – |
| T-new | E4 | 37, 41 | 22, 37, 41, 51 | 22, 37, 41, 51 | 1.00 / 0.83 | 0.984 | – |
| T-new | E4 | 22, 37, 41, 51 | computed | 22, 37, 41, 51 | 1.00 / 0.34 | 0.984 | – |
| T-new | E4 | 37, 41 | computed | 22, 37, 41, 51 | 1.00 / 0.71 | 0.984 | – |
| T-new | W-freq | 10 | 10, 25, 33, 39 | 22, 37, 41, 51 | 0.00 / 0.00 | 0.087 | – |
| T-new | W-freq | 10, 25, 33 | 10, 25, 33, 39 | 22, 37, 41, 51 | 0.00 / 0.00 | 0.029 | – |
| T-new | W-freq | 10, 25, 33 | 10, 25, 33, 39 | 22, 37, 41, 51 | 0.00 / 0.00 | 0.166 | – |
| T-paper | E1 | 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.925 | 0.006 / 0.925 |
| T-paper | E1 | 3, 14, 17, 25, 41, 46, 52 | 3, 14, 17, 25, 35, 41, 46, 52 | 14, 35, 41, 42, 52 | 0.53 / 0.53 | 0.600 | 0.276 / 0.600 |
| T-paper | E1 | 30, 34, 44, 46 | computed | 14, 35, 41, 42, 52 | 0.01 / 0.01 | 0.244 | 0.047 / 0.244 |
| T-paper | E3 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.944 | – |
| T-paper | E3 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.963 | – |
| T-paper | E3 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.913 | – |
| T-paper | E4 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.929 | – |
| T-paper | E4 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.945 | – |
| T-paper | E4 | 35, 41 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.938 | – |
| T-paper | W-freq | 8, 21, 39, 44, 49 | computed | 14, 35, 41, 42, 52 | 0.00 / 0.00 | 0.091 | – |
| T-paper | W-freq | 8, 21, 39, 44, 49 | computed | 14, 35, 41, 42, 52 | 0.00 / 0.00 | 0.099 | – |
| T-paper | W-freq | 8, 21, 39, 44, 49 | computed | 14, 35, 41, 42, 52 | 0.00 / 0.00 | 0.161 | – |
| T-paper | E-paper | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.984 | – |
| T-paper | E-paper | 14, 35, 41, 42, 52 | computed | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.904 | – |
| T-paper | E-paper | 14, 35, 41, 42, 52 | computed | 14, 35, 41, 42, 52 | 1.00 / 1.00 | 0.961 | – |
<!-- END frequencies -->

**Reading.** E0 and E1 are the same text on every target and the sandbox does not reveal the target, so the E1 runs on
T0 and on the published checkpoint are replicates of one condition (pooled since the second audit; the key results
above give the counts). Most E1 build scripts hard-code at least four of the paper's five frequencies (e.g.
`[14, 35, 41, 52, 73]`, 73 ≡ −40 mod 113), one picks frequencies by a random search, and the same weights score far
higher against the published checkpoint than against T0 (target swap). With no explanation at all (E0) this happened in
one run of three; the other two built blocks of consecutive frequencies and score near the random-initialisation floor.
On the published checkpoint, scores of underspecified explanations measure memory, not the explanation.

## Grader sensitivity (no new runs)

<!-- BEGIN grader-sensitivity-T0 -->
| Grader setting | E0 | E1 | E2 | E3 | E4 | W-freq | W-alg | W-role | W-scramble | retrain | random init | Kendall τ vs. preregistered |
|--------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-----------|-------|-----------|-----------|
| preregistered (λ 1e-6, 5 folds, seed 0, 3 sites, general linear) | 0.13 | 0.05 | 0.89 | 0.93 | 0.94 | 0.07 | 0.91 | 0.20 | 0.94 | 0.12 | 0.24 | 1.00 |
| ridge λ × 100 | 0.11 | 0.03 | 0.87 | 0.90 | 0.93 | 0.06 | 0.89 | 0.19 | 0.92 | 0.09 | 0.21 | 1.00 |
| ridge λ / 100 | 0.14 | 0.05 | 0.91 | 0.94 | 0.95 | 0.08 | 0.92 | 0.22 | 0.94 | 0.14 | 0.26 | 1.00 |
| 3 folds | 0.13 | 0.04 | 0.89 | 0.93 | 0.94 | 0.07 | 0.91 | 0.20 | 0.94 | 0.12 | 0.23 | 1.00 |
| 10 folds | 0.13 | 0.05 | 0.89 | 0.93 | 0.94 | 0.07 | 0.91 | 0.21 | 0.94 | 0.12 | 0.24 | 1.00 |
| fold seed 1 | 0.13 | 0.05 | 0.89 | 0.93 | 0.94 | 0.07 | 0.91 | 0.20 | 0.94 | 0.12 | 0.24 | 1.00 |
| fold seed 2 | 0.13 | 0.05 | 0.89 | 0.93 | 0.94 | 0.07 | 0.91 | 0.20 | 0.94 | 0.12 | 0.24 | 1.00 |
| resid_mid only | 0.10 | 0.01 | 0.88 | 0.92 | 0.92 | 0.03 | 0.90 | 0.41 | 0.91 | 0.09 | 0.26 | 0.89 |
| mlp_post only | 0.17 | 0.13 | 0.94 | 0.97 | 0.99 | 0.18 | 0.96 | 0.11 | 0.97 | 0.21 | 0.39 | 0.78 |
| resid_post only | 0.11 | 0.00 | 0.85 | 0.90 | 0.92 | 0.00 | 0.86 | 0.09 | 0.93 | 0.06 | 0.06 | 0.89 |
| orthogonal alignment (with scale) | 0.01 | 0.00 | 0.64 | 0.74 | 0.81 | 0.00 | 0.66 | 0.08 | 0.78 | 0.00 | 0.04 | 0.99 |
| whitened soundness, eig > 1e-6·max (post hoc) | 0.10 | 0.04 | 0.76 | 0.76 | 0.81 | 0.07 | 0.80 | 0.26 | 0.75 | 0.10 | 0.14 | 0.72 |
| whitened soundness, eig > 1e-3·max (post hoc) | 0.12 | 0.04 | 0.88 | 0.88 | 0.90 | 0.07 | 0.90 | 0.25 | 0.88 | 0.13 | 0.18 | 0.78 |

*H7 margin d = HS(T0) − max HS over the 12 wrong targets, per E3/E4 replacement (smallest / median):* preregistered (λ 1e-6, 5 folds, seed 0, 3 sites, general linear): 0.63 / 0.69; whitened soundness, eig > 1e-6·max (post hoc): 0.52 / 0.60; whitened soundness, eig > 1e-3·max (post hoc): 0.60 / 0.67
<!-- END grader-sensitivity-T0 -->

**Reading.** The ranking of explanations is unchanged by ridge strength, number of folds and fold seed, nearly
unchanged under orthogonal alignment, and less stable when only one site is used or when soundness is whitened
(post hoc): whitening ties E2 with E3 and moves W-alg above E3, while the coarse pattern (W-freq, E0, E1, W-role at the
bottom) stays.

## All agent runs, rounds 1–2

<!-- BEGIN runs -->
| Stage | Target | Explanation | Simulator | T | Variant | N | failed | accuracy | KL_out ↓ (mean ± sd) | HS ↑ (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ablations | T0 | E2 | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 4.73 | 0.000 |
| ablations | T0 | E2 | qwen3.5-9b | 0.6 | – | 1 | 0 | 0.009 | 7.63 | 0.000 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-code | 3 | 0 | 1.000 | 0.00175 ± 0.000396 | 0.897 ± 0.0301 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-math | 3 | 0 | 0.999 | 0.00322 ± 0.00379 | 0.915 ± 0.00311 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | noselftest | 3 | 0 | 1.000 | 0.00376 ± 0.00218 | 0.942 ± 0.0253 |
| ablations | T0 | E3 | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 4.73 | 0.000 |
| ablations | T0 | E4 | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 4.73 | 0.000 |
| ablations | T0 | W-freq | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 6.07 | 0.000 |
| controls | T0 | E0 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00455 ± 0.00431 | 0.130 ± 0.109 |
| controls | T0 | W-freq | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00293 ± 0.00195 | 0.070 ± 0.0211 |
| controls | T0 | W-alg | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.024 ± 0.0171 | 0.909 ± 0.0268 |
| controls | T0 | W-role | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00098 ± 0.00076 | 0.204 ± 0.06 |
| controls | T0 | W-scramble | claude-haiku-5-5 | default | – | 3 | 0 | 0.998 | 0.0062 ± 0.00838 | 0.936 ± 0.0183 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-high | 3 | 0 | 1.000 | 0.00102 ± 0.000717 | 0.936 ± 0.0344 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-low | 3 | 0 | 0.853 | 0.661 ± 1.14 | 0.921 ± 0.0033 |
| hyper | T0 | W-freq | claude-haiku-5-5 | default | effort-high | 3 | 0 | 1.000 | 0.00143 ± 0.000605 | 0.093 ± 0.00464 |
| hyper | T0 | W-freq | claude-haiku-5-5 | default | effort-low | 3 | 0 | 0.923 | 0.344 ± 0.594 | 0.088 ± 0.0252 |
| main | T0 | E1 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.0969 ± 0.152 | 0.046 ± 0.00177 |
| main | T0 | E2 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.000212 ± 0.000178 | 0.894 ± 0.00542 |
| main | T0 | E3 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00972 ± 0.0115 | 0.929 ± 0.0124 |
| main | T0 | E4 | claude-haiku-5-5 | default | – | 3 | 0 | 0.986 | 0.06 ± 0.0455 | 0.943 ± 0.0282 |
| seed | T-new | E3 | claude-haiku-5-5 | default | – | 3 | 0 | 0.910 | 0.328 ± 0.237 | 0.910 ± 0.0613 |
| seed | T-new | E4 | claude-haiku-5-5 | default | – | 3 | 0 | 0.997 | 0.0124 ± 0.0214 | 0.984 ± 0.000181 |
| seed | T-new | W-freq | claude-haiku-5-5 | default | – | 3 | 0 | 0.907 | 0.432 ± 0.461 | 0.094 ± 0.0689 |
| seed | T-paper | E1 | claude-haiku-5-5 | default | – | 3 | 0 | 0.998 | 0.621 ± 0.725 | 0.590 ± 0.34 |
| seed | T-paper | E3 | claude-haiku-5-5 | default | – | 3 | 0 | 0.962 | 0.141 ± 0.207 | 0.940 ± 0.0251 |
| seed | T-paper | E4 | claude-haiku-5-5 | default | – | 3 | 0 | 0.894 | 0.413 ± 0.18 | 0.937 ± 0.00801 |
| seed | T-paper | W-freq | claude-haiku-5-5 | default | – | 3 | 0 | 0.998 | 0.00677 ± 0.0116 | 0.117 ± 0.0384 |
| seed | T-paper | E-paper | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00115 ± 0.00196 | 0.950 ± 0.041 |
| seed | T1 | E2 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 0.00127 ± 0.00161 | 0.957 ± 0.00305 |
| seed | T1 | E3 | claude-haiku-5-5 | default | – | 3 | 0 | 0.908 | 0.384 ± 0.143 | 0.936 ± 0.0167 |
| seed | T1 | E4 | claude-haiku-5-5 | default | – | 3 | 0 | 0.901 | 0.498 ± 0.245 | 0.960 ± 0.00558 |
| seed | T1 | W-freq | claude-haiku-5-5 | default | – | 3 | 0 | 0.854 | 0.537 ± 0.136 | 0.051 ± 0.0247 |
| seed | T2 | E2 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 3.41e-05 ± 1.17e-05 | 0.900 ± 0.0969 |
| seed | T2 | E3 | claude-haiku-5-5 | default | – | 3 | 0 | 1.000 | 5.84e-05 ± 1.37e-05 | 0.962 ± 0.015 |
| seed | T2 | E4 | claude-haiku-5-5 | default | – | 3 | 0 | 0.990 | 0.0487 ± 0.0548 | 0.951 ± 0.0146 |
| seed | T2 | W-freq | claude-haiku-5-5 | default | – | 3 | 0 | 0.999 | 0.00563 ± 0.00588 | 0.098 ± 0.0113 |
<!-- END runs -->

## Deviations in this round

Deviation 6 (round 3 postponed, user decision), deviation 7 (two analysis-code corrections to match §5–§6), deviation 8
(the first audit's missing preregistered statistics and invariance erratum) and deviation 9 (the second audit's errata
and post-hoc checks) are logged at the end of the preregistration.
