# Independent scientific audit of the simulation grader

26 November 2026 · Libor “Burny” Burian

The main target shows an increasing hidden-state similarity (HS) trend across the scripted explanation ladder. Its output-KL trend fails the registered criterion. The evidence supports a frequency-sensitive reconstruction score on this task. It leaves finer algorithm discrimination, causal identification and transfer to new operand pairs unresolved. Some published statements about replication and metric validity exceeded the evidence.

## What was checked

We started from the tagged protocol, manifests, per-run directional site scores, reference grades and checkpoint series. The independent analysis enumerates rank-test assignments, reconstructs the bootstrap intervals, and uses the two condition means specified by each contrast. It uses no earlier audit verdicts. The registered tests treat an agent build as the replicate. Grading all inputs exhausts a finite task grid. These inputs are neither independent simulator replicates nor a sample of tasks. The production reports are compared with this second implementation. [Frozen protocol, §§4–8](https://github.com/BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/blob/801edf013db7239f8994e6780c5873e1890f5103/reports/preregistration.md) · [Run manifests](../../results/manifests) · [Raw grades](../../results/runs) · [Independent arithmetic](../../src/vmi/analysis/audit_statistics.py) · [All estimates, intervals and run IDs](../../results/scientific-audit/reconstruction.json).

<!-- BEGIN reconstructed-evidence -->
The manifests contain 108 slots. We matched 101 unique non-development grades, excluded 15 development records and found 7 unfilled local slots. The 1 archived infrastructure failure shares its ID with a successful retry. The included cohort has 0 failure grades. Recombining the saved directional scores at each site into HS gave a maximum absolute difference of 1.11e-16.

| Round | Included run IDs | H1 HS on T0 | H1 decreasing KL on T0 | E0 minus E1 (HS, KL) |
|---|---:|---|---|---|
| 1 | 37 | not supported, J=5, p=4/24 (0.166667), ρ=0.800, CI unavailable | not supported, J=1, p=23/24 (0.958333), ρ=-0.800, CI unavailable | -0.0447, +0.0090 |
| 2 | 101 | supported, J=51, p=35/369600 (9.4697e-05), ρ=0.907, CI [0.787, 0.987] | not supported, J=17, p=344077/369600 (0.930944), ρ=-0.324, CI [-0.871, 0.197] | +0.0831, -0.0923 |

Positive values of ρ indicate the predicted trend for both endpoints. The KL analysis ranks negative KL. E0 is a directional check outside the E1–E4 trend. A single observation per level yields no within-level uncertainty estimate.

### Replications on each available ladder

| Round | Target | Measured E levels | H1 HS p / ρ | H1 decreasing KL p / ρ | H3 W-freq agreement, KL, true KL range | H3 interval / upper reading |
|---|---|---|---|---|---|---|
| 1 | T0 | E1, E2, E3, E4 | 0.16667 / +0.800 | 0.95833 / -0.800 | 1.00000, 0.000931, [0.000012, 0.087197] | passes / passes |
| 1 | T1 | E2, E3, E4 | 0.83333 / -0.500 | 1 / -1.000 | 0.80625, 0.668767, [0.003079, 0.772810] | fails / fails |
| 1 | T2 | E2, E3, E4 | 1 / -1.000 | 1 / -1.000 | 1.00000, 0.004780, [0.000047, 0.111952] | passes / passes |
| 1 | T-new | E3, E4 | 0.5 / +1.000 | 0.5 / +1.000 | 0.89678, 0.377838, [0.000022, 0.343712] | fails / fails on observed levels only, E2 missing |
| 1 | T-paper | E1, E3, E4 | 0.5 / +0.500 | 0.5 / +0.500 | 1.00000, 0.000096, [0.032060, 0.518206] | fails / passes on observed levels only, E2 missing |
| 2 | T0 | E1, E2, E3, E4 | 9.4697e-05 / +0.907 | 0.93094 / -0.324 | 0.99995, 0.002932, [0.000212, 0.060045] | passes / passes |
| 2 | T1 | E2, E3, E4 | 0.5 / +0.053 | 0.99524 / -0.791 | 0.85433, 0.537208, [0.001271, 0.498475] | fails / fails |
| 2 | T2 | E2, E3, E4 | 0.5 / +0.053 | 1 / -0.949 | 0.99869, 0.005627, [0.000034, 0.048712] | passes / passes |
| 2 | T-new | E3, E4 | 0.05 / +0.878 | 0.05 / +0.878 | 0.90743, 0.431755, [0.012370, 0.328050] | fails / fails on observed levels only, E2 missing |
| 2 | T-paper | E1, E3, E4 | 0.06131 / +0.632 | 0.66607 / -0.105 | 0.99822, 0.006775, [0.140572, 0.413357] | fails / passes on observed levels only, E2 missing |

The primary H3 prediction requires agreement ≥0.99, KL inside the range of the three true condition means, and HS below each. Its falsifier names only the upper KL boundary and the HS comparison. We retain the agreement check for both tabulated readings. Missing levels restrict both to observed arms. The JSON also gives each W-freq minus true HS difference and its interval.

### H2 discrimination

| Round | Target | True minus control | Δ mean HS | Exact bootstrap percentile interval | Rank-biserial r | Raw p | Holm p | Registered verdict |
|---|---|---|---:|---|---:|---:|---:|---|
| 1 | T0 | E3 − W-freq | +0.8841 | unavailable | +1.000 | 0.5 | 1 | cannot tell at this N |
| 1 | T0 | E3 − W-alg | +0.0506 | unavailable | +1.000 | 0.5 | 1 | cannot tell at this N |
| 1 | T0 | E3 − W-role | +0.7714 | unavailable | +1.000 | 0.5 | 1 | cannot tell at this N |
| 1 | T0 | E4 − W-scramble | -0.0061 | unavailable | -1.000 | 1 | 1 | cannot tell at this N |
| 1 | T1 | E3 − W-freq | +0.8830 | unavailable | +1.000 | 0.5 | 0.5 | cannot tell at this N |
| 1 | T2 | E3 − W-freq | +0.8549 | unavailable | +1.000 | 0.5 | 0.5 | cannot tell at this N |
| 1 | T-new | E3 − W-freq | +0.7589 | unavailable | +1.000 | 0.5 | 0.5 | cannot tell at this N |
| 1 | T-paper | E3 − W-freq | +0.8525 | unavailable | +1.000 | 0.5 | 0.5 | cannot tell at this N |
| 2 | T0 | E3 − W-freq | +0.8587 | [0.835, 0.880] | +1.000 | 0.05 | 0.2 | not supported |
| 2 | T0 | E3 − W-alg | +0.0200 | [-0.010, 0.045] | +0.333 | 0.35 | 0.7 | not supported |
| 2 | T0 | E3 − W-role | +0.7251 | [0.658, 0.768] | +1.000 | 0.05 | 0.2 | not supported |
| 2 | T0 | E4 − W-scramble | +0.0068 | [-0.025, 0.037] | +0.111 | 0.5 | 0.7 | not supported |
| 2 | T1 | E3 − W-freq | +0.8853 | [0.857, 0.911] | +1.000 | 0.05 | 0.05 | not supported |
| 2 | T2 | E3 − W-freq | +0.8641 | [0.847, 0.880] | +1.000 | 0.05 | 0.05 | not supported |
| 2 | T-new | E3 − W-freq | +0.8159 | [0.729, 0.902] | +1.000 | 0.05 | 0.05 | not supported |
| 2 | T-paper | E3 − W-freq | +0.8228 | [0.779, 0.863] | +1.000 | 0.05 | 0.05 | not supported |

At one replicate §6 prescribes “cannot tell” for every H2, H4 and H7 contrast. Enumerated probabilities in this report show what the ranks do. T0 has four comparisons. Each seed target ran only the wrong-frequency control, so the remaining H2 comparisons are missing. At three versus three the minimum one-sided untied p is 1/20. Section 5 requires p strictly below 0.05. The later “cannot tell” label for a test with inadequate resolution is post hoc. Difference intervals above enumerate bootstrap support. The production report uses the registered 10,000 resamples and can differ by one or more discrete quantiles.

### Baselines, calibration, specificity and trajectory

| Round | Hypothesis | Result from saved measurements | Verdict and evidence limit |
|---|---|---|---|
| 1 | H4, retrains | E3 r=+1.00, p=0.1, Holm p=0.2, E4 r=+1.00, p=0.1, Holm p=0.2 | E3 cannot tell at this N, E4 cannot tell at this N. Fixed same-recipe reference seeds. |
| 1 | H4, students | E3 r=+1.00, p=0.5, Holm p=1, E4 r=+1.00, p=0.5, Holm p=1 | E3 cannot tell at this N, E4 cannot tell at this N. Two-sided secondary comparison with temperature-one probability students. |
| 1 | H4b | Step-zero HS 0.999999966590. KL sequence 0, 0.370622, 0.0401094, 0.000425171 | Failed. HS falls across snapshots. KL peaks and then falls. One planted series cannot validate calibration under other perturbations. |
| 1 | H5 | T0 agent minus compiler HS, E2 +0.0046, E3 +0.0129, E4 -0.0403, W-freq -0.0392 | Not evaluable for capability. Only 5/12 local cells completed. |
| 1 | H6 | 5/12 local cells, all observed local HS zero | Not evaluable. The stopped cells leave missing comparisons and tied observed rankings. |
| 1 | H7 | 2 margins over 12 fixed comparators, +0.6652 to +0.6854, signed-rank p=0.25 | cannot tell at this N. Specificity conditional on this low-overlap target panel. |
| 1 | H8 | 2 builds × 401 checkpoints, median ρ=0.7863, minimum 0.7523 | supported (descriptive). Successive checkpoints share a training path. |
| 2 | H4, retrains | E3 r=+1.00, p=0.0045455, Holm p=0.0090909, E4 r=+1.00, p=0.0045455, Holm p=0.0090909 | E3 supported, E4 supported. Fixed same-recipe reference seeds. |
| 2 | H4, students | E3 r=+1.00, p=0.1, Holm p=0.2, E4 r=+1.00, p=0.1, Holm p=0.2 | E3 not supported, E4 not supported. Two-sided secondary comparison with temperature-one probability students. |
| 2 | H4b | Step-zero HS 0.999999966590. KL sequence 0, 0.370622, 0.0401094, 0.000425171 | Failed. HS falls across snapshots. KL peaks and then falls. One planted series cannot validate calibration under other perturbations. |
| 2 | H5 | T0 agent minus compiler HS, E2 -0.0006, E3 +0.0025, E4 -0.0085, W-freq -0.0242 | Not evaluable for capability. Only 5/12 local cells completed. |
| 2 | H6 | 5/12 local cells, all observed local HS zero | Not evaluable. The stopped cells leave missing comparisons and tied observed rankings. |
| 2 | H7 | 6 margins over 12 fixed comparators, +0.6258 to +0.7263, signed-rank p=0.015625 | supported. Specificity conditional on this low-overlap target panel. |
| 2 | H8 | 6 builds × 401 checkpoints, median ρ=0.7739, minimum 0.7508 | supported (descriptive). Successive checkpoints share a training path. |

The original round-one analysis used the selected E3 seed-zero replacement over 41 sampled checkpoints (ρ=0.841812). The first-round H8 row above is a retrospective reconstruction from the later complete trajectories restricted to seed-zero builds. It was unavailable at that decision. [Original Git record](https://github.com/BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/blob/26ff26200c0c4b7d07742d7e3cdc9e253d1defeb/results/controls/trajectory_T0.json).

The initial round-two summary also sampled 41 checkpoints for one selected E4 build (ρ=0.856098). [Superseded round-two measurements](../../results/controls/trajectory_T0_superseded.json).

### Metric discrepancies and uncertainty

- The character counts agree with the archived text plus any neuron table for all 67 stimuli. T0 W-alg contains 59.9% as many characters as E3. The compiler can access more precision than the agent text. At T0 E3 frequency 2, its logit amplitude is 0.5342947876292539, displayed as 0.534. The effect of this precision difference on grades remains unknown. [Full specification](../../results/explanations/T0/E3/spec.json) · [Rendered text](../../results/explanations/T0/E3/EXPLANATION.md) · [Compiler](../../src/vmi/reference/compile.py).
- The wrong-role compiler achieved 0.6500% top-1 argmax accuracy. The fraction with its true logit within 1e-12 of the maximum was 2.7802%. The fields have distinct definitions. [Primary grade](../../results/compiler/T0/W-role.json) · [Diagnostic](../../results/controls/clock_pizza.json).
- Scale-adjusted logit FVU ranges were E3 0.12324–0.20951, W-alg 0.23171–0.23210, E4 0.07977–0.13972, W-scramble 0.07383–0.09784. All observed E3 errors were lower than W-alg. E4 and W-scramble overlapped. This post-hoc comparison used three builds per arm. [FVU records](../../results/controls/logit_fvu_T0.json).
- 10,236 of 12,656 off-diagonal ordered inputs (80.9%) have their reverse in a different fold. Both operands also recur throughout the grid. Alignment generalization to entirely unseen unordered pairs remains unmeasured. [Fold code](../../src/vmi/grader/metrics.py).
- For 20,000 independent experiments with three iid standard-normal observations per arm, the exact bootstrap percentile range covered the population mean in 75.11% and the difference of means in 80.79%. Monte Carlo standard errors were 0.31% and 0.28%. The single-arm range covers a continuous population median with probability 3/4. For independent draws from a continuous distribution F with finite mean μ, its coverage is 1 − F(μ)³ − [1 − F(μ)]³. Skewness can put the mean far from the median. [Statistical code](../../src/vmi/analysis/audit_statistics.py) · [Hesterberg](https://arxiv.org/pdf/1411.5279).
<!-- END reconstructed-evidence -->

## Validity of the experiment

- **Selection and generalization.** The protocol selected T0 for low overlap with the published frequencies and selected seed targets by frequency count. They are purposive checks from one architecture and task. T-new was trained after the protocol freeze with the same task and recipe. The published checkpoint has a contamination warning. E0/E1 scripts containing published frequencies establish prior-compatible behaviour. The pinned public repository contains the matching frequency lists for the selected seeds. Its history and the provider's stated cutoff constrain the timeline of public exposure. The origin of the simulator's knowledge remains uncertain. [Target repository history](https://huggingface.co/BurnyCoder/grokking-modular-addition-runs/commits/98c61de0c6e6219be70bb7eef60390ab510164f2) · [Provider training-data cutoff](https://platform.claude.com/docs/en/models/overview) · [Provider release date](https://www.anthropic.com/claude-haiku-5-5). [Selection record](../../results/targets_selection.json) · [Target inventory](../../results/training) · [Frequency and script records](../../results/frequencies_round2.json) · [Original modular-addition study](https://arxiv.org/pdf/2301.05217).
- **Information available to the simulator.** The sandbox provided an architecture, task, labels via arithmetic, build tools and normally an accuracy self-test. A simulator can use architecture knowledge, arithmetic and self-testing in addition to the explanation. The reported ablation removes the supplied self-test while retaining the task and the ability to calculate labels. Failure to reproduce the target can originate in the text, simulator implementation or grading metric. The compiler provides a specified implementation of the text with its own design choices. [Task](../../sandbox_template/TASK.md) · [Architecture](../../sandbox_template/ARCHITECTURE.md) · [Agent excerpts](../../results/saved_report_inputs.json) · [Compiler](../../src/vmi/reference/compile.py).
- **Interventions and controls.** Explanation levels change content, length and sometimes numerical constraints together. W-alg is shorter than E3 and shares the target frequencies. W-scramble preserves frequency information while changing the neuron table. H7's trained wrong targets share at most one key frequency with T0. The comparison cannot isolate target identity from frequency overlap. The planted series changes both internals and outputs. A failed KL monotonicity check falsifies H4b as stated. It supplies no general bound on grader reliability. [Stimuli and lengths](../../results/explanations/T0) · [Wrong-target comparisons](../../results/controls/wrong_targets_T0.json) · [Planted calibration](../../results/controls/planted_calibration_T0.json).
- **Hidden states and causal claims.** HS fits a separate affine map for each direction and fold, centres on training data, pools squared errors, clips each directional score to [0,1], takes the harmonic mean per site and averages sites. A strong score means measured states predict one another under this rule. Isotropic ridge and response-coordinate weighting affect invariance. The whitened sensitivity variant uses eigenvalue cutoffs that can change the retained subspace on rescaling. At the ReLU layer, arbitrary linear mixtures can predict neurons with different roles. Stitching is a separate intervention at one interface. Neither the HS trend nor a logit diagnostic establishes a unique causal algorithm. [Grader](../../src/vmi/grader/metrics.py) · [Whitening implementation](../../src/vmi/sensitivity.py) · [Kornblith et al., Table 1](https://proceedings.mlr.press/v97/kornblith19a.html) · [Stitching](https://arxiv.org/abs/2106.07682).
- **Output metrics.** Exact positive softmax probabilities identify centred logits by taking log probabilities and subtracting their row mean. A small temperature-one KL need not constrain wrong-class logit errors tightly. Scaled FVU fits a nonnegative scalar on the full grading grid. Its observed E3/W-alg ordering is descriptive. Only one logit-matched student met the pre-specified threshold for the later student experiment. Distance irrelevance measures variation of the correct-answer logit at a fixed operand difference. Zhong et al.'s typical ranges are empirical and compatible with mixed algorithms. [Logit comparisons](../../results/controls/logit_fvu_T0.json) · [Students](../../results/baselines/T0.json) · [Hinton et al., §2](https://arxiv.org/pdf/1503.02531) · [Zhong et al., §§4–5](https://arxiv.org/html/2306.17844).
- **Uncertainty, stopping and repeated looks.** The initial and cumulative rounds reuse the first builds. There was no across-round or across-hypothesis Type I error guarantee. H2 adjusts within its four T0 contrasts. H4 adjusts across the two levels in each baseline family. The sign-rank test assumes independent, symmetrically distributed differences under its null. Margins share T0 and the fixed comparison panel, so external target sampling uncertainty is outside that test. Available controls can confound positive H8 correlations with time and shared frequencies. Checkpoints along one optimizer path supply no independent training replicates. The local matrix stopped after its observed failures and the planned larger round was postponed. Simulator replicate IDs are labels for repeated hosted calls, with no control over provider sampling seeds. [Protocol and dated decisions](../preregistration.md) · [SciPy rank-sum test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) · [SciPy signed-rank test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

## Evidence boundary and reproducibility

The saved grades allow exact checks from site scores to HS, run grouping, reported tests, descriptive contrasts and published table cells. They do not contain the full per-input target/replacement logits and site activations used to fit each grading map. The original forward computation, its numerical conditioning, alternative grouped-fold scores, and figure pixels derived from those networks therefore remain independently unverified in this audit. Archived weights would permit a different, forward-running audit. The saved-only publisher authenticates the archived example panels by SHA-256 and identifies the commit and precision of the example table. Transcript excerpts retain run attribution. [Saved reporting inputs](../../results/saved_report_inputs.json) · [Statistical reconstruction](../../results/scientific-audit/reconstruction.json).

From the repository root run `uv run --frozen vmi analyse 1 --saved-only` for the first-round report and `uv run --frozen vmi analyse 2 --saved-only` for the cumulative report, figures, explorer data and this audit. Use `uv run --frozen vmi paper` to typeset the paper. The historical protocol, original agent stimuli (including intentionally false explanations), quotes and earlier dated reviews remain observational records. Earlier readings are superseded where this report identifies an error. [Claim ledger](claim-ledger.md) · [Primary literature receipts](../../results/scientific-audit/sources.json) · [Results explorer](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/).
