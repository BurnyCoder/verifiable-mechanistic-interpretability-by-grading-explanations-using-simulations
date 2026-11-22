# Independent audit — 2026-11-22 (after round 2, N = 3)

**Scope.** Everything after round 2 was checked again from scratch, ignoring earlier self-reviews:
- the results (recomputed from `results/` by a separate script);
- the methods (against the papers and library documentation they come from);
- every claim in the README, summary, paper, preregistration amendments, experiment reports, learnings, progress, docs, site, Hugging Face card and CITATION;
- every external source (about 130 URLs, each fetched).

Six checkers ran in parallel: three for claims (one per file group) and three for citations. Each finding below was re-verified against the files before anything was changed.

**Bottom line.**
- **What holds:**
  - Every reported statistic reproduces exactly from the raw run records.
  - Regeneration is deterministic.
  - The methods are used as their sources define them.
  - The study's main conclusions hold, with narrower wording.
- **What changed:** what the study claims to know, in several places:
  - an overstated invariance property of the primary metric;
  - overstated memorisation and "built" wording;
  - three preregistered analyses that had been computed only partly or not at all;
  - a length-matching claim that is false for one control;
  - several citations used for more than they say.
- **Verdicts:** no preregistered verdict changed. H8, now computed in its preregistered form (all six E3/E4 replacements × 401 checkpoints), stays supported: every replacement's ρ > 0, 0.75–0.84 (`results/controls/trajectory_T0.json`).

## 1. Checked and found correct

- **Numbers.**
  - An independent script (numpy/scipy only, not importing `vmi.analysis`) recomputed every H1–H8 statistic from `results/runs/*.json` and the no-agent result files. All match to floating-point precision:
    - condition means;
    - the Jonckheere–Terpstra statistic, with exact enumeration of all 369,600 assignments as a check on the Monte-Carlo p;
    - Spearman ρ;
    - exact Mann–Whitney p-values and Holm adjustment;
    - the H7 Wilcoxon p (1/64);
    - H8's ρ;
    - the Clopper–Pearson interval;
    - the Claude spend ($2.33 over 953 requests).
  - `vmi analyse 2` on unchanged code gives an empty `git diff`, figures included.
  - The README's generated blocks, `docs/data.json` and a fresh regeneration agree character for character.
- **Methods**, each checked against its source:
  - KL direction KL(M ‖ M̂) ([Conmy et al. 2023](https://arxiv.org/abs/2304.14997) §3);
  - linear CKA ([Kornblith et al. 2019](https://arxiv.org/abs/1905.00414) Table 1, Eq. 14);
  - orthogonal Procrustes 2 − 2‖XᵀY‖_* ([Ding et al. 2021](https://arxiv.org/abs/2108.01661) Eq. 7);
  - stitching ([Bansal et al. 2021](https://arxiv.org/abs/2106.07682));
  - Holm step-down;
  - the exact Mann–Whitney minimum p = 1/C(n₁+n₂, n₁);
  - the exact Wilcoxon at n = 6 (scipy `method='auto'` is exact for n ≤ 50 without ties or zeros);
  - Kendall's W without ties;
  - the rank-biserial correlation 2U/(n₁n₂) − 1;
  - Clopper–Pearson via beta quantiles;
  - the key-frequency rule, matching grokviz and [Nanda et al. 2023](https://arxiv.org/abs/2301.05217) §4.1 (frequencies {14, 35, 41, 42, 52});
  - the claim that a LayerNorm-free residual stream is defined only up to GL(d), which holds with biases included and is tested by `test_invariant_to_residual_basis_change`.
- **Facts about the outside world:**
  - **Claude Haiku 5.5:** the id `claude-haiku-5-5`, its June 2026 training-data cutoff and its October 2026 release ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)).
  - **Public dates:** the target runs repository has been public since 2026-10-21, after that cutoff, so T0–T2 cannot be in the simulator's training data.
  - **Tool versions:** Claude Code 2.1.330, llama.cpp b12600, bubblewrap 0.9.0 and PyTorch 2.14.
  - **Sources and links:** every arXiv ID resolves with the right title, first author and year; the Hugging Face repositories, collection and GitHub Pages site exist.
- **Design:**
  - no data leakage: explanations contain no weight arrays (tested), and the agent never sees the target;
  - the agent examples are chosen by fixed rules;
  - deviations 1–7 are disclosed, including the results-driven stop of the local runs (deviation 5).

## 2. Findings and fixes

Severity: **major** = a claim about the method or the results that was wrong or unsupported; **moderate** = a preregistered analysis missing or narrower than written, or a claim much stronger than its evidence; **minor** = wording, staleness or citation metadata.

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| 1 | major | HS was described as "invariant to exactly" GL(128) (preregistration §4, docs, state-of-the-art review, code docstring). Only completeness R²(M̂→M) and the HS = 1 case are. Soundness R²(M→M̂) pools over M̂'s dimensions and is invariant only to orthogonal maps and uniform scaling of M̂'s basis. The existing test only checked the perfect-equivalence case. | Kornblith et al. 2019 Table 1 (regression R² invariant in the predictor only); new test `test_soundness_depends_on_the_replacement_basis_and_whitened_soundness_does_not` (a non-orthogonal map of a half-explained replacement moves soundness by more than 0.2) | Erratum (deviation 8 e); corrected wording in every file; post-hoc whitened soundness, `sensitivity.cv_r2_whitened`, a held-out analogue of mean squared canonical correlation, reported in the grader sensitivity table with its effect on the ordering, the retrains, random networks and H7 |
| 2 | major | "Haiku built the paper's frequencies in all three T0 E1 runs." By the project's own key-frequency rule, all three used four of the five (14, 35, 41, 52) and none of T0's; two runs hard-coded `[14, 35, 41, 52, 73]` (73 ≡ −40 mod 113). | `results/frequencies_round2.json`; the three `build_weights.py` files | Key results, agent-example explanations, README, paper §4.6, summary and learnings now state four of five; the memorisation conclusion stands (four specific frequencies of 56 is far beyond chance) |
| 3 | major | The diagnosis's "whose fault" sentence ("slots at 12, 22, 32, 42, it skipped 32") was invented and contradicts the agent's own docstring, which puts heads 2 and 3 *together* at dimensions 42–51. The intended value is P3 = 42. A variant label also said "neuron counts as the explanation states" for counts that are the explanation's proportions scaled to 512. | the failed run's `build_weights.py` docstring; its `EXPLANATION.md` | `diagnose.py`: corrected owner text, new variant "P3 = 42 as its own docstring states" (now the fix shown), corrected label; `vmi diagnose` re-run |
| 4 | major | "HS measures how the answer is computed" and "retrains stay at the random-initialisation floor." Random networks score HS above working networks that use other frequencies, so HS is a feature-overlap score whose scale is not anchored at 0, not an algorithm-identity score. The retrains (same algorithm, other frequencies) are below the random floor, not at it. | `results/controls/random_init_T0.json`, `results/baselines/T0.json`, W-freq and E1 runs | Generated key results and highlights reworded; glossary gloss of HS changed to "1 = internals match up to a linear map"; limitation added |
| 5 | moderate | Preregistered statistics not computed: H3's HS difference of W-freq to each of E2–E4 with bootstrap CI; H4's secondary two-sided test against the distilled students; H1–H3 on the seed-sensitivity targets "with the same tests" (only directions were compared). | preregistration §5 versus `hypotheses.py`, `predictions.replication_rows` | All three computed (deviation 8 a). The student comparison reads "cannot tell at this N" (3 vs 3, two-sided minimum p = 0.1). |
| 6 | moderate | H8 used only the highest-HS E3/E4 replacement and every 10th checkpoint (41 of 401), labelled "all checkpoints". §5 says E3/E4 replacements against T0's 401 checkpoints. | `nonagent.trajectory` (`every=10`, `max(runs, key=hs)`) | Re-run as preregistered over every E3/E4 replacement and all 401 checkpoints, supported only if every ρ > 0 (deviation 8 b): supported, ρ 0.75–0.84; the superseded file is kept |
| 7 | moderate | W-alg (Pizza) is called length-matched and "E3-length" but is about 60 % of E3's length on every target, so H2's E3 > W-alg comparison is not length-controlled. | `results/explanations/*/W-alg/length.json` versus `E3/length.json` | Disclosed (deviation 8 d) and corrected in README, summary and paper |
| 8 | moderate | Preregistered analyses never run and never disclosed: §8's redrawn W-freq and W-scramble (3 seeds); §7's T-paper retrain baselines. The compiler accuracy criterion was relaxed in tests. | `manifests.py`, `results/baselines/` | Disclosed (deviation 8 c) and listed under limitations |
| 9 | moderate | Claims stronger than the evidence: "scores on famous networks measure what the AI remembers" (one simulator, one published model; full explanations still scored as expected there); "needs a capable coding agent" (5 runs of two small models in one harness); "long tables give the AI more ways to get answers slightly wrong" (an untested causal story); where H3 fails on other targets, "an implementation error of the simulator" (not diagnosed, and the true replacements there also fall below 0.99 agreement); H5 "fails in one condition" (the table showed 7 of 15 pairs by sign). | `predictions.highlight_rows`, round reports, `hypotheses_round2.json` | Hedged to the evidence; H5 now counts pairs by sign and beyond the bootstrap CI, as the falsifier requires |
| 10 | moderate | Citations used for more than they say: iWSP does have a no-description control and a feature-CKA comparison; Heap et al. test fuzzing and detection scores of SAE features, not simulation; Poché et al. concern output simulatability of classifiers; Zhong et al.'s Pizza differs from Clock in its logits, so it is not a "same outputs" case; Haklay et al. show an agent's LLM recalling one circuit (IOI), not agents in general; Gross et al.'s longest proof is brute force, not the weights; Huang et al. do not compare simulation scores with causal tests directly. | each source, fetched | Reworded in README, paper, summary, literature review and state-of-the-art review; iWSP's no-description control credited |
| 11 | minor | "Built exactly T0's frequencies" in the agent examples is read from the script's constants, not measured in the weights. | `agent_examples.why` versus `frequencies_round2.json` | Now "its script uses …" |
| 12 | minor | "Well within the 60-step, 60-minute budget": Claude Code runs have no step cap (amendment 2). "Tuned four constants" was not verified. Self-test counts included reading `selftest.py`. A line was shown twice. | `run_one.sh`, event logs | Corrected in `agent_examples.py` |
| 13 | minor | The generated text said "Monotone along the ladder" although E0 scores above E1 (against the preregistered prediction). "A separate algorithm-only run on the published checkpoint" was a mean of three runs. | `hypotheses_round2.json`, runs | Generator fixed |
| 14 | minor | "Unpublished targets": T0–T2 have no published analysis, but their weights are public. | HF repo created 2026-10-21 | Reworded throughout, with the training-cutoff argument |
| 15 | minor | Stale or wrong details: paper date; "offline" for a sandbox whose only connection is the model endpoint; logs called "every byte" (headers are excluded); "one command builds the paper" (`vmi paper` does); the check cost "≈ $0.01" (it was $0.0002); "44 agent conditions" (round 1 only); the HF card's run count included 15 dev runs; CITATION and pyproject descriptions still described a local-LLM simulator; Wilcoxon's minimum p counted zero differences; the trend statistic J is Terpstra's form of Jonckheere's S; the Procrustes scale is from Schönemann & Carroll 1970. | files named | Corrected |
| 16 | minor | Experiment reports and progress: stale or wrong counts, a misattributed quote (T-paper vs T0 E1), unsupported development numbers, the H8 wording, missing replications of the W-freq low-effort failure, and the stage-0 report missing its agent runs. | `results/`, git log | Corrected in the reports; development numbers labelled as notes |

## 3. Interpretation after the audit

- **The behaviour-versus-mechanism result stands** (H3, H4, H7): output KL rates a wrong-frequency explanation, retrains and distilled students as good as the true explanations, while HS does not.
  - The audit narrows what HS is: a linear feature-overlap score, valid at the level of key frequencies.
  - Its scale is not anchored at 0, and its soundness half depends on the replacement's basis.
  - **Whitened check, the outcome** (`results/sensitivity/grader_T0_round2.json`, paper Appendix B). With soundness made invariant to any invertible map of the replacement's basis, the coarse results all hold:
    - wrong frequencies, no explanation, algorithm only and wrong roles stay at the bottom;
    - W-freq stays below every E2–E4 mean (H3);
    - every E3/E4 run stays above every retrain (H4);
    - every E3/E4 replacement keeps a positive margin over all 12 wrong targets (H7);
    - H1's trend test still passes.
  - **What whitening changes:** the order within the high group (Kendall τ 0.72 and 0.78 at the two eigenvalue cut-offs). E2 and E3 tie, and the Pizza explanation moves above E3. So "every E3 run is above every E2 run" holds only under the preregistered weighting, and is now stated that way.
  - **Reproduction check:** the regrade also reproduced every stored primary HS from the saved weights, to about 1e-16.
- **H4b's failure** reads better as a failed assumption of the calibration series (the planted model's function does not drift monotonically) than as a failure of the grader. HS still falls monotonically there.
- **All evidence is N = 3, one task family and one simulator.** Every summary statement now says so.

## 4. Completion check (second pass against the audit plan)

A second pass against every item of the audit plan found steps that had been skipped or done only in part. Each was then finished:

- **Hand-typed result numbers.**
  - Counts and ratios quoted inside README, summary and paper prose are now generated inline values (`<!-- v:name -->…<!-- /v -->`, `tables.inline_values`), refilled by `vmi analyse` like the tables. They cover the local-run count, W-alg's length ratio, the memorisation counts, the low-effort failure counts, round 2's run count, the E3-versus-E2 comparison, the number of targets with the wrong-frequency gap and the E0/E1 accuracy.
  - A count of failed expectations was removed instead.
- **New statistics recomputed independently.** H3's differences, H4's two-sided student p (0.1), H8's per-replacement ρ (min 0.751, median 0.774) and the replication tests (exact enumeration) all match.
  - The grader's pooled R² equals scikit-learn's documented `variance_weighted` R² in-sample.
- **Citations.** One consolidated list: [citation-check-2026-11-22.md](citation-check-2026-11-22.md). It includes the plan's URLs, which no checker had covered; all of them were among the sources already checked.
- **Figures.** All were checked for graph crimes.
  - The simulator figure mixed N = 1 (Qwen) and N = 3 (Haiku) under "round 2"; its legend now gives N per simulator.
  - An overlapping label in the length figure was moved.
  - The PDF's sensitivity table had overlapping columns; it now has wider columns.
- **Results-driven changes.** Deviations 5, 7 and 8 are now named in the README's predicted-versus-measured section and in the paper's methods.
- **Learnings.** The "thinking not returned" claim was corrected and the Ollama sources were added.

## 5. Not changed

- The primary metrics, the verdict rules and the preregistered hypotheses.
- Round 3 stays postponed. No Claude budget was spent by this audit.
