# Progress

*Entries are dated snapshots, newest first; older entries keep what was believed at the time. Statements later
corrected — e.g. "HS tracks circuit formation" (now: shared frequency structure, deviation 11), "blind to Clock vs Pizza
and to scrambled tables" (now: did not separate the agents' builds at N = 3, deviation 12), "41 of 96 runs announce a
sweep" (now 44 of 96 by reading, deviation 13) — are listed in the preregistration's deviations.*

## 2026-11-23 — README restructured (no new runs, no result changed)

- README reordered to the project's README format. The order is now: summary, ✅/❌ highlights, a visual overview of
  four figures, the agent's own words in the example runs, a new Methodology section (model, data, explanations, simulator, baselines, evaluation; with the
  generated `targets` table), architecture with a code-module graph, results, setup and a new Reproduce section,
  then the to-do list.
- Every example and reason is kept. Refilling the README with the generator changes nothing but the new `targets`
  table.
- Stale bookkeeping fixed: the README said "twelve deviations" and listed six audits. The deviation disclosure now
  points to the preregistration's full list, and the audit links point to `reports/reviews/`.

## 2026-11-22 20:00 — seventh independent audit (deviation 14)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews or audit scripts (`reviews/audit-7-2026-11-22.md`). Every round-2
  verdict re-derived with numpy and the standard library only, every null distribution enumerated in full
  (`results/audit7/verdicts.json`, all checks agree); the T0 runs and compiler builds the tests use re-graded with a
  numpy forward pass written from `ARCHITECTURE.md` (`results/audit7/regrade.json`); Zhong et al.'s distance
  irrelevance recomputed for 41 networks (`results/audit7/clock_pizza_check.json`); three parallel source checks read
  the cited papers in full.
- Found and disclosed: the static audit catches gradient-descent and optimiser patterns, not small searches over a
  constant; reading every final build script found such searches in some runs, none in a hypothesis test, and a
  sensitivity check without them changes no verdict (`results/audit7/build_script_searches.json`). Corrected: Nanda et
  al.'s logit scale (given as a figure) and key-frequency reading (six embedding frequencies, five used), Gromov,
  Gross et al., Huang et al., Poché et al.'s shortcut hypothesis, iWSP's agents.

**Effect:** no primary metric, preregistered test or verdict changed.

**Next:** round 3 (N = 10) only when you ask; before it, extend the static audit to flag selection loops.

## 2026-11-22 18:00 — sixth independent audit (deviation 13)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews or audit scripts (`reviews/audit-6-2026-11-22.md`). Every round-2
  verdict re-derived with new code and exact methods (`results/audit6/verdicts.json`, all checks agree); the scores
  re-graded by a third, numpy-only implementation read straight from the published checkpoints
  (`results/audit6/regrade.json`); metric, statistics and Clock/Pizza code read against the sources; three parallel
  source checks opened every citation, and each discrepancy was re-checked by hand.
- Found and corrected (no verdict changed): H2's "cannot tell" is deviation 1's rule, applied at N = 3 after the
  verdicts were read; §5 read literally gives "not supported", and both readings are now shown. The count of agents'
  parameter searches was a pattern match with false matches and misses; read sentence by sentence it is 44 of 96
  (`results/audit6/sweep_labels.json`). Zhong et al.'s Fig. 7 lines, Sutter et al.'s own suggestion of linear maps,
  Nanda et al.'s key-frequency rule for the main model (embedding, not W_L), Poché et al.'s scope, "exact recipe",
  grokviz's restricted-loss figure in the state-of-the-art review, the socket-path limit, a Klabunde docstring and an
  overlapping figure label.
- Added (post hoc): the embedding-spectrum cosine, a preregistered secondary metric with no forward pass, orders the
  T0 builds almost as HS does; HS adds discrimination mainly for W-role (paper §5).

**Effect:** no primary metric, preregistered test or verdict changed.

**Next:** round 3 (N = 10) only when you ask.

## 2026-11-22 11:30 — fifth independent audit (deviation 12)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews or audit scripts (`reviews/audit-5-2026-11-22.md`). Every hypothesis
  verdict re-derived from the raw run records with exact methods and no project analysis code: all 88 stored
  statistics, verdicts and published values reproduce, rounds 1 and 2 (`results/audit5/hypotheses_check.json`); every
  score those tests use re-graded with an independent forward pass and ridge, 306 gradings, all reproduce
  (`results/audit5/regrade_independent.json`). Metric,
  statistics and method code read against their sources; three parallel source checks opened every citation.
- Found and corrected (no verdict changed): T0's "true" E2–E4 call frequency 2 a full key frequency although T0 has it
  only in its neuron-to-logit map (likely a harmonic of 37), and embeddings carry harmonic frequencies the explanations
  omit (`results/audit5/checks.json`); the scrambled-table blind spot holds for the agents' builds, not for a literal
  build on T0; "seven deviations after results" undercounted; wording that overstated "cannot tell" and the KL
  ordering; Zhong et al.'s Fig. 7 lines, Sutter's wording and the source of the 75 %/80 % coverage figures; the nine
  retrains, the compiler's E0 build, the H8 lower bound, the plumbing-check count and two file lists.
- Added: `results/targets_selection.json` (the preregistered target rules applied to the published key frequencies).

**Effect:** no primary metric, preregistered test or verdict changed; two results are described more narrowly.

**Next:** round 3 (N = 10) only when you ask.

## 2026-11-22 10:15 — fourth independent audit (deviation 11)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews or audit scripts (`reviews/audit-4-2026-11-22.md`). An independent
  numpy/scikit-learn grader written from the preregistration reproduces all 261 stored scores that come from weight
  files (`results/audit4/regrade.json`); independent exact tests reproduce all 44 stored test statistics and verdicts
  and the replication table (`results/audit4/hypotheses.json`). Four parallel source checks read every cited source.
- Corrected interpretations: H4b fails because the planted model's function does not drift monotonically (KL follows
  it), not because KL fails; H8's positive correlation is shared by final T0 itself and every network with T0's
  frequencies during circuit formation, and its size is set by drift after grokking, so it shows shared frequency
  structure, not a separate tracking of circuit formation (post-hoc positive control,
  `results/audit4/h8_positive_control.json`). Three-run bootstrap intervals cover about 75 %, not 95 %.
- Corrected facts and citations (iWSP, Huang, Gao, Haklay, Poché, Kornblith/Klabunde, Bansal, netrep, Zhong Fig. 7,
  Nanda App. C.2, Sutter, Qwen cards, Claude Code budget flag, pricing source, VRAM units); twelve method references
  added to the paper; Claude Haiku 5.5's release date (2026-10-07, before the targets' upload) added to the
  contamination argument; stale code comments fixed; figures for H4b and H8 extended.

**Effect:** no primary metric, preregistered test or verdict changed (`results/hypotheses_round2.json` regenerates
byte-identically); H4b and H8 are read more narrowly.

**Next:** round 3 (N = 10) only when you ask.

## 2026-11-22 09:00 — third independent audit (deviation 10)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews (`reviews/audit-3-2026-11-22.md`): every hypothesis verdict recomputed
  by an independent script written from the preregistration (all test results reproduce), the primary metric
  re-implemented (five runs agree to 1e-12), every condition mean and the cost recomputed, all 96 transcripts read
  for accuracy checks, every citation in the paper, README and both reviews checked against its source on the web.
- Corrected: H1's verdict now reports its failed "E0 below E1" clause; the T-paper replication of H1 uses every level
  it ran (E1 < E3 < E4: not supported); H1's trend test is exact; the self-test ablation is shown not to remove label
  feedback (all three runs measured accuracy with their own code), so the claim built on it is withdrawn; the
  alignment-class rationale is corrected for `mlp_post`; T1's "fewest frequencies" is a tie; citation wording
  (Sutter, Poché, Paulo, Huang, Yip, Haklay, Zhong, Cammarata, OpenCode, Qwen3.8 GGUFs).
- Added (post hoc): Zhong et al.'s Clock/Pizza diagnostics (`vmi clockpizza`): every target is Clock-like and the
  literal Pizza build Pizza-like (W-alg is a wrong explanation of T0, as designed), but the replacements built from
  the true E1–E3 explanations are Pizza-like too — and so are the compiler's literal E1–E4 builds, so H2's E3 > W-alg
  compares two Pizza-like builds (audit 4) — and HS still rates them about 0.9. Own label checks counted in Appendix D.

**Effect:** no primary metric or preregistered test result changed; H1's reported verdict and the T-paper replication
changed as above.

**Next:** round 3 (N = 10) only when you ask.

## 2026-11-22 07:00 — second independent audit (deviation 9)

**Done** (no new agent runs, no API spend; N = 3 — not yet N = 10)
- Audit made without reading the earlier reviews: code against its definitions, every hand-written number against
  `results/`, every citation against its source (two web passes), the agents' build scripts and transcripts, the
  Hugging Face target repository. Report: `reviews/audit-2-2026-11-22.md`.
- Corrected: the targets' key frequencies *are* public (but only since after the simulator's cutoff, and the sandbox
  has no internet); H5 described on T0 (preregistered) apart from the seed targets and T-paper; memorisation reported
  on all interchangeable E0/E1 runs; agents' self-test tuning and explanation repair disclosed; "no step cap"
  erratum; overclaims narrowed; citations reworded (Yip, Heap, Sutter, Haklay, Poché, Stander, FADE, Flash-Next,
  Claude Code sandbox) and two missing references added.
- Added (post hoc, no verdict uses them): H8 negative controls, threshold-free spectrum share and declared frequency
  lists, transcript counts of self-test use, calibrated-KL table, T0 E2–E4 replication row, cost split, recorded
  software environment, exact trend test.

**Effect:** no preregistered verdict changed (see the audit report, §4).

**Next:** round 3 (N = 10) only when you ask.

## 2026-11-22 04:00 — independent audit of results, methods and claims

**Done** (no new agent runs; N = 3 — not yet N = 10)
- Every reported statistic recomputed by an independent script from the raw run records: all reproduce exactly;
  regeneration is deterministic.
- Methods checked against their sources and library docs; about 130 cited URLs fetched and checked.
- Fixed (deviation 8, `reviews/independent-audit-2026-11-22.md`):
  - the primary metric's invariance claim (soundness is not invariant to non-orthogonal maps; a whitened check is
    added to the grader sensitivity);
  - missing preregistered statistics (H3 CIs, H4 student test, replication tests);
  - H8 re-run as preregistered (all E3/E4 replacements × 401 checkpoints);
  - W-alg's length mismatch and two analyses never run, now disclosed;
  - the memorisation wording (four of the paper's five frequencies);
  - the failed run's diagnosis (intended P3 = 42, from its own docstring);
  - HS's floor (random networks score above other-frequency networks);
  - citations that were cited for more than they say.

**Now:** nothing running after the audit's regeneration. **Next:** round 3 (N = 10) only when you ask.

**Estimate:** round 2 of 3 complete: 101 of 325 planned agent runs (31 %); every Claude Haiku condition has N = 3.

## 2026-11-22 01:30 — round 2 (N = 3) complete; round 3 postponed; waiting for your go-ahead

**Done since 20:45** (N = 3 — not yet N = 10)
- Round 2: 64 Claude Haiku runs (main, controls, ablations, hyperparameter, seed), no failures; no-agent controls and
  grader sensitivity recomputed; hypotheses, tables, figures, paper, site regenerated. Claude spend $2.33 of $20.
- Deviation 6 (round 3 postponed, your decision) and deviation 7 (H1 now reports its KL part; H2 applies the §6 power
  rule at N = 3; round reports keep their own round's numbers) logged in the preregistration.
- README: predicted vs. measured (plain and expert), highlighted examples, concrete network outputs vs. ground truth,
  the agents' own words for successes and failure modes; new figures (frequencies each network built; outputs vs.
  internals); ablations table; `reports/agent-examples.md`; round-2 report and self-review.

**Positive (N = 3):** H1's HS trend, H3, H4, H7 and H8 pass their preregistered tests; the wrong-frequency gap holds on
all five targets; format, self-test and effort barely move HS; the algorithm-only memorisation shows in most build scripts
(T0 and T-paper pooled, since the E1 text is identical on every target; counts in the paper's §4.6).
**Negative (N = 3):** output KL's trend fails (H1, KL part); HS cannot separate Pizza or a shuffled neuron table (H2
ties); H4b's KL half fails; the compiler is not a ceiling (H5: within its level on T0, exceeded on the seed targets);
local models never above chance; the round-1
low-effort collapse and the E0 memorisation did not replicate in every run.

**Now:** nothing running. **Next:** round 3 (N = 10, 224 Haiku runs, about 10–15 hours, about $5) only when you ask.

**Estimate:** round 2 of 3 complete (101 of 325 planned agent runs, 31 %); every Claude Haiku condition through round 2
ran (see the audit entry above for two no-agent analyses that were not run).

## 2026-11-21 20:45 — round 1 complete; local Qwen runs stopped (user decision)

**Done since 18:30** (N = 1 — not yet N = 3 or N = 10)
- Local Qwen runs stopped on your instruction after five runs, all at chance (Qwen3.5-4B × 4, Qwen3.5-9B × 1);
  rounds 2–3 have no local conditions (32 Claude Haiku conditions per round). Logged as deviation 5 (a results-driven
  stop); H5 and H6 become "not evaluable".
- Stitching KL regraded for all runs and the compiler (primary metrics reproduced exactly); it has completeness's invariance (any invertible linear map of M̂'s resid_mid).
- Post-hoc target swap confirms memorisation; round-1 review addendum; hypotheses, figures, tables, paper regenerated.

**Positive results (Haiku):** HS jumps from about 0.05 (E0, E1) to about 0.9 once the key frequencies are named, close
to the agent-free compiler; wrong frequencies score at the floor on all five targets; output KL is fooled where HS is
not (H3 supported); HS tracks circuit formation (H8 supported); per-neuron detail (E4) raises matched-neuron correlation.
**Negative results:** HS and stitching are blind to Clock vs Pizza and to scrambled neuron tables; no HS gain beyond E2;
E3/E4 detail lowers task accuracy; H1 and H4b not supported as preregistered; local Qwens never above chance;
memorisation makes scores on the published checkpoint meaningless.

**Next:** merge, Hugging Face upload, then round 2 (N = 3): 64 Haiku runs, ≈ 8–9 h, ≈ $1.7.

**Estimate:** about 50 % of the plan is done (round 1 of 3 complete; 37 of 325 agent runs). Round 2 finishes ≈ 06:00;
round 3 (224 Haiku runs, ≈ 30 h) ≈ Nov 23–24. Claude spend so far $0.86 of $20.

## 2026-11-21 18:30 — round 1: all Claude stages done; Qwen3.5-4B at chance; 9B and 35B running

**Done since 15:30** (N = 1 — not yet N = 3 or N = 10)
- Haiku hyperparameter and seed stages; T-new trained and graded; Hugging Face repo and collection refreshed.
- Post-hoc memorisation check (`vmi analyse` → `results/frequencies_round1.json`): with no frequencies in the
  explanation, Haiku builds in the published checkpoint's frequencies, so scores on T-paper are confounded.
- Qwen3.5-4B ablation (E2, E3, E4, W-freq): all four replacements at chance (task accuracy 1/113, HS 0). E2 used all
  60 steps; E3, E4 and W-freq hit the 60-minute cap. The smallest simulator cannot use even the most detailed
  explanation in this harness.
- `vmi local` serves each local model in turn and runs its pending conditions.

**Now:** Qwen3.5-9B (4 conditions), then Qwen3.6-35B-A3B (4 conditions), each run up to 60 minutes. Update 20:05:
9B E2 also at chance (accuracy 1/113, HS 0, 24 steps, hit the 60-minute cap); 9B E3 running. Post-hoc target swap:
the same T0 E0/E1 weights score far higher against the published checkpoint than against T0.

**Next:** regrade, `vmi analyse 1` (H5, H6 become evaluable), paper, round-1 review addendum, push; then round 2.

**Estimate:** local runs finish by ≈ 02:30 at the latest. Round 1 is then complete; about 45 % of the plan is done.

## 2026-11-21 15:30 — round 1: main, controls, baselines done; ablations, hyperparameter and seed stages running

**Done since 12:25** (all at N = 1 — not yet N = 3 or N = 10; numbers in the generated tables of
[experiments/01-round1/experiment.md](experiments/01-round1/experiment.md))
- Main stage (Haiku, T0 × E1–E4), controls stage (E0, W-*, random init, wrong targets, planted calibration, training
  trajectory), baselines stage (independent retrains, logit-distilled students, Fourier-restricted model, weights dump),
  reference compiler on all five targets.
- Haiku ablations: math and code explanation formats and the no-self-test variant all build working replacements.
- One-command report generator (`vmi analyse`): hypotheses, figures, and every table in README, reports and paper; the
  paper draft builds to `reports/paper.pdf` (`vmi paper`).
- Fixes, each logged as a post-freeze deviation in the preregistration: analysis code aligned with amendment 2 and
  power-aware verdicts; a unix socket path over Linux's length limit (one infrastructure failure, archived and re-run);
  a NaN secondary stitching metric.

**Positive results so far:** naming the key frequencies lifts HS from the floor to near the agent-free compiler; output
KL is fooled by a wrong-frequency explanation, independent retrains and distilled students, HS is not (H3 supported at
N = 1); the planted-mechanism HS falls monotonically; the score tracks circuit formation (H8).
**Negative results so far:** HS does not separate the Pizza explanation or the scrambled neuron table from the true
ones; output KL does not rank the ladder; on the planted mechanism KL is not monotone (H4b not supported as
preregistered); E4 < E3 in the single run.

**Now:** Haiku hyperparameter (effort low/high) and seed-sensitivity runs (≈ 16 runs, ≈ 80 min); Qwen3.5-4B ablation
runs on the GPU (each up to 60 min), then Qwen3.5-9B and Qwen3.6-35B-A3B (4 runs each).

**Next:** regrade the stitching metric, `vmi analyse 1`, round-1 review in `reports/reviews/`, Hugging Face upload,
then round 2 (N = 3).

**Estimate:** Haiku stages finish ≈ 17:00; the local Qwen ablations need up to ≈ 10 more GPU-hours (finish ≈ 01:00).
About 40 % of the plan is done. Claude spend so far is in the generated cost table (well under $1 of the $20 cap).

## 2026-11-21 12:25 — protocol frozen; round 1 started

**Protocol freeze:** tag `protocol-freeze` = commit `801edf013db7239f8994e6780c5873e1890f5103` (preregistration
amendment 2). Main simulator: Claude Haiku 5.5 in Claude Code; local Qwens are the capability ablation.

**Stage 0 outcome (dev target, never reported):** all local Qwen runs stayed at chance; Claude Haiku 5.5 met the
stage-0 rule in its first run (E4: accuracy 100 %, KL 3.6·10⁻⁷, HS 0.985, $0.02). Haiku across the dev ladder:
HS 0.166 (E0), 0.069 (E1), 0.944 (E2), 0.953 (E3), 0.985 (E4), 0.096 (W-freq) with accuracy ≥ 98 % everywhere.
Claude spend so far: $0.13 of $20.

**Now:** round 1 (N = 1), main stage: T0 explained and compiled (compiler HS: E1 0.05, E2 0.895, E3 0.927,
E4 0.952, W-freq 0.095); Haiku runs T0 × {E1, E2, E3, E4} (≈ 2–8 min each).

**Next:** controls stage (5 Haiku runs + no-agent controls incl. planted-calibration training), baselines (retrains,
distilled students, Fourier-restricted, weights dump), ablations (3 Haiku + 12 local Qwen runs ≈ 12 h GPU), hyper
(4 Haiku), seed sensitivity (T-new trained now that the protocol is frozen; 16 Haiku runs); then hypotheses at N = 1,
README, site, paper.

## 2026-11-21 11:15 — stage 0: no local agent above chance yet; Claude prepared

**Done since 07:00**
- Custom bash-only harness (mini-SWE-agent design) replaced OpenCode after six OpenCode runs at chance (user
  suggestion); it removed the protocol failures (truncation, compaction, dropped replies). Two harness fixes: replies
  in `reasoning_content`, and eliding old messages before the context overflows.
- The Clock explanation's MLP step now states the neuron function as precisely as the wrong-algorithm (Pizza) text;
  verified on the dev target first.
- Decision rule fixed before any Claude run: "better than chance" = task accuracy ≥ 5 %.
- Local results on the dev target (accuracy / HS): Qwen3.6-35B E3 0.95 % / 0.33, E4 2.44 % / 0.47; Qwen3.5-9B E4
  0.88 % / 0.12 — all below the rule. Qwen3.5-4B E4 is running.
- Claude Code mode built and tested (user decision: only if all Qwens stay at chance; Haiku → Sonnet → Opus; API
  credits; $20 cap): key and workspace id injected by a proxy outside the offline sandbox, per-request budget ledger,
  web tools blocked, pre-push secret scan. Plumbing check passed; total spend $0.0002.

**Now:** Qwen3.5-4B × E4 (ends ≈ 12:15).

**Next:** if the 4B run is at chance, Claude Haiku 5.5 × E4 on the dev target (≤ 60 min, ≤ $2); then Sonnet 5.5
(≤ $6) and Opus 5.5 (≤ $10) only if the previous model stays at chance. Once one run beats chance: protocol-freeze
amendment and round 1. If the $20 is spent without success: evaluate, update README/site/paper, wait for the user.

**Estimate:** about 25 % of the plan is done. Finding a working simulator: 1–4 h. If only Claude works, the full
400-run matrix will not fit in $20 and the scale-up needs the user's decision.

## 2026-11-21 07:00 — stage 0, agent smoke runs (dev target)

**Done since 05:00**
- Smoke runs 4 and 5 produced the first graded agent replacements (accuracy 1.4 % and 0.5 %, HS 0.34 and 0.32): the
  pipeline works end to end (sandbox → weights → audit → grade), but the agent spent most of each run on truncated
  whole-file writes and context compactions.
- Root cause found in the Qwen3.6 model card: the 4k thinking budget was far below the model's trained regime (32,768
  output tokens recommended for most queries), so it continued designing inside code comments; and the presence penalty of 1.5 was the
  general-task preset, not the coding one (0.0). Settings now follow the card's coding preset, with a 16k thinking
  budget, 24k-token turns, an 80k context with an explicit OpenCode input limit, and a 60-minute run cap.

**Now:** smoke run 6 (dev × E3, seed 5) with these settings.

**Not done:** smoke runs on E0/E2/E4, protocol-freeze amendment (amendment 2), rounds 1–3, paper, site, HF upload.

**Estimate:** with 60-minute runs, round 1 (40 runs) takes up to ~40 h of GPU time and round 2 up to ~80 h; round 3
will stop at a stage boundary when the week ends (preregistered rule). About 23 % of the plan is done.

## 2026-11-21 05:00 — stage 0, agent smoke runs (dev target)

**Done since 02:30**
- Throughput decided: warm, Qwen3.6-35B-A3B runs at 289 prompt / 21.6 generated tokens/s with 48k context and
  `--n-cpu-moe 34` (6,790 MiB VRAM), above the 8 tokens/s threshold → kept as main simulator (amendment 1).
- Isolation check 15/15 including the LLM being reachable through the relay.
- Three end-to-end smoke runs on dev × E3, none produced weights yet; each failure fixed on the dev target only
  (details in `experiments/learnings.md`): (1) 16k tokens of thinking without acting → thinking budget 4,096 tokens per
  turn; (2) a repetition loop inside a file write → presence penalty 1.5 and an 8k-token per-turn cap; (3) reasoning in
  code comments inside one 500-line write, truncated three times → starter `build_weights.py` plus budget guidance in
  TASK.md.

**Now:** smoke run 4 (dev × E3, seed 3) with the starter file; then smoke runs on E0, E2 and E4.

**Not done:** protocol freeze amendment, rounds 1–3, paper, site, Hugging Face upload.

**Estimate:** a smoke run takes ≤ 40 min; stage 0 needs 2–4 more runs (~2–3 h) if run 4 works. Round 1 then takes
≈ 15–27 h of GPU time (40 runs at 20–40 min). About 22 % of the plan is done.

## 2026-11-21 02:30 — stage 0 (tooling, dev target)

**Done**
- Setup, reviews, preregistration (pushed before any experiment, commit `08dcc51`).
- Implemented and tested (19 tests pass): weight schema, target registry, grader (float64; KL, cross-validated
  hidden-state similarity, secondary metrics), explainer (analysis, ladder E0–E4 + W-*, format variants, E-paper),
  reference compiler, non-agent controls/baselines, hypothesis evaluation, offline bubblewrap sandbox with unix-socket
  relay and seccomp AF_VSOCK filter, OpenCode runner, llama-server manager, run manifests (40 conditions, 400 runs).
- Isolation check inside the sandbox: all checks pass (no network, no DNS, only loopback, no Windows drives, no
  /home, no interop, vsock blocked, /usr read-only, numpy and OpenCode present); LLM reachability pending server.
- Dev target (`wd1_s10`): explanations rendered; reference compiler reaches 100 % accuracy for E1, E3, E4, W-freq,
  W-alg, W-scramble (E2 99.75 %); W-role breaks behaviour as intended.
- Reproduction check: re-training reproduces key frequencies for `wd1_s10` but not bitwise weights (batch-composition
  dependence; see learnings).
- Throughput: Qwen3.6-35B-A3B UD-IQ4_XS at 64k context was far below the preregistered 8 tokens/s threshold
  (3.9 tokens/s prompt processing; free RAM 1.3 GB during the request; the non-expert weights + KV cache used 5,141 MiB
  VRAM, leaving no room for experts on the GPU). A warm re-test at 32k context is running; Qwen3.5-9B/4B GGUFs are
  downloading for the preregistered fallback.

**Now:** finishing the throughput decision, then the first OpenCode smoke runs on the dev target.

**Not done:** protocol freeze, rounds 1–3, paper, site, Hugging Face upload.

**Estimate:** stage 0 finishes in ~2–4 h; round 1 ≈ 13–16 h of GPU time; rounds 2–3 ≈ 5 days. About 20 % of the
plan is done.
