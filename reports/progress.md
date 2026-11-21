# Progress

## 2026-11-21 20:45 — round 1 complete; local Qwen runs stopped (user decision)

**Done since 18:30** (N = 1 — not yet N = 3 or N = 10)
- Local Qwen runs stopped on your instruction after five runs, all at chance (Qwen3.5-4B × 4, Qwen3.5-9B × 1);
  rounds 2–3 have no local conditions (32 Claude Haiku conditions per round). Logged as deviation 5 (a results-driven
  stop); H5 and H6 become "not evaluable".
- Stitching KL regraded for all runs and the compiler (primary metrics reproduced exactly); it shares HS's invariance.
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
- Root cause found in the Qwen3.6 model card: the 4k thinking budget was far below the model's trained regime (≥32k
  output recommended), so it continued designing inside code comments; and the presence penalty of 1.5 was the
  general-task preset, not the coding one (0.0). Settings now follow the card's coding preset, with a 16k thinking
  budget, 24k-token turns, an 80k context with an explicit OpenCode input limit, and a 60-minute run cap.

**Now:** smoke run 6 (dev × E3, seed 5) with these settings.

**Not done:** smoke runs on E0/E2/E4, protocol-freeze amendment (amendment 2), rounds 1–3, paper, site, HF upload.

**Estimate:** with 60-minute runs, round 1 (40 runs) takes up to ~40 h of GPU time and round 2 up to ~80 h; round 3
will stop at a stage boundary when the week ends (preregistered rule). About 23 % of the plan is done.

## 2026-11-21 05:00 — stage 0, agent smoke runs (dev target)

**Done since 02:30**
- Throughput decided: warm, Qwen3.6-35B-A3B runs at 289 prompt / 21.6 generated tokens/s with 48k context and
  `--n-cpu-moe 34` (6.8 GB VRAM), above the 8 tokens/s threshold → kept as main simulator (amendment 1).
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
  (3.9 tokens/s prompt processing; free RAM 1.3 GB during the request; the non-expert weights + KV cache used 5.1 GB
  VRAM, leaving no room for experts on the GPU). A warm re-test at 32k context is running; Qwen3.5-9B/4B GGUFs are
  downloading for the preregistered fallback.

**Now:** finishing the throughput decision, then the first OpenCode smoke runs on the dev target.

**Not done:** protocol freeze, rounds 1–3, paper, site, Hugging Face upload.

**Estimate:** stage 0 finishes in ~2–4 h; round 1 ≈ 13–16 h of GPU time; rounds 2–3 ≈ 5 days. About 20 % of the
plan is done.
