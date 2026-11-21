# Progress

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
