# Progress

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
