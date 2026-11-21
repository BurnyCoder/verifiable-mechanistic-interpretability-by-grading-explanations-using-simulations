# Progress

## 2026-11-21 — setup and preregistration

**Done**
- Environment: uv project (Python 3.12, torch 2.14.1+cu130 sees the RTX 5070), `grokviz` pinned at `d52a898`.
- Public GitHub repo created; `.env` git-ignored before the first commit.
- llama.cpp b12600 (CUDA 13.5) installed in `D:\AI\vmi-sim\llama.cpp`; `llama-server` detects the GPU (7 GB free).
- Downloads started at pinned revisions: Qwen3.6-35B-A3B UD-IQ4_XS (17.7 GB), the 22 target runs.
- Literature review and state-of-the-art review written; all 40 cited arXiv IDs verified against the arXiv API.
- Preregistration v1, planned runs, README, plan pushed before any experiment.

**Now:** implementing the pipeline test-first (targets → explainer → reference compiler → grader → sandbox runner).

**Not done:** stage 0, rounds 1–3, paper, site, Hugging Face upload.

**Estimate:** implementation ~1 day; stage 0 + round 1 ≈ 16 h of mostly GPU time; rounds 2–3 ≈ 5 days of GPU time.
Roughly 10 % of the whole plan is done.
