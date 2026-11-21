# Planned runs

*2026-11-21, before any run. Hypotheses and tests: [preregistration.md](preregistration.md).*

## Fixed for every agent run

| Setting | Value |
|---|---|
| Architecture (target and replacement) | 1-layer transformer, p = 113, d_model 128, 4 heads × 32, d_mlp 512, ReLU, no LayerNorm (paper layout, `grokviz` `d52a898`) |
| Main simulator | OpenCode 2.3.2 + Qwen3.6-35B-A3B UD-IQ4_XS (rev `a483e9e`), llama.cpp b12600, 64k context, temperature 0.6, per-run seed |
| Sandbox | WSL2 + bubblewrap, no network, only the run folder writable, numpy only, no access to any target file |
| Agent inputs | `TASK.md`, `EXPLANATION.md` (+ CSV for E4), `ARCHITECTURE.md`, `forward.py`, `selftest.py` |
| Agent output | `build_weights.py` + `weights.npz` (11 named arrays in the paper layout) |
| Limits | 40 min wall clock, 60 agent steps, build script ≤ 60 s |
| Grading | all 113² inputs; KL_out and HS (preregistration §4) plus secondary metrics |

## Stage 0 — tooling and smoke test (dev target `wd1_s10`, never reported)

| Run | Purpose |
|---|---|
| Reproduction check | re-train T0's spec with `grokviz.train_runs`; compare with the published run |
| Isolation check | `scripts/isolation_check.sh` inside the sandbox |
| Throughput | llama-server tokens/s with `--n-cpu-moe` tuned to ~6 GB VRAM |
| ~5 agent runs | dev × {E0, E2, E3, E4}: end-to-end, transcripts read by hand, prompt fixes |
| Reference compiler | dev × E1–E4: must reach 100 % accuracy for E2–E4 |

## Rounds (each stage in this order; N grows 1 → 3 → 10)

| Stage | Conditions (agent) | Runs per round (R1 / R2 / R3) | No-agent work |
|---|---|---|---|
| Main method | T0 × {E1, E2, E3, E4} | 4 / 8 / 28 | reference compiler on E1–E4 |
| Controls | T0 × {E0, W-freq, W-alg, W-role, W-scramble} | 5 / 10 / 35 | random init ×5, wrong targets, planted calibration (trained), trajectory over 401 checkpoints |
| Baselines | — | — | 9 retrains, 3 distilled students (trained), compiler on all explanations, Fourier-restricted M, weights dump |
| Ablations | Qwen3.5-9B, Qwen3.5-4B × {E2, E3, E4, W-freq}; E3 math-format, E3 code-format; E3 self-test off | 11 / 22 / 77 | — |
| Hyperparameter sensitivity | temperature 0.2, 1.0 × {E3, W-freq} | 4 / 8 / 28 | grader and explainer settings re-scored |
| Seed sensitivity | T1, T2 × {E2, E3, E4, W-freq}; T-new × {E3, E4, W-freq}; T-paper × {E-paper, E1, E3, E4, W-freq} | 16 / 32 / 112 | T-new trained (after freeze); redrawn W-freq/W-scramble; grader fold seeds |
| **Total agent runs** | 40 conditions | **40 / 80 / 280** (400 total) | |

Estimated GPU time at ~20 min per run: round 1 ≈ 13 h, round 2 ≈ 27 h, round 3 ≈ 90 h. Training jobs take minutes.

## Training jobs (all with the paper's official recipe unless stated)

| Job | Data / init | Steps | When |
|---|---|---|---|
| Reproduction of T0 | `wd1` seed 1 | 40,000 | stage 0 |
| Planted calibration | reference-compiled E3 model for T0; full-batch AdamW, task labels | 10,000 (snapshots 0/100/1,000/10,000) | round 1, controls |
| Distilled students ×3 | fresh inits (seeds 1000–1002); loss = KL to T0's logits on all 113² inputs | 20,000 | round 1, baselines |
| T-new | init = split seed 11 | 40,000 | round 1, before seed sensitivity |

## Postponed (not in this plan)

Claude Haiku 5.5 and Claude Opus 5.5 as simulators in Claude Code; other toy models (Clock/Pizza p = 59, the paper's
2-layer and LayerNorm variants, ARENA monthly-problem models).
