# Experiments — summary of everything run so far

*Every number in the tables below is generated from `results/` by `vmi analyse <round>`
(`src/vmi/analysis/report.py`); nothing in a marked block is typed by hand. Verdicts carry their N: at N = 1 there is
no within-condition spread, so they rest on directions and on the deterministic no-agent arms.*

| # | Experiment | Status | Report |
|---|---|---|---|
| 00 | Stage 0 — tooling and smoke tests on the dev target (never reported as results) | done | [00-stage0-dev/experiment.md](00-stage0-dev/experiment.md) |
| 01 | Round 1 (N = 1) — main method → controls → baselines → ablations → hyperparameter → seed sensitivity | running | [01-round1/experiment.md](01-round1/experiment.md) |
| 02 | Round 2 (N = 3) | planned | – |
| 03 | Round 3 (N = 10) | planned | – |

Learnings across experiments: [learnings.md](learnings.md).

## For a layman

We take a tiny neural network that has learned to add numbers modulo 113, and a written explanation of how it does
that. We give only the explanation to an AI agent that cannot see the network or the internet, and ask it to write
down every weight of a new network. Then we compare the two networks: do they give the same answers (output KL), and
do they compute the same intermediate values (hidden-state similarity, HS)? If an explanation is good, the rebuilt
network should match on both. If it is wrong, the rebuilt network may still give the right answers — there are many
ways to add numbers — but its internals should not match. The experiments test whether this score really behaves
like that: does it rise with true detail, fall for wrong explanations, and catch wrong mechanisms that still give
correct answers?

## Key terms

| Term | Meaning |
|---|---|
| **M**, target | A trained one-layer transformer for (a + b) mod 113 ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217) architecture: d_model 128, 4 heads, d_mlp 512, no LayerNorm). |
| **M̂**, replacement | The network whose weights the agent wrote (same architecture, `weights.npz`). |
| **KL_out** | Mean KL(P_M ‖ P_M̂) over all 113² = 12,769 inputs, in nats (float64). Lower = more similar behaviour. |
| **HS** | Hidden-state similarity: 5-fold cross-validated linear-regression R² in both directions (M̂ → M and M → M̂) at three sites (residual stream after attention, MLP activations, final residual stream), harmonic mean per site, averaged over sites. Invariant to any invertible linear change of the residual basis. Higher = more similar internals. |
| **Matched-neuron corr** | Secondary metric: mean correlation of MLP neurons after Hungarian matching; sensitive to per-neuron assignments, which HS is not. |
| **Key frequencies** | The few Fourier frequencies w_k = 2πk/113 the model uses ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217), Sec. 4). |
| **E0–E4** | Explanation ladder of increasing true detail: E0 task only; E1 algorithm without model-specific numbers; E2 key frequencies and component roles; E3 + per-frequency amplitudes, attention fits, neurons per frequency and the precise neuron function; E4 + per-neuron table. |
| **W-freq / W-alg / W-role / W-scramble** | Controls: E3 with wrong key frequencies; a different algorithm (the "Pizza" algorithm of [Zhong et al. 2023](https://arxiv.org/abs/2306.17844)); E3 with component roles misattributed; E4 with each neuron's frequency reassigned at random. |
| **E-paper** | The published explanation of the paper's own checkpoint (T-paper only). |
| **Reference compiler** | A deterministic script that turns an explanation's parameter file into weights — the agent-free ceiling for what each explanation contains. |
| **Simulator** | The agent that writes the weights: Claude Haiku 5.5 in Claude Code (main), or local Qwen models in a bash-only harness (capability ablation). |
| **N** | Agent runs per condition (replicates; for the local models the index is the sampling seed, Claude cannot be seeded). Round 1: N = 1; round 2: N = 3; round 3: N = 10. |

## Networks (targets)

All targets use the paper's training recipe (full-batch AdamW, lr 10⁻³, weight decay 1, 30 % training split,
40,000 steps) and come from [BurnyCoder/grokking-modular-addition-runs](https://huggingface.co/BurnyCoder/grokking-modular-addition-runs),
except T-new (trained here after the protocol freeze with the same recipe) and T-paper (the paper's checkpoint).

<!-- BEGIN targets -->
| Target | run | role | key frequencies | neurons per key frequency | unclustered neurons | logit fit FVE | loss (all 113² inputs) |
|---|---|---|---|---|---|---|---|
| T0 | `wd1_s1` | main target | 2, 3, 23, 37, 39 | 0, 197, 49, 42, 99 | 125 | 0.933 | 2.06e-07 |
| T1 | `wd1_s4` | seed sensitivity (fewest frequencies) | 11, 25, 49 | 154, 111, 156 | 91 | 0.984 | 1.36e-07 |
| T2 | `wd1_s8` | seed sensitivity (most frequencies) | 37, 44, 45, 46, 47, 51 | 58, 63, 81, 78, 88, 79 | 65 | 0.949 | 8.42e-07 |
| T-new | `new_s11` | held-out, trained after the protocol freeze | 22, 37, 41, 51 | 138, 77, 79, 130 | 88 | 0.954 | 2.41e-07 |
| T-paper | `paper` | published checkpoint (contamination-flagged) | 14, 35, 41, 42, 52 | 44, 93, 145, 87, 64 | 79 | 0.961 | 2.41e-07 |
| dev | `wd1_s10` | development only (never reported) | 19, 31, 45, 47 | 80, 81, 189, 79 | 83 | 0.970 | 1.77e-07 |
<!-- END targets -->

## Simulators

| Simulator | Harness | Sampling | Limits | Role |
|---|---|---|---|---|
| Claude Haiku 5.5 (`claude-haiku-5-5`) | Claude Code 2.1.330, `--bare`, offline sandbox, API via a key-injecting proxy outside the sandbox | model default (Claude 5.5 rejects non-default sampling); effort `medium` | 60 min, $2 per run, $20 total | main simulator (preregistration amendment 2) |
| Qwen3.6-35B-A3B (UD-IQ4_XS) | custom bash-only harness (mini-SWE-agent design), llama-server | Qwen3.6 coding preset: T 0.6, top-p 0.95, top-k 20 | 60 min, 80k context, 16k thinking tokens per turn | capability ablation |
| Qwen3.5-9B (Q4_K_M) | same | same | same | capability ablation |
| Qwen3.5-4B (Q4_K_M) | same | same | same | capability ablation |

Why these settings: the Qwen preset is the model card's coding preset; the context and thinking budgets were the
smallest that removed the truncation and compaction failures seen in stage 0 (`learnings.md`). Claude Haiku was
chosen by a rule fixed before any Claude run: try Haiku → Sonnet → Opus only while every cheaper simulator stays
below 5 % task accuracy (chance is 1/113 ≈ 0.9 %); Haiku met it in its first run, so Sonnet and Opus were not used.

## Evaluation

Every replacement is graded against its target on all 12,769 inputs (no train/test split is needed: the agent never
sees the target's outputs, only the explanation; the self-test checks accuracy against the task labels
(a + b) mod 113). Build scripts are audited for training loops and iterative optimisation (forbidden), and flagged
scripts are reviewed by hand (`results/audit_reviews.json`).

## All agent runs so far

<!-- BEGIN runs -->
| Stage | Target | Explanation | Simulator | T | Variant | N | failed | accuracy | KL_out ↓ (mean ± sd) | HS ↑ (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ablations | T0 | E2 | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 4.73 | 0.000 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-code | 1 | 0 | 1.000 | 0.00212 | 0.918 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-math | 1 | 0 | 1.000 | 0.000716 | 0.911 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | noselftest | 1 | 0 | 1.000 | 0.00134 | 0.964 |
| controls | T0 | E0 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.00908 | 0.004 |
| controls | T0 | W-freq | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000931 | 0.055 |
| controls | T0 | W-alg | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.0146 | 0.889 |
| controls | T0 | W-role | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000109 | 0.168 |
| controls | T0 | W-scramble | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.00283 | 0.918 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-high | 1 | 0 | 1.000 | 0.00152 | 0.923 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-low | 1 | 0 | 0.562 | 1.97 | 0.922 |
| hyper | T0 | W-freq | claude-haiku-5-5 | default | effort-low | 1 | 0 | 0.770 | 1.03 | 0.114 |
| main | T0 | E1 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000105 | 0.048 |
| main | T0 | E2 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 1.2e-05 | 0.899 |
| main | T0 | E3 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.0229 | 0.939 |
| main | T0 | E4 | claude-haiku-5-5 | default | – | 1 | 0 | 0.979 | 0.0872 | 0.912 |
<!-- END runs -->

## Claude API spend

<!-- BEGIN cost -->
| Model | runs (incl. checks) | requests | spend (USD) |
|---|---|---|---|
| claude-haiku-5-5 | 23 | 176 | 0.424 |
| **total** | 23 | 176 | **0.424** |
<!-- END cost -->
