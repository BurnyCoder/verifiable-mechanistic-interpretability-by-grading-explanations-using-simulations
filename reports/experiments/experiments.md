# Experiments — summary of everything run so far

*Every number in the tables below is generated from `results/` by `vmi analyse <round>`
(`src/vmi/analysis/report.py`); nothing in a marked block is typed by hand. Verdicts carry their N: the current ones are at
N = 3 (round 2) with the preregistered tests; round 1's N = 1 verdicts rested on directions and on the deterministic
no-agent arms.*

| # | Experiment | Status | Report |
|---|---|---|---|
| 00 | Stage 0 — tooling and smoke tests on the dev target (never reported as results) | done | [00-stage0-dev/experiment.md](00-stage0-dev/experiment.md) |
| 01 | Round 1 (N = 1) — main method → controls → baselines → ablations → hyperparameter → seed sensitivity | done | [01-round1/experiment.md](01-round1/experiment.md) |
| 02 | Round 2 (N = 3) — the same stages, two more replicates per condition, Claude Haiku only | agent runs done; H8 trajectory re-run as preregistered (still supported); redrawn W-freq/W-scramble and T-paper's extra retrains not run (deviation 8); second audit: post-hoc H8 negative controls, pooled E0/E1 memorisation analysis, transcript counts of self-test tuning, exact trend test (deviation 9); third audit: verdicts recomputed independently, H1's E0 clause and the T-paper replication corrected, exact H1 trend test, own label checks counted, Clock/Pizza diagnostics (deviation 10); fourth audit: independent regrade and tests, H4b and H8 re-read (deviation 11); fifth audit: every verdict re-derived with exact methods, T0's read-out-only frequency 2, scrambled-table blind spot narrowed to the agents' builds (deviation 12); sixth audit: verdicts re-derived and scores re-graded with new code, H2's literal §5 reading reported beside "cannot tell", the agents' parameter searches counted by reading, citation fixes (deviation 13); seventh audit: verdicts re-derived without project code or scipy, searches inside 7 of 96 final build scripts disclosed (deviation 14); eighth audit: the preregistered verdict leads (H2 "not supported" at N = 3 by §5), H4b stated as falsified, H7's scope, the self-test ablation's leftover cue disclosed (deviation 15) | [02-round2/experiment.md](02-round2/experiment.md) |
| 03 | Round 3 (N = 10) | postponed to a later day (user decision, deviation 6) | – |

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
| **HS** | Hidden-state similarity: 5-fold cross-validated ridge-regression R² (λ = 10⁻⁶·tr(XᵀX)/d) in both directions (M̂ → M and M → M̂) at three sites (residual stream after attention, MLP activations, final residual stream), harmonic mean per site, averaged over sites. Completeness (M̂ → M) is invariant to invertible linear maps of the replacement's basis, soundness (M → M̂) only to orthogonal maps and uniform scaling; HS = 1 (up to the ridge term) under any exact invertible linear equivalence. Higher = more similar internals. |
| **Matched-neuron corr** | Secondary metric: mean correlation of MLP neurons after optimal one-to-one matching (linear assignment, scipy `linear_sum_assignment`); invariant only to the MLP's own symmetries (permutation, positive scaling). It separated the agents' builds of the scrambled neuron table from E4 at N = 3, which HS did not (the compiler's literal scrambled build does lose 0.09 HS on T0). |
| **Key frequencies** | The few Fourier frequencies w_k = 2πk/113 the model uses ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217), Sec. 4). |
| **E0–E4** | Explanation ladder of increasing true detail: E0 task only; E1 algorithm, including the precise neuron function, without model-specific numbers; E2 key frequencies and component roles; E3 + per-frequency amplitudes, attention fits, neurons per frequency and the logit scale; E4 + per-neuron table. |
| **W-freq / W-alg / W-role / W-scramble** | Controls: E3 with wrong key frequencies; a different algorithm (the "Pizza" algorithm of [Zhong et al. 2023](https://arxiv.org/abs/2306.17844)); E3 with component roles misattributed; E4 with each neuron's frequency reassigned at random. |
| **E-paper** | The published explanation of the paper's own checkpoint (T-paper only). |
| **Reference compiler** | A deterministic script that turns an explanation's parameter file into weights — what a literal, agent-free build of each explanation yields (a reference point, not a ceiling: the agent exceeds it on some targets). |
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
| Qwen3.6-35B-A3B (UD-IQ4_XS) | custom bash-only harness (mini-SWE-agent design), llama-server | Qwen3.6 thinking-mode coding preset: T 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0 | 60 min, 80k context, 16k thinking tokens per turn | capability ablation; round 1 not run (stopped, deviation 5) |
| Qwen3.5-9B (Q4_K_M) | same | same | same | capability ablation; 1 run (E2), at chance, then stopped |
| Qwen3.5-4B (Q4_K_M) | same | same | same | capability ablation; 4 runs, all at chance |

Why these settings: the Qwen preset is the model card's coding preset; the context and thinking budgets were raised
after stage-0 truncation and compaction failures but did not remove them all (smoke run 6 still truncated a whole-file
write, `learnings.md`); the remaining failures (dropped truncated replies, session-ending errors, early compaction)
were removed by the custom bash-only harness (preregistration amendment 2 item 4). Claude Haiku was
chosen by a rule fixed before any Claude run: try Haiku → Sonnet → Opus only while every cheaper simulator stays
below 5 % task accuracy (chance is 1/113 ≈ 0.9 %); Haiku met it in its first run, so Sonnet and Opus were not used.

## Evaluation

Every replacement is graded against its target on all 12,769 inputs (no train/test split is needed: the agent never
sees the target's outputs, only the explanation; the self-test checks accuracy against the task labels
(a + b) mod 113). Build scripts are audited for training loops and iterative optimisation (forbidden), and flagged
scripts are reviewed by hand (`results/audit_reviews.json`). The audit's patterns catch gradient descent and
optimisers, not small derivative-free searches: reading every final script (audit 7, deviation 14) found <!-- v:a7-search -->7 of 96<!-- /v --> that
try several values of one constant and keep the best, none in a T0 run that a hypothesis test uses
(`results/audit7/build_script_searches.json`).

## All agent runs so far

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

## Claude API spend

<!-- BEGIN cost -->
| Kind of run | model | runs | requests | spend (USD) |
|---|---|---|---|---|
| reported runs | claude-haiku-5-5 | 96 | 881 | 2.200 |
| development runs (stage 0, dev target) | claude-haiku-5-5 | 6 | 68 | 0.128 |
| plumbing checks | claude-haiku-5-5 | 2 | 4 | 0.000 |
| **total** | | 104 | 953 | **2.328** |
<!-- END cost -->
