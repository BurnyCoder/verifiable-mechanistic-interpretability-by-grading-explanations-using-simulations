# Plan

*Approved plan, copied verbatim from the planning session on 2026-11-21. Progress against it: [progress.md](progress.md).*

## Changes since approval (kept up to date; the approved plan below is unchanged for the record)

| When | Change | Why | Where recorded |
|---|---|---|---|
| Stage 0 | Throughput fallback rules applied; context, thinking budget and sampling preset of Qwen3.6-35B-A3B tuned on the dev target | stage-0 smoke runs | amendment 1, `experiments/learnings.md` |
| Stage 0 | OpenCode replaced by a custom bash-only harness (mini-SWE-agent design) for the local models | OpenCode truncation/compaction failures; your suggestion | amendment 2 |
| Stage 0 | **Claude Haiku 5.5 in Claude Code becomes the main simulator** (API credits, $20 cap, Haiku → Sonnet → Opus only while the cheaper one stays at chance); local Qwens become the capability ablation (3 models × 4 explanations) | every local Qwen stayed at chance on the dev target; your instruction | amendment 2, memory note on the Claude budget |
| Stage 0 | Temperature sweep replaced by Claude effort low/high (Claude 5.5 accepts no other sampling) | API constraint | amendment 2 |
| Round 1 | 44 conditions per round instead of 40 | amendment 2 | `planned-runs.md` |
| Round 1 | Four post-freeze deviations: analysis code aligned with amendment 2; local Qwen ablations in parallel with later Haiku stages; one infrastructure failure re-run (unix socket path limit); NaN stitching metric repaired | bugs and pacing | end of `preregistration.md` |
| Round 1 | Post-hoc memorisation analysis (frequencies built into each replacement) | the simulator fills gaps with the paper's published frequencies | `experiments/01-round1/experiment.md` |
| Round 1 | Results explorer on GitHub Pages; Hugging Face repo and collection | plan deliverables | README |
| Round 1 | Post-hoc target swap: the same E0/E1 weights graded against the published checkpoint | the most direct memorisation test | `experiments/01-round1/experiment.md` |
| Round 1 | **Local Qwen runs stopped** after 5 of 12 (4B × 4, 9B × 1), all at chance; no local conditions in rounds 2–3, so 32 conditions per round; H5/H6 not evaluable | your instruction ("stop doing Qwen runs if they don't work") | deviation 5 in `preregistration.md` |
| Round 2 | **Round 3 (N = 10) postponed** to a later day; round 2 (N = 3) completes and its verdicts are reported as "at N = 3 — not yet N = 10" | your instruction ("Postpone N=4 - N=10 to some other day") | deviation 6 in `preregistration.md` |

Round 2 (N = 3) runs the 32 Claude Haiku conditions; round 3 (N = 10) is postponed to a later day and starts only when
you ask for it.


## Context (the plan addresses the user as "you")

Agents can now mass-produce interpretability explanations, so the bottleneck moves to **checking** them automatically. The strongest rung of the evidence ladder is *simulation*: rebuild the mechanism well enough to reproduce the model. This project builds and validates a grader for that rung:

**explanation (text / math / code) → a local-LLM agent in OpenCode, fully offline and never shown the original weights, writes the weights of a replacement model → score = output KL divergence + hidden-state similarity between original and replacement.**

Target: the 1-layer modular-addition transformer of Nanda et al. 2023 ([arXiv 2301.05217](https://arxiv.org/abs/2301.05217)). The question is **not** "is the paper's explanation right" but **"is this simulation score a valid measure of explanation quality?"** — does it rise with true detail, fall for false explanations, and does hidden-state similarity catch wrong mechanisms that output KL cannot see.

**Your decisions so far:** scripted explanation ladder + deliberately wrong controls · public GitHub/HF repos from the first push · local agent model Qwen3.6-35B-A3B on `llama-server` (Qwen3.8 if usable) · the paper's public checkpoint as a second target · **reuse existing trained models and code** (ARENA and others) where it helps · **train models where that helps, including to validate** · controls, baselines, ablations, hyperparameter sensitivity and seed sensitivity as **separate stages in that order** · **N=1 of everything first, then N=3, then N=10** · **Victor Levoso is coauthor** of the paper · the write-up **credits the hackathon and community ([mechinterp.com](https://mechinterp.com/))** it was developed in · **all Claude Code work (Haiku, Opus, subscription or API credits) is postponed to another day** and is not part of this plan.

**Qwen3.8 check (done):** no usable Qwen3.8 exists for 8 GB VRAM + 31 GB RAM. [Qwen3.8-27B](https://hf.co/Qwen/Qwen3.8-27B) is dense (the same-size dense Qwen3.6-27B was reported at ~0.75 tok/s on 8 GB); [Qwen3.8-Flash-Next](https://hf.co/Qwen/Qwen3.8-Flash-Next) is a 125–180B MoE whose smallest GGUF is ~68 GiB; there is no official Qwen3.8-35B-A3B (only an unofficial third-party distill into Qwen3.6-35B-A3B, not used). → **Primary = Qwen3.6-35B-A3B.**

**Positioning (from the literature pass; every source is re-verified before it is cited):** closest work is iWSP ([arXiv 2611.18114](https://arxiv.org/abs/2611.18114), Nov 2026: a fresh LLM session rebuilds weights from a ≤200-word description, but only for agent-written models and with no KL or hidden-state comparison); Leo Gao's post proposes simulating *activations* and scoring by downstream KL; Cammarata et al. 2021 hand-built one curve circuit. The claimed novelty is the **combination**: SGD-trained target, weights-blind agent, output KL + symmetry-aware hidden-state similarity, explanation-length accounting, no-explanation and wrong-explanation controls.

**Machine (verified):** RTX 5070 Laptop 8 GB VRAM, 31 GB RAM; OpenCode 2.3.2 (Windows, and a native Linux binary in WSL at `~/.opencode/bin`), Ollama (qwen3.5:9b, :4b), uv, gh + hf logged in as BurnyCoder, MiKTeX, pandoc; WSL2 Ubuntu 24.04 with bubblewrap 0.9.0, mirrored networking. Tested: `bwrap --unshare-net` blocks the internet and hides `/mnt/c` and `/home`; WSL reaches Windows `127.0.0.1`. No passwordless sudo, no Docker → everything below is user-level. C: has 38 GB free → models and caches go to `D:\AI`.

## Pipeline

```
target model M ─┬─► explainer script ──► EXPLANATION.md ──►┌─ sandbox: WSL + bwrap, no network, no path to M ─┐
 (weights)      │   (Fourier analysis;   (+ data appendix) │ OpenCode + local Qwen3.6-35B-A3B, offline        │
                │    ladder + wrong)                       │   ─► build_weights.py                            │
                │                                          │ given: ARCHITECTURE.md, forward.py, selftest.py  │
                │                                          └───────────────────────────┬──────────────────────┘
                │                                                                weights.npz (M̂)
                └──────────────────────────────────────────► grader ◄──────────────────┘
                                                              KL(M‖M̂) over all 113² inputs + hidden-state similarity
 controls:  no explanation · wrong explanations · random init · wrong target · planted calibration · training trajectory
 baselines: independent retrain · distilled student · reference compiler · Fourier-restricted M · weights dump
```

## Components

### 1. Target models — reused, with a little training where it validates
Your earlier project already published what this one needs, so targets and baselines are reused and this project writes no model code. Training is added only where it validates the method (see "Training" below).

- **Models:** [`BurnyCoder/grokking-modular-addition-runs`](https://hf.co/BurnyCoder/grokking-modular-addition-runs) (MIT, safetensors, no pickle): 22 one-layer transformers trained deterministically with the paper's official recipe (full-batch AdamW, lr 1e-3, betas (0.9, 0.98), 10-step warmup, 30% split, float64 loss, 40,000 steps), each with all 401 checkpoints and its own `analysis/key_freqs.json`. Pinned to a fixed revision.
- **Code:** the `grokviz` package from [`BurnyCoder/visualizing-grokking-modular-addition-in-transformer`](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer) (MIT), added as a pinned git dependency. It already re-implements the paper's and ARENA's code with a porting map: `GrokTransformer` (paper's parameter names, hooks, `run_with_cache`), `RunStore`, the Fourier basis and transforms, key frequencies, neuron clusters, restricted/excluded loss, attention-pattern fits (paper Table 2), the W_L read-off (paper Table 1), float64 loss, and the download + `weights_only=True` conversion of the paper's checkpoint.
- **Targets, chosen by rules fixed before any run** (key frequencies from the published `key_freqs.json`):

| Role | Run | Key frequencies | Rule |
|---|---|---|---|
| **T0** main | `wd1_s1` | 2, 3, 23, 37, 39 | first seed whose frequencies differ from the paper's |
| **T1** seed sensitivity | `wd1_s4` | 11, 25, 49 | fewest key frequencies |
| **T2** seed sensitivity | `wd1_s8` | 37, 44, 45, 46, 47, 51 | most key frequencies |
| **dev** (never reported) | `wd1_s10` | 19, 31, 45, 47 | last seed |
| **T-paper** (contamination-flagged) | the paper's own checkpoint, epoch 40,000 | 14, 35, 41, 42, 52 | published model |

- `wd1_s0` reproduces the paper's exact frequency set (14, 35, 41, 42, 52), so it is **not** a clean target; it serves as the paper model's independently trained "twin" in the baselines. The runs were published in October 2026 and never appeared in a paper, so the agents cannot have memorized T0–T2; the E0/E1 arms check this.
- A target must have 100% test accuracy at step 40,000; otherwise the next seed by the same rule is used.

**Training — four small jobs, each a few minutes on this GPU, all with the official recipe:**

| Job | What is trained | What it validates |
|---|---|---|
| Reproduction check | T0's exact spec again with `grokviz.train_runs` (deterministic) | The published target is what its card says: same key frequencies, weights equal within tolerance |
| **T-new**, held-out target | A never-published seed (init = split = 11; next seed if it does not reach 100% test accuracy), trained **after** the protocol freeze | The whole pipeline on a model nobody, including me during development, could have tuned to; strictest contamination control |
| Distilled students | 3 fresh models per main target trained to match M's logits on all 113² inputs | The strongest explanation-free baseline: does copying behaviour (full logits) already reproduce the hidden states? |
| Planted-mechanism calibration | The reference-compiled E3 model for T0, trained further and snapshotted at 0, 100, 1,000 and 10,000 steps | A target whose mechanism is known by construction: the score must be perfect at step 0 and fall as the model drifts from the planted explanation |

Fresh seeds use `grokviz.train_runs` unchanged (resumable, densely logged). Distillation and continued training need a pluggable loss and initial weights, so they use one small loop in `training.py` that must first reproduce a published run in plain label mode. Each job runs in the stage that uses it: the reproduction check in stage 0, and in round 1 the planted calibration in the controls stage, the distilled students in the baselines stage, and T-new just before the seed-sensitivity stage. Training runs while `llama-server` is unloaded, or on CPU, so it never competes with the agent queue.

**Reuse decisions (checked against the sources and their licences):**

| Resource | Licence | Decision |
|---|---|---|
| Your 22 runs + `grokviz` | MIT | **Use** as targets, baselines, controls and analysis code |
| ARENA 1.5.2 checkpoint (`arena-education/grokking_full_run_data`) | none stated | It **is** the paper's mainline run (same SHA-256 as the copy already planned) → used as T-paper; not redistributed |
| ARENA 1.5.2 code, paper repo code | no licence | Not copied; `grokviz` already re-implements both with a porting map, and ARENA's published numbers serve as test oracles |
| TransformerLens | MIT | Not needed: v4 (Sept 2026) removed `HookedTransformer`, and `GrokTransformer` loads the paper's layout directly |
| Similarity metrics | netrep MIT, Google CKA Apache-2.0, ReSi CC-BY | Own small implementations (regression R², CKA, Procrustes, Hungarian matching), cross-checked in tests against netrep and Google's CKA; no maintained PyPI package covers all four |
| Other trained toy models: Clock/Pizza p=59 (MIT), the paper's 2-layer and LayerNorm variants (`smithjessk/grokking-saved-runs`), ARENA's monthly-problem models with written solutions (unlicensed) | mixed | **Later targets** for "does it generalize to other toy transformers"; listed in the to-do, outside this run matrix |

### 2. Explainer — scripted ladder (has weights access)
`explainer/analysis.py` is a thin layer over `grokviz` (key frequencies, neuron clusters, attention fits, read-off coefficients) plus the per-neuron phase/gain fits that E4 needs; it emits a parameter JSON, and `render.py` turns it into markdown (+ CSV appendix for E4). No number is hand-typed.

| ID | Content | Truth |
|---|---|---|
| E0 | task only: "computes (a+b) mod 113" + architecture (no-explanation control; measures what the LLM does from priors) | true, no mechanism |
| E1 | algorithm sketch: Fourier multiplication via trig identities, no model-specific numbers | true |
| E2 | key frequencies + role of embedding / attention / MLP / unembedding | true |
| E3 | + per-frequency amplitudes, neurons per frequency, attention-pattern fits, logit scale | true |
| E4 | + per-neuron table (frequency, phases, gains) as a CSV appendix, per-head summary | true |
| W-freq | E3 with the key frequencies swapped for other valid ones | false mechanism, correct behaviour |
| W-alg | E3-length description of a different algorithm | false |
| W-role | E3 with component roles misattributed | false |
| W-scramble | E4 with each neuron's frequency reassigned at random | false |
| E-paper | (T-paper only) the published explanation, paraphrased faithfully with the paper's Tables 1–2 numbers | as published |

Each explanation's length is recorded (characters, gzip bytes, count of numbers); scores are always shown against length, with the weights dump as the degenerate endpoint.

### 3. Simulator — a local agent in a weights-blind, offline sandbox
The study uses one simulator harness, fully offline.

**Sandbox (WSL Ubuntu):** `bwrap --unshare-net --clearenv`, only the run folder writable, no `/mnt`, `/home` or `/run`; the model endpoint is reached through a unix socket.
- **Contents:** `TASK.md`, `EXPLANATION.md` (+ appendix), `ARCHITECTURE.md`, `forward.py` (numpy forward pass), `selftest.py` (shape checks + accuracy against the task labels (a+b) mod p — never against M). Required output: `build_weights.py` + `weights.npz`.
- **Rules:** numpy only; no training loops or iterative optimisation (static audit + manual review of flagged scripts); build script ≤ 60 s; wall-clock and step caps per run.
- **Isolation self-check** logged before every batch: external HTTP/DNS/TCP fail, only `lo` exists, `/mnt/c` and M's path are absent, Windows interop is unreachable, AF_VSOCK is blocked by a seccomp filter. Threat model: prevent accidental access to weights or internet, not a malicious model.
- **Logging:** event stream, session transcript, raw wire log (auth headers never logged) and a readable timestamped transcript per run, echoed to the terminal.

**The local agent — OpenCode + Qwen3.6-35B-A3B.**
- `llama-server` on Windows (GPU) with `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `--n-cpu-moe` (experts in RAM), 64k context, fixed model file (no pull endpoint), per-run sampling seed. A stdlib Python relay bridges the socket.
- `opencode run --standalone --auto --format json --thinking --agent weights`; config denies web tools, MCP, subagents, code mode and updates and allows only the local provider. 40-minute cap, 60 steps.
- Uses the OpenCode 2.3.2 Linux binary you installed in WSL, bound read-only into the sandbox so it cannot update itself mid-study; the version is recorded with every run and the study does not continue on a different one.

### 4. Grader
- **Output fidelity — primary:** mean KL(P_M ‖ P_M̂) over all 113² inputs, 113 number classes, nats, float64. Secondary: argmax agreement, task accuracy, centred-logit FVU, temperature-calibrated KL.
- **Hidden-state similarity — primary:** at three final-position sites (residual after attention, MLP post-activations, final residual), 5-fold held-out linear-regression variance explained in both directions (M̂→M "completeness", M→M̂ "soundness"), combined by harmonic mean and averaged over sites. A linear map is the alignment class because, without LayerNorm, the residual stream is symmetric under any invertible linear map; N=12,769 inputs ≫ 128 dims keeps it non-degenerate.
- **Secondary:** Hungarian-matched neuron correlation, head-matched attention-pattern distance, linear CKA, orthogonal Procrustes, embedding Fourier-spectrum overlap, and **stitching KL** (map M̂'s hidden state linearly into M, run the rest of M, KL on held-out inputs — a causal test).
- A run with no valid weights is a failure: reported as a failure rate and scored as the all-zero-weights model; analyses are given with and without failures.

### 5. After the main method — five separate stages, in this order
The main method is the true explanation ladder E1–E4 on T0. Everything else follows as five separate stages, each with its own push, experiment report and look at the results before the next begins. The number of agent runs per condition is the same everywhere and grows in rounds (1, then 3, then 10; see the run matrix).

**5.1 Controls** — what should score badly if the grader works.
- Agent: T0 × {E0 (no explanation), W-freq, W-alg, W-role, W-scramble}.
- No agent: random-init models (5 seeds, the floor) · wrong target (every T0 replacement graded against the other targets and against models with no clean mechanism, `wd0_*` and `rand_*`: an explanation of one model must not fit another) · planted-mechanism calibration series (trained; see "Training") · training trajectory (T0 replacements graded against all 401 checkpoints of T0, next to the paper's restricted and excluded loss).

**5.2 Baselines** — what you get without an agent, or without an explanation.
- Independent retrains (the 9 other `wd1` seeds: what the training recipe alone buys; for T-paper also its twin `wd1_s0` and the three same-split, different-init runs) · distilled students (trained; behaviour-only clones of M) · **reference compiler** for every explanation (deterministic script: explanation parameter JSON → weights; the agent-free ceiling that separates explanation quality from agent skill) · Fourier-restricted M and the weights dump (ceilings and the far end of the length axis).

**5.3 Ablations** — remove or swap one part of the method.
- Local simulator size (qwen3.5:9b, qwen3.5:4b on the same server) × {E2, E3, E4, W-freq} · explanation format at fixed E3 content (math and code versus prose) · self-test tool off on E3.

**5.4 Hyperparameter sensitivity** — do the conclusions survive other reasonable settings?
- Agent: sampling temperature 0.2 and 1.0 (frozen default 0.6) × {E3, W-freq}; sensitivity to the run budget is read from existing transcripts (score against steps and time used), with no new runs.
- Grader (no new runs): ridge strength, number of folds, choice of sites, orthogonal versus general linear alignment, and how failed runs are scored.
- Explainer (no agent): the key-frequency and neuron-cluster thresholds are varied, and the explanations re-rendered and re-compiled.

**5.5 Seed sensitivity** — do the conclusions survive other random draws?
- Target seeds: T1, T2 × {E2, E3, E4, W-freq} · held-out T-new × {E3, E4, W-freq} · the published model T-paper × {E-paper, E1, E3, E4, W-freq}.
- Agent seeds: variance across the replicates per condition, reported as intervals and as a split between condition and seed variance.
- Other draws (no agent): W-freq and W-scramble redrawn with 3 seeds each, grader fold seeds, random-init seeds.

### 6. Postponed to another day — all Claude Code work
Not part of this plan: nothing below is built, tested or run now, and nothing is sent to Anthropic by this project. **It stays postponed after round 3 as well** — I do not start it when the rounds finish; it waits until you ask for it. It is recorded in the repo's to-do so the design is not lost.

- **Claude Haiku 5.5 as a second simulator** in Claude Code, on the same matrix as the main-method, controls and seed-sensitivity stages; **a few Claude Opus 5.5 runs** up to a $10-equivalent ceiling.
- **Auth, as you decided:** your subscription's limits by default (the claude.ai login in WSL, runs strictly sequential and paced below a share of the 5-hour and weekly limits), with an API-credit mode kept as a switchable option.
- **Isolation:** the same weights-blind sandbox, with egress allowed only to Anthropic's API hosts; web and server-side tools denied and audited from the transcript.
- **What it would add:** a cross-family test that different simulators rank the explanations the same way, a longer simulator-capability axis, and a harness control (the local model driven through Claude Code).
- **Open points to settle then:** headless limit behaviour, `--restricted` with a subscription login, the exact hosts needed, and the subscription terms' "ordinary, individual usage" wording for a few hundred scripted runs.

## Hypotheses to preregister (with falsifiers)
- **H1 validity:** both primary scores improve monotonically E1→E4 (with E0, from the controls stage, below E1). Falsified if the trend test's CI includes zero.
- **H2 discrimination:** true explanations beat length-matched false ones (E3 vs W-freq/W-alg/W-role, E4 vs W-scramble) on hidden-state similarity. Falsified if the difference CI includes zero.
- **H3 behaviour ≠ mechanism:** the W-freq replacement matches M's behaviour (≥99% argmax agreement, KL within the range of the true E2+ explanations — a grokked model's outputs are near one-hot, so output KL ≈ the replacement's task loss) while its hidden-state similarity is below every true E2+ explanation. Falsified if output KL alone ranks W-freq below all of them.
- **H4 added value:** E3/E4 replacements beat independent retrains on hidden-state similarity. Open question, tested two-sided with no predicted direction: do they also match a logit-distilled student, which has full behavioural access but no explanation?
- **H4b calibration:** on the planted-mechanism series the score is perfect at step 0 and decreases monotonically with further training. Falsified if it does not, which would mean the grader itself is unreliable.
- **H5 simulator gap:** agent score ≤ reference-compiler score, and the gap shrinks with simulator capability (qwen3.5:4b → 9b → Qwen3.6-35B-A3B).
- **H6 simulator invariance (secondary):** the three local models rank {E2, E3, E4, W-freq} the same way. With only four explanations this is a weak test; if the rankings disagree, the score describes the agent as much as the explanation. The stronger cross-family version is postponed with the Claude work.
- **H7 specificity:** a replacement built from T0's explanation scores higher on hidden-state similarity against T0 than against any other target or no-mechanism model. Falsified if a wrong target scores as high.
- **H8 trajectory (secondary):** against T0's checkpoints, the E3/E4 replacement's hidden-state similarity rises through circuit formation and cleanup, tracking the fall of restricted loss.

Statistics: every condition is run in three rounds — **N=1 of everything first, then N=3 (2 more), then N=10 (7 more)** — with run seeds 0, then 1–2, then 3–9 fixed per condition. **Every hypothesis is evaluated after every round**, with the same preregistered analysis, and each verdict is labelled with its N: "at N=1 — not yet N=3 or N=10", "at N=3 — not yet N=10", "at N=10". The write-up keeps all three in one table (supported / not supported / cannot tell at this N), so it shows how each verdict moved. What each N can support is stated plainly: at N=1 there is no within-condition spread, so verdicts rest on directions, the trend across conditions and the deterministic no-agent arms; at N=3 intervals exist but are wide, and a two-group rank test cannot go below p = 0.1; N=10 is the full analysis. Nothing in the protocol or analysis changes between rounds, and the study stops on time, never on results; if it stops before N=10, the last completed verdicts stand with their N. Bootstrap 95% CIs; Jonckheere/Spearman trend test (H1); Mann–Whitney with Holm correction; effect sizes; anything not preregistered is labelled post-hoc. Preregistration v1 (hypotheses, metrics, analysis) is pushed before any code runs; a dated amendment freezes the agent prompt, templates and model after development on the dev target only, before round 1. H1 is tested on the main-method stage, H2, H3, H4b, H7 and H8 on the controls stage, H4 on the baselines stage, and H5 and H6 on the ablations stage.

## Run matrix (each stage ends with a push to GitHub + Hugging Face)
Stage 0 comes first: tooling and smoke test of the local agent in the sandbox on the dev target (throughput, isolation checks, end-to-end) and the reproduction-check training, about 5 runs and 2–3 hours.

Then three rounds, each going through the same stages in the same order:

| Stage (in this order within every round) | Conditions | Round 1: N=1 | Round 2: +2 → N=3 | Round 3: +7 → N=10 |
|---|---|---|---|---|
| Main method | T0 × {E1, E2, E3, E4} = 4 | 4 runs | 8 | 28 |
| Controls (5.1) | T0 × {E0, W-freq, W-alg, W-role, W-scramble} = 5 | 5 | 10 | 35 |
| Baselines (5.2) | no agent | — | — | — |
| Ablations (5.3) | 2 models × 4 + 2 formats + self-test off = 11 | 11 | 22 | 77 |
| Hyperparameter sensitivity (5.4) | 2 temperatures × {E3, W-freq} = 4 | 4 | 8 | 28 |
| Seed sensitivity (5.5) | T1 × 4 + T2 × 4 + T-new × 3 + T-paper × 5 = 16 | 16 | 32 | 112 |
| **Total agent runs** | **40 conditions** | **40 (~13 h)** | **80 (~27 h)** | **280 (~90 h)** |

- Everything without an agent is done in full in round 1 and refreshed in later rounds where it depends on agent-built replacements: the no-agent controls, all baselines (including the planted-calibration and distilled-student training), and the grader and explainer sensitivity analyses. T-new is trained just before round 1's seed-sensitivity stage.
- Each round ends with every hypothesis evaluated at that N, a full regeneration of tables, figures, paper, README and site labelled with that N, a push to GitHub and Hugging Face, and a review pass. Round 1 fits the 14-hour first-run budget.
- In total 400 local agent runs, roughly 130 hours of GPU time at an assumed 20 minutes per run (the two smaller models should be faster). **Cost:** nothing is billed; every model runs locally.

Decision rules fixed in advance: if the local primary runs below 8 tok/s or fails tool calling in stage 0, qwen3.5:9b becomes the local primary. Rounds 1 and 2 always complete. If round 3 cannot finish within the week, it stops at a stage boundary in the order above, and the write-up states the N reached for each stage. The run queue is resumable (JSONL manifest) with a live dashboard; I check progress every 30 minutes during long stages.

## Downloads and outward-facing actions this plan performs
| What | Source | Size | Where |
|---|---|---|---|
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | [unsloth/Qwen3.6-35B-A3B-GGUF](https://hf.co/unsloth/Qwen3.6-35B-A3B-GGUF) | ~17.7 GB | `D:\AI\models` |
| Qwen3.5-9B / 4B GGUFs (only if llama-server cannot use the existing Ollama files) | unsloth GGUF repos | ~5.7 + ~3 GB | `D:\AI\models` |
| llama.cpp Windows CUDA build + runtime | [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) | < 1 GB | `D:\AI\llama.cpp` |
| ~18 runs' `checkpoints.safetensors` (targets, retrain baselines, no-mechanism controls) | [BurnyCoder/grokking-modular-addition-runs](https://hf.co/BurnyCoder/grokking-modular-addition-runs), pinned revision | ~6.5 GB (364 MB each) | `D:\AI\data` |
| `grokviz` package | [your GitHub repo](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer), pinned commit, MIT | small | project `.venv` |
| `full_run_data.pth` (pickle; fetched and converted by `grokviz` with sha256 check and `weights_only=True`) | [callummcdougall/grokking_full_run_data](https://huggingface.co/callummcdougall/grokking_full_run_data), pinned revision | 456 MB | `D:\AI\data` |
| uv (OpenCode 2.3.2 is already installed in WSL) | official installer, user-level in WSL | small | WSL home |
| Python deps (torch 2.14 CUDA 13.0 as in `grokviz`, numpy, scipy, matplotlib, pytest, …) | PyPI / PyTorch index via uv | ~5 GB | project `.venv` |

Outward-facing: create public GitHub repo `BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations`, push via feature branches and self-merged PRs (no AI attribution), enable GitHub Pages; create a public Hugging Face repo for the replacement weights, explanations, transcripts and the few models trained here, and one collection holding it together with your existing runs repo. The experiment itself contacts no outside service: every agent run is local and offline. No firewall, Windows security or `.wslconfig` changes; Ollama is quit while `llama-server` holds the GPU.

Secrets: `.env` is git-ignored before the first commit; this project does not read the Anthropic key in it or the Claude Code login in WSL. A test plus a pre-push check scan logs, transcripts and results for key and token patterns.

## Repository layout
```
src/vmi/  config.py · targets.py (pin, fetch, select runs via grokviz) · training.py (distillation + continued training) · pipeline.py (single wrapper: targets → explain → simulate → grade → analyse)
          explainer/ (analysis, ladder, render) · reference/compile.py
          simulator/ (sandbox, relay, opencode_runner, queue, audit) · grader/ (outputs, hidden, stitching, score)
          analysis/ (stats, tables, figures) · dashboard/ (static page reused as the GitHub Pages explorer)
sandbox_template/  TASK.md · ARCHITECTURE.md · forward.py · selftest.py · opencode.jsonc
scripts/ run_one.sh · isolation_check.sh   tests/ · results/ · logs/ · docs/ · site/
reports/  literature-review.md · state-of-the-art-review.md · preregistration.md · planned-runs.md · plan.md · progress.md
          summary.md · paper/ (LaTeX → paper.pdf) · experiments/ (experiments.md, learnings.md, per-experiment folders) · reviews/
```

## Order of work
1. `uv init` (Python 3.12, local venv, `.env`), git, public GitHub repo.
2. Literature review + state-of-the-art review → fact-check every citation → **preregistration + README pushed before any result exists** → `planned-runs.md`.
3. Implement test-first (target loading, explainer, reference compiler, grader, the OpenCode sandbox runner) → push → stage 0. Training is minutes, with a live dashboard and a push before and after each job; the "first run" budget goes to stage 0 and round 1.
4. Methodology red-team written to `reports/reviews/`, fixes, prompt-freeze amendment → then three rounds (N=1, N=3, N=10), each strictly in the order main method → controls → baselines → ablations → hyperparameter sensitivity → seed sensitivity, pushing code, results, paper, site and HF models after each stage; a review pass after each round.
5. One command regenerates all tables and figures from `results/`. Paper (authors Libor "Burny" Burian and [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en), in that order; dated; AI statement at the end), README with figures/tables/architecture diagram, docs, interactive GitHub Pages site, HF collection. The same two authors appear in the README, the site, the HF card and `CITATION.cff`. All of these also state the project's origin: it was developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic interpretability explanations, which the authors co-organized in their mechanistic interpretability community and its Discord server, [mechinterp.com](https://mechinterp.com/) — in the paper's introduction and acknowledgements, near the top of the README, and on the site's front page.
6. Final review pass and final checks; the README's to-do lists the postponed Claude Code work (section 6) and the other toy models as next steps.

## Verification
- Training checks: retraining T0's spec reproduces the published run; `training.py` in label mode reproduces a published run before it is used for distillation or continued training; T-new reaches 100% test accuracy.
- Reuse checks: the explainer's key frequencies equal the published `analysis/key_freqs.json` for every target; each loaded target reproduces its recorded test loss; the converted paper checkpoint passes `grokviz`'s own validation.
- Unit tests: `forward.py` (numpy) equals `grokviz`'s `GrokTransformer` (torch) on random weights and on every target; grader gives KL=0 and similarity=1 for M vs M, low scores for M vs random; CKA and Procrustes agree with netrep and Google's reference CKA on random matrices; similarity invariant under an invertible linear change of residual basis and under neuron permutation; reference compiler reaches 100% task accuracy for E2–E4 and W-freq; explanation files contain no raw weight arrays; the audit flags a planted training loop.
- Sandbox: `scripts/isolation_check.sh` passes inside the sandbox; an OpenCode run inside it completes a trivial file-write task.
- End-to-end: `uv run python -m vmi.pipeline --stage smoke` on the dev target, then I read the agent transcript, wire log and grader output by hand before freezing the protocol and scaling up.

## Main risks
- The local model may fail to build working weights even from E4 (floor effect) → self-test tool, reference compiler as the agent-free ceiling, model-size ablation; a high failure rate is reported as a negative result.
- OpenCode v2 offline behaviour and its config keys are documented but untested here, and the docs I read predate 2.3.2 → stage 0 validates them on the installed version (`opencode debug config`) before any reported run.
- The local models may know the published algorithm, and possibly the paper's frequencies → the unpublished targets T0–T2 and T-new, the E0/E1 no-mechanism arms and the contamination flag on T-paper measure and bound this.
- All simulators are local models from one family, so the conclusions hold for that family and harness only → stated as a limitation; the cross-family check is the postponed Claude work.
- E4 approaches "compressed weights" → handled by always reporting score against description length.
- Reported throughput (20–35 tok/s) comes from comparable hardware, not this laptop → measured in stage 0, with the fallback rules above.
- Round 3 alone is about 90 GPU-hours, so N=10 for every condition may not fit the week → rounds 1 and 2 give complete N=1 and N=3 results first, and round 3 stops at a stage boundary if time runs out, with the N reached reported per stage.
- With N=1 and N=3 the intervals are absent or wide → hypotheses are still evaluated at those rounds, but every verdict, table, figure and sentence carries its N and says it is not yet N=3 or N=10, so an early verdict is never presented as the final one.
