---
title: "Grading mechanistic explanations by weights-blind simulation: a preregistered test on the modular-addition transformer"
author:
  - Libor "Burny" Burian
  - Victor Levoso
date: 2026-11-21
geometry: margin=2.2cm
fontsize: 10pt
mainfont: Cambria
mathfont: Cambria Math
monofont: Consolas
colorlinks: true
---

## Abstract

Interpretability agents can now produce mechanistic explanations faster than people can check them. We test one
automatic check: give the explanation, and nothing else, to an LLM agent that has never seen the network and has no
internet access, ask it to write every weight of a replacement network, and score the replacement against the
original by output KL divergence and by cross-validated hidden-state similarity (HS). The target is the one-layer
modular-addition transformer of Nanda et al. (2023). A script with weights access renders explanations at five levels
of true detail plus length-matched wrong explanations; Claude Haiku 5.5 in Claude Code, inside an offline sandbox,
builds the replacements; an agent-free reference compiler, independent retrains, logit-distilled students, random
initialisations, wrong targets and a planted-mechanism model calibrate the score. Hypotheses, metrics and analyses were
preregistered before any result. First results (round 1, one run per condition — not yet three or ten):

<!-- BEGIN key-results -->
<!-- END key-results -->

Output KL alone ranks a wrong-frequency explanation, independent retrains and distilled students as good as or better
than true explanations; HS does not. HS is blind to the per-neuron frequency assignment and to the Clock-versus-Pizza
distinction, both of which keep the key frequencies.

## 1 Introduction

Mechanistic interpretability explains a network by naming the algorithm its weights implement. The evidence for such
an explanation is usually a ladder: correlation, then steering, ablation and counterfactual swapping, and at the top
*simulation* — rebuilding the mechanism well enough to reproduce the model. As agents begin to generate explanations
at scale, grading them becomes the bottleneck, and the top rung is the one hardest to automate.

Automated scores that exist simulate single features' activations with an LLM and correlate them with the real ones
([Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html);
[Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). They are known to disagree with causal tests
([Huang et al. 2023](https://arxiv.org/abs/2309.10312)), to be bypassable by a simulator that ignores the explanation
([Poché et al. 2026](https://arxiv.org/abs/2609.08585)), and not to separate trained from random transformers
([Heap et al. 2025](https://arxiv.org/abs/2501.17727)). [Gao (2026)](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)
proposes substituting simulated activations and scoring downstream KL. We go one step further and let the simulator
write the *whole network's weights*, then ask the question that has to come first: **is the resulting score a valid
measure of explanation quality?** Concretely: does it rise with true detail, fall for wrong explanations, beat what
the training recipe or behavioural cloning give for free, and catch a wrong mechanism that still behaves correctly?

**Contributions.** (i) A weights-blind, offline simulation grader for mechanistic explanations of an SGD-trained
transformer, with an output metric and a symmetry-aware hidden-state metric. (ii) A preregistered validation with a
scripted ladder of true explanations, length-matched wrong explanations, explanation-free baselines, wrong-target,
planted-mechanism and training-trajectory controls, and an agent-free reference compiler that separates what an
explanation contains from what the agent makes of it. (iii) Evidence that output KL is not a valid score on this task
and that HS is, for frequency-level content, with two documented blind spots.

This project was developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic
interpretability explanations, which the authors co-organized in their mechanistic interpretability community and
its Discord server, [mechinterp.com](https://mechinterp.com/).

## 2 Background and related work

**The target.** [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) reverse-engineered a one-layer transformer
(p = 113, d_model 128, four heads, 512 ReLU neurons, no LayerNorm) trained on (a + b) mod 113: it embeds a and b as
cos/sin(w_k a), cos/sin(w_k b) for a few key frequencies w_k = 2πk/113, forms cos/sin(w_k(a + b)) with attention and
the MLP, and reads out logit(c) ≈ Σ_k α_k cos(w_k(a + b − c)). The published explanation leaves per-neuron phases and
gains, the allocation of neurons to frequencies and the logit scale unspecified. [Zhong et al. (2023)](https://arxiv.org/abs/2306.17844)
showed that a second algorithm ("Pizza") solves the same task with the same outputs — an existence proof that
behaviour does not determine mechanism here. [Yip et al. (2024)](https://arxiv.org/abs/2412.03773) and
[Gromov (2023)](https://arxiv.org/abs/2301.02679) give closed-form constructions for related networks.

**Explanations checked against weights.** Tracr ([Lindner et al. 2023](https://arxiv.org/abs/2301.05062)) compiles
programs to weights; causal abstraction ([Geiger et al.](https://arxiv.org/abs/2301.04709)) and causal scrubbing test
faithfulness by interventions, and [Sutter et al. (2025)](https://arxiv.org/abs/2507.08802) show that unrestricted
non-linear alignment maps make any network match any algorithm, so the alignment class must be justified.
[Cammarata et al. (2021)](https://distill.pub/2020/circuits/curve-circuits/) hand-built one curve-detector circuit and
compared its activations with the natural one — our idea, done by hand for one circuit. Compact proofs
([Gross et al. 2024](https://arxiv.org/abs/2406.11779)) make explanation length explicit, with the weights themselves
as the longest exact explanation.

**Agents that write weights.** The closest work, iWSP ([Mehta et al. 2026](https://arxiv.org/abs/2611.18114)), has
Claude Code write small transformers' weights and rebuild them from a short description, but only for agent-written
models and without output-KL or hidden-state comparison. Interpretability agents can reproduce published circuits from
memory ([Haklay et al. 2026](https://arxiv.org/abs/2603.20101)), which is why our targets are unpublished models.

**What is new here** is the combination: an SGD-trained target, a weights-blind offline agent that compiles a
mechanistic explanation into a full replacement, output KL plus a symmetry-aware hidden-state metric, explanation-length
accounting, and no-explanation, wrong-explanation, wrong-target and planted-mechanism controls. Each ingredient has
precedent (full review: `reports/literature-review.md`).

## 3 Method

![Pipeline. A script with weights access writes the explanation; an LLM agent inside an offline sandbox, with no path to the target's weights, writes a replacement; the grader compares replacement and target. The reference compiler turns the explanation's parameters into weights with no agent.](../figures/architecture.png)

### 3.1 Targets

All targets share the architecture and training recipe of Nanda et al. (full-batch AdamW, lr 10⁻³, weight decay 1,
30 % training split, 40,000 steps) and reach 100 % test accuracy. T0 is the main target; T1 and T2 have the fewest and
most key frequencies among the published seeds; T-new was trained after the protocol freeze, so no development choice
could be tuned to it; T-paper is the paper's own checkpoint (contamination-flagged: its explanation is published); the
dev target was used for all development and is never reported. Targets come from
[BurnyCoder/grokking-modular-addition-runs](https://huggingface.co/BurnyCoder/grokking-modular-addition-runs).

<!-- BEGIN targets -->
<!-- END targets -->

### 3.2 Explanations

A script reads each target's weights, runs the paper's Fourier analysis (via the paper-faithful
[grokviz](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer) package) and renders:
**E0** the task and architecture only; **E1** the Fourier-multiplication algorithm without model-specific numbers;
**E2** the key frequencies and each component's role; **E3** adds per-frequency amplitudes, neurons per frequency,
attention-pattern fits, logit scale and the precise neuron function; **E4** adds a per-neuron table (frequency, phase,
gain). Wrong controls: **W-freq** is E3 with the key frequencies swapped for other valid ones (a working adder with a
different mechanism), **W-alg** describes the Pizza algorithm at E3 length, **W-role** misattributes component roles,
**W-scramble** reassigns each E4 neuron's frequency at random among the key frequencies. **E-paper** paraphrases the
published explanation with the paper's numbers (T-paper only). No number in an explanation is typed by hand; every
explanation's length is recorded.

### 3.3 Simulator and sandbox

The simulator is Claude Haiku 5.5 in Claude Code 2.1.330 (`--bare`, permissions bypassed inside the sandbox, web
tools disallowed, default sampling, effort `medium`). It runs in a bubblewrap sandbox in WSL2 in its own empty network
namespace (no interface but loopback), with a cleared environment, no Windows drives or home directory, and AF_VSOCK blocked by seccomp; it
reaches the model only through a unix socket to a proxy outside the sandbox that injects the API key, allows only the
Messages endpoint and the run's models, rejects server-side tools with internet access, and enforces per-run and total
budgets. The sandbox holds `TASK.md`, `EXPLANATION.md`, `ARCHITECTURE.md`, a numpy `forward.py`, a `selftest.py`
that checks shapes and accuracy against the task labels (never against M), and a starter `build_weights.py`. Training
loops and iterative optimisation are forbidden; build scripts are audited and flagged ones reviewed by hand. Runs are
capped at 60 minutes. Local Qwen models (Qwen3.6-35B-A3B, Qwen3.5-9B, Qwen3.5-4B) in a bash-only harness modelled on
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) form the simulator-capability ablation: in development
none of them exceeded 5 % task accuracy, which made Haiku the main simulator by a rule fixed before any Claude run
(preregistration amendment 2).

### 3.4 Grader

**Output KL:** mean KL(P_M ‖ P_M̂) over all 12,769 inputs and 113 classes, in nats, float64. **Hidden-state
similarity (HS):** at three final-position sites (residual stream after attention, MLP activations, final residual
stream), 5-fold cross-validated ridge-regression R² in both directions (M̂ → M and M → M̂), combined by harmonic mean
per site and averaged over sites. Linear maps are the alignment class because, without LayerNorm, the residual stream
is defined only up to an invertible linear map; anything richer makes every network match every algorithm
([Sutter et al. 2025](https://arxiv.org/abs/2507.08802)). Secondary metrics include Hungarian-matched neuron
correlation, centred-logit FVU and temperature-calibrated KL. A run without valid weights is scored as a failure.

### 3.5 Controls, baselines and statistics

Controls: E0 and the W-* explanations through the agent; random initialisations; every T0 replacement graded against
the other targets and against models trained without a clean mechanism (H7); the reference-compiled E3 model trained
further and graded against its own snapshots (planted mechanism, H4b); T0 replacements graded against T0's training
checkpoints next to the paper's restricted loss (H8). Baselines: independent retrains (other seeds of the same
recipe), three students distilled on M's full logits, the reference compiler for every explanation, M with only its
key-frequency Fourier components kept in the embedding and unembedding, and the weights themselves. Every condition
runs in three rounds (N = 1, 3, 10 seeds); every hypothesis is evaluated after every round with Jonckheere–Terpstra
and Spearman trend tests (H1), one-sided Mann–Whitney tests with Holm correction (H2, H4), Kendall's W (H6) and
Wilcoxon signed-rank tests (H7). A test whose smallest attainable p-value at the current N is ≥ 0.05 returns
"cannot tell at this N". The preregistration, two dated amendments and every post-freeze deviation are in
`reports/preregistration.md`.

## 4 Results (round 1, N = 1 per condition)

### 4.1 Explanation ladder and controls on T0

<!-- BEGIN scoreboard-T0 -->
<!-- END scoreboard-T0 -->

![Scores per explanation on T0. Dots: Claude Haiku 5.5 runs; diamonds: reference compiler (no agent); dashed lines: means of no-agent reference models.](../figures/ladder_T0.png)

E0 and E1 produce working adders (task accuracy near 100 %) whose hidden states share almost nothing with T0: the
simulator's priors supply *an* algorithm, not *this model's* mechanism. Naming the key frequencies (E2) lifts HS close
to the reference compiler's value, and E3 and E4 stay close to it; E4 is below E3 in this single run. Wrong frequencies
and wrong component roles give working adders with low HS. The Pizza description and the scrambled neuron table score
like true explanations on HS; the matched-neuron correlation separates the scrambled table from E4.

### 4.2 Behaviour is not mechanism

![Output KL against HS on T0 for agent runs and no-agent reference models.](../figures/kl_vs_hs_T0.png)

<!-- BEGIN baselines-T0 -->
<!-- END baselines-T0 -->

Independent retrains and logit-distilled students reproduce T0's outputs more closely than any explanation-built
replacement, yet their HS is at or below the random-initialisation floor. The W-freq replacement likewise has small
output KL and low HS. Output KL alone would rank all of these at or above the true explanations; HS ranks them at the
bottom. Keeping only T0's key-frequency components in its embedding and unembedding leaves HS almost unchanged: the
key frequencies carry nearly all of what HS measures.

### 4.3 Calibration, specificity and training dynamics

<!-- BEGIN controls-T0 -->
<!-- END controls-T0 -->

![Planted-mechanism calibration.](../figures/planted_T0.png)

![HS of the best T0 replacement against T0's training checkpoints, above the checkpoints' restricted loss.](../figures/trajectory_T0.png)

On the planted mechanism, HS is 1 at step 0 and falls monotonically as training moves the model away from the
planted construction; output KL first rises and then falls again as training re-sharpens the outputs, so KL is not a
calibrated distance here. Replacements built from T0's explanations fit T0 much better than any other target. Against
T0's checkpoints, HS rises during circuit formation in step with the fall of the restricted loss.

### 4.4 Explanation length

![HS against explanation length (characters) on T0.](../figures/length_T0.png)

### 4.5 Ablations, sensitivity and other targets

<!-- BEGIN runs -->
<!-- END runs -->

### 4.6 Hypotheses at N = 1

<!-- BEGIN hypotheses -->
<!-- END hypotheses -->

## 5 Discussion

**What the score measures.** On this task HS measures whether the replacement uses the target's key frequencies in the
same places. That is the level at which the paper's explanation is stated, and HS separates it sharply from correct
behaviour obtained any other way. It does not measure the per-neuron assignment (W-scramble) or the non-linearity that
distinguishes Clock from Pizza (W-alg): both preserve every frequency subspace, and a linear map over a whole site
cannot see what is inside them. A complete grader needs a second, neuron-level or causal metric; the matched-neuron
correlation is a candidate; a stitching test (map M̂'s hidden state into M and run the rest of M) is implemented but
returned no value in these runs and is not yet validated.

**Why output KL fails.** A grokked model's outputs are nearly one-hot, so output KL mostly measures the replacement's
logit scale and its few wrong answers. Many different mechanisms reach the same near-zero KL — the training recipe
alone does — while a faithful replacement with a slightly wrong logit scale gets a larger one.

**The agent versus the compiler.** Haiku's replacements reach HS close to the reference compiler's for E2–E4: the
explanation, not the agent, is the limiting factor at this level of detail. The local Qwen models (ablation) are
reported in Section 4.5 as they finish.

## 6 Limitations

- **Sample size.** Round 1 has one run per condition; most hypothesis tests cannot reach significance at N = 1, and
  those verdicts are reported as "cannot tell at this N". Rounds 2 (N = 3) and 3 (N = 10) follow.
- **One task, one architecture.** All targets are one-layer modular-addition transformers. Whether HS stays informative
  for deeper networks, where linear maps across layers are less constrained, is untested.
- **One simulator family in the main arm.** Claude Haiku 5.5 may know the paper; T-paper is flagged, E0/E1 measure what
  the simulator brings without model-specific information, and T0–T2 and T-new are unpublished.
- **Scripted explanations.** The explanations are rendered by our script, so they share its style and its errors; the
  compiler's non-monotonicity on T2 and T-paper (Appendix) shows that E3 as rendered can mislead a literal build.
- **HS blind spots.** Per-neuron assignment and Clock-versus-Pizza (Section 5).

## 7 Conclusion

Giving a weights-blind agent an explanation and grading the network it writes is feasible today at a cost of cents per
run. On the modular-addition transformer, output KL is not a valid score for explanations, while hidden-state
similarity is valid for frequency-level content and blind below it. The method's main requirement is a hidden-state
metric whose invariances match the explanation's level of detail.

## Acknowledgements

We thank the participants of the mechanistic interpretability hackathon on automatically evaluating explanations held
in the [mechinterp.com](https://mechinterp.com/) community, where this project started.

## References

- Bills et al. 2023. Language models can explain neurons in language models. [link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- Cammarata et al. 2021. Curve Circuits. Distill. [link](https://distill.pub/2020/circuits/curve-circuits/)
- Gao 2026. Neural networks are not already clean. LessWrong. [link](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)
- Geiger et al. Causal abstraction: a theoretical foundation for mechanistic interpretability. [arXiv:2301.04709](https://arxiv.org/abs/2301.04709)
- Gromov 2023. Grokking modular arithmetic. [arXiv:2301.02679](https://arxiv.org/abs/2301.02679)
- Gross et al. 2024. Compact proofs of model performance via mechanistic interpretability. [arXiv:2406.11779](https://arxiv.org/abs/2406.11779)
- Haklay et al. 2026. Pitfalls in evaluating interpretability agents. [arXiv:2603.20101](https://arxiv.org/abs/2603.20101)
- Heap et al. 2025. Automated interpretability metrics do not distinguish trained and random transformers. [arXiv:2501.17727](https://arxiv.org/abs/2501.17727)
- Huang et al. 2023. Rigorously assessing natural language explanations of neurons. [arXiv:2309.10312](https://arxiv.org/abs/2309.10312)
- Lindner et al. 2023. Tracr: compiled transformers as a laboratory for interpretability. [arXiv:2301.05062](https://arxiv.org/abs/2301.05062)
- Mehta et al. 2026. Interpretable weight-space programming builds transparent transformers and brain encoding models. [arXiv:2611.18114](https://arxiv.org/abs/2611.18114)
- Nanda et al. 2023. Progress measures for grokking via mechanistic interpretability. ICLR. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Paulo et al. 2024. Automatically interpreting millions of features in large language models. [arXiv:2410.13928](https://arxiv.org/abs/2410.13928)
- Poché et al. 2026. Limitations of automated simulatability: LLM simulators can bypass explanations. [arXiv:2609.08585](https://arxiv.org/abs/2609.08585)
- Sutter et al. 2025. The non-linear representation dilemma: is causal abstraction enough for mechanistic interpretability? [arXiv:2507.08802](https://arxiv.org/abs/2507.08802)
- Yip et al. 2024. Modular addition without black-boxes: compressing explanations of MLPs that compute numerical integration. [arXiv:2412.03773](https://arxiv.org/abs/2412.03773)
- Zhong et al. 2023. The Clock and the Pizza: two stories in mechanistic explanation of neural networks. [arXiv:2306.17844](https://arxiv.org/abs/2306.17844)

## Appendix A — Reference compiler on every target

T0:

<!-- BEGIN compiler-T0 -->
<!-- END compiler-T0 -->

T1:

<!-- BEGIN compiler-T1 -->
<!-- END compiler-T1 -->

T2:

<!-- BEGIN compiler-T2 -->
<!-- END compiler-T2 -->

T-new:

<!-- BEGIN compiler-T-new -->
<!-- END compiler-T-new -->

T-paper:

<!-- BEGIN compiler-T-paper -->
<!-- END compiler-T-paper -->

## Appendix B — Claude API cost

<!-- BEGIN cost -->
<!-- END cost -->

## AI statement

Claude Code (Anthropic) wrote this codebase and this paper; human cleanup and rewriting are in progress. A human
provided the exact specification that Claude Code executed.
