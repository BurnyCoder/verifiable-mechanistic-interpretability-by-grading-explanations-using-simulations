# Summary — grading mechanistic explanations by weights-blind simulation

*Libor "Burny" Burian, [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en) — 2026-11-21.
Developed in a mechanistic interpretability hackathon on automatically evaluating mechanistic interpretability
explanations, co-organized by the authors in the [mechinterp.com](https://mechinterp.com/) community. Every number
below is generated from `results/` by `vmi analyse`.*

## Question

Can an explanation of a neural network be graded automatically by giving it to an agent that has never seen the
network, letting the agent write the network's weights, and comparing the rebuilt network with the original? Is
that score a valid measure of explanation quality?

## Setup in five lines

1. Target: one-layer modular-addition transformers ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)); the main
   target T0 and the seed-sensitivity targets are unpublished, T-new was trained after the protocol freeze.
2. Explanations rendered by script at five levels of true detail (E0–E4) plus length-matched wrong ones (W-*).
3. Simulator: Claude Haiku 5.5 in Claude Code, offline, no path to the weights; local Qwen models as an ablation.
4. Scores: output KL (behaviour) and cross-validated hidden-state similarity HS (mechanism, linear alignment).
5. Calibration: an agent-free reference compiler, retrains, distilled students, random init, wrong targets, a planted
   mechanism, the training trajectory; hypotheses preregistered.

## Results so far

<!-- BEGIN key-results -->
*Round 1, N = 1 run(s) per condition — not yet N=3 or N=10*

- **Model-specific detail raises hidden-state similarity:** HS on T0 is 0.05 (E1, algorithm only), 0.90 (E2), 0.94 (E3) and 0.91 (E4); with no explanation (E0) it is 0.00. Not monotone: E4 < E3.
- **Correct behaviour, wrong mechanism is caught by HS, not by output KL:** the wrong-frequency replacement reaches KL 9.31e-04 nats (true E3: 0.023) but HS 0.06.
- **Copying behaviour does not copy the mechanism:** independent retrains (HS 0.12) and logit-distilled students (HS 0.23, KL 2.38e-06) stay at the random-initialisation floor (HS 0.24).
- **Negative result:** the primary HS does not separate the Pizza explanation (W-alg, HS 0.89) or the scrambled neuron table (W-scramble, HS 0.92) from the true explanations; the secondary matched-neuron correlation separates W-scramble (0.62) from E4 (0.89).
- **Memorisation (post hoc):** when the explanation gives no frequencies (E0, E1), the simulator's replacements on unpublished targets use the paper's published frequencies (mean overlap 0.80 with {14, 35, 41, 42, 52}, 0.00 with the target's own). On the published checkpoint the same algorithm-only explanation therefore scores HS 0.92 (on T0: 0.05): scores on published models measure memory, not the explanation.
- **Cost:** all Claude Haiku 5.5 runs so far cost $0.86 in total.
<!-- END key-results -->

## Hypotheses

<!-- BEGIN hypotheses -->
| Hypothesis | Verdict | Evidence size |
|---|---|---|
| H1 | not supported | at N=1 — not yet N=3 or N=10 |
| H2 | E3>W-freq: cannot tell at this N; E3>W-alg: cannot tell at this N; E3>W-role: cannot tell at this N; E4>W-scramble: not supported | at N=1 — not yet N=3 or N=10 |
| H3 | supported | at N=1 — not yet N=3 or N=10 |
| H4 | E3: cannot tell at this N; E4: cannot tell at this N | at N=1 — not yet N=3 or N=10 |
| H4b | not supported | deterministic |
| H5 | not evaluated yet | – |
| H6 | not evaluated yet | – |
| H7 | cannot tell at this N | 2 E3/E4 replacements (at N=1 — not yet N=3 or N=10) |
| H8 | supported | one replacement, all checkpoints |
<!-- END hypotheses -->

## What this means

- **Use hidden-state similarity, not output KL**, to grade explanations of this kind of model: behaviour can be copied
  without the mechanism, and KL rewards that.
- **HS works at the level of frequencies, not below it.** It cannot tell the Clock algorithm from the Pizza algorithm
  with the same frequencies, nor a correct neuron table from a scrambled one; a neuron-level or causal second metric
  is needed for finer explanations.
- **Grade on unpublished models.** The simulator fills gaps in an explanation with what it remembers from the
  literature, which makes scores on famous models meaningless.

## Status and next steps

Round 1 (N = 1 per condition) is complete for the Claude simulator; the local Qwen ablation runs are finishing.
Rounds 2 (N = 3) and 3 (N = 10) follow, then a second metric for sub-frequency structure. Details:
[paper](paper.pdf), [round-1 report](experiments/01-round1/experiment.md), [self-review](reviews/round1-review.md),
[results explorer](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/).
