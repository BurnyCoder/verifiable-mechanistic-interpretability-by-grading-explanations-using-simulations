# Next steps — to-do for the next iteration

*Added 2026-11-23 from your list of ideas. **Nothing here is scheduled:** like round 3 (N = 10), each item waits until
you ask for it. Every metric and analysis below is post hoc for rounds 1–2: on existing results it is reported as
post hoc, and before it is used as a test on new runs it needs a dated preregistration amendment
([preregistration.md](preregistration.md)). Simulator runs follow this study's Claude budget rule (API credits,
cheapest model first, Haiku → Sonnet → Opus, within the agreed cap; [planned-runs.md](planned-runs.md)). The README's
[To do](../README.md#to-do) has one line per group; this file holds each item's design. The mapping from your list to
the entries is at the end.*

**Cost classes:** *existing results* (no new agent runs, no API spend) · *agent runs* (API spend) · *new targets*
(training, downloads, GPU time).

**Suggested order:** the existing-results items first (A1, A2, B, D1, D3, the diff part of D4), then the agent-run
items (A3, C, D2, E, G1, H), then the new targets (F, G2, G3).

## A. Existing results, examined further

**A1. More negative results, highlighted.** *existing results*
- Re-read `results/` (runs, controls, baselines, audits) for failures not yet among the README's ❌ highlights or the
  paper's results: orderings that contradict the known ground truth, "not supported" verdicts, per-target exceptions.
- Add each one as a generated row in `highlight_rows` (`src/vmi/analysis/predictions.py`), so its numbers come from
  `results/` like the existing rows.

**A2. Check the built networks directly, in more ways.** *existing results*
- For every replacement in `results/`, read its build script and compute and plot the following:
  - the Fourier spectrum of W_E (`embedding_spectrum_cosine`, `src/vmi/grader/metrics.py`);
  - neuron-frequency clusters at `mlp_post`;
  - attention patterns (`matched_attention_distance`);
  - direct logit attribution per frequency;
  - the effect of ablating each key frequency (restricted and excluded loss, `grokviz.metrics.restricted_loss`);
  - Zhong et al.'s Clock/Pizza diagnostics (`src/vmi/analysis/clock_pizza.py`).
- Compare each with that build's HS and KL, and list where they disagree.
- Why: HS is one linear summary per site. It did not separate Clock from Pizza, or a true neuron table from a
  shuffled one (README highlights).

**A3. An agent as similarity judge.** *agent runs*
- A sandboxed agent gets the target and a replacement, with weights access, and one question: do these implement the
  same mechanism, and how similar are they? It chooses its own methods (reading weights, running code, plotting) and
  returns a verdict, a score and its evidence.
- Score its verdicts on cases whose answer is known:
  - true-explanation builds;
  - W-freq, W-alg and W-scramble;
  - independent retrains;
  - logit-distilled students;
  - the planted model's snapshots.

  Compare its verdicts with HS and KL. File names are anonymised, so the agent never sees a condition label.
- Related work: MAIA, an agent that runs experiments on other models' components
  ([Rott Shaham et al. 2024](https://arxiv.org/abs/2404.14394)).
- Limitation: this judge sees weights, so it grades the similarity of two networks, not an explanation.

## B. Grader metrics

**B1. Rotation-aware hidden-state similarity.** *existing results*
- The problem: HS fits a ridge map, so any invertible linear transform of a correct representation scores 1. A metric
  without a fit (raw cosine or R²) can score the same rotated copy near 0. A correct but rotated build should land in
  between.
- The proposal is an invariance ladder at each site, from strict to loose:
  1. identity: no fit (R² or cosine of the raw activations);
  2. permutation + positive scaling: the README's group-matched `mlp_post` item;
  3. orthogonal: `procrustes_distance`;
  4. full linear: the current `cv_r2`.

  Report every rung and their mean. A rotated copy then scores high on rungs 3–4 and low on rungs 1–2.
- Caveat: the no-LayerNorm residual stream has no privileged basis, because any invertible map with compensating
  weights computes the same function. There, rung 1 measures whether the explanation *specified* a basis, not
  whether the mechanism is right. `mlp_post`, after the ReLU, does have a privileged neuron basis.
- Test: compare explanations that state a basis with ones that do not. The prediction is that stating a basis raises
  only the strict rungs.
- Code: `cv_r2`, `procrustes_distance` and `matched_neuron_corr` in `src/vmi/grader/metrics.py`.

**B2. Weight-space similarity.** *existing results*
- Cosine similarity and mean absolute difference between the two networks' weight matrices (embedding, positional
  embedding, attention, MLP, unembedding). Report each raw and after aligning the network's symmetries:
  - MLP neurons and attention heads by optimal one-to-one matching (scipy `linear_sum_assignment`, already used by
    `matched_neuron_corr`), the permutation alignment that Git Re-Basin uses before merging models
    ([Ainsworth et al. 2022](https://arxiv.org/abs/2209.04836));
  - the residual basis by orthogonal Procrustes.
- Why the alignment matters: a permuted copy of the same network has low raw cosine.
- This sits next to the README's existing item on a weights-only frequency statistic (the embedding-spectrum cosine).

**B3. Noise in the target.** *existing results*
- The problem: a trained network carries content that no explanation describes (components outside the key
  frequencies), so part of every score rewards reproducing noise.
- What exists: the Fourier-restricted baseline (`fourier_restricted`, `src/vmi/nonagent.py`) projects only W_E and W_U
  onto the key frequencies (README baselines table).
- The proposal:
  1. Project the hidden activations as well, as functions of (a, b), onto the key frequencies' 2D Fourier components.
     This is the construction behind Nanda et al.'s restricted loss, applied to hidden states instead of logits.
  2. Grade every replacement against the projected target as well as the raw one.
  3. Report a noise ceiling: HS of T0 against its projection, and against its own checkpoints after cleanup. Scores
     are then read against what is attainable, not against 1.
- Links to the README item on rendering the embedding's own frequency set separately from the read-out's.

## C. Simulation granularity

**C1. Layer by layer instead of the whole network.** *agent runs*
- The agent builds one component at a time (embedding, attention, MLP, unembedding) from that component's
  explanation. Each component is graded with the target's true inputs (teacher forcing), so one component's error
  does not carry into the next. The output is per-component scores plus their aggregate.
- Code to reuse: `stitching_kl` (`src/vmi/grader/metrics.py`) already splices a replacement's residual stream into the
  target, and `site_score` grades any site.
- Why: whole-network rebuilds do not scale beyond toy models (G3), and per-component scores show where an
  explanation fails.

**C2. Activations instead of weights.** *agent runs*
- The simulator predicts a site's activations directly from the explanation, for all 113² inputs or a sample, without
  building weights. This applies the per-neuron simulation of
  [Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) to whole sites.
  Grade with `site_score` and the B1 ladder.

**C3. Predict the next layer from the previous one.** *agent runs*
- Given the previous layer's true activations and an explanation of the layer being replaced, an LLM or an agent
  predicts the next layer's activations.
  - Optional extra inputs: the input tokens, and earlier layers' activations.
  - Variants: predict that layer's weights instead, or predict single neurons.
- Aggregate the per-layer scores into one score, optionally together with the whole-model rebuild (E1).
- How C1 and C3 differ: C1 produces weights from an explanation; C3 produces activations from activations plus an
  explanation.

## D. Explanation quality beyond fidelity

**D1. Weight scores by explanation length.** *existing results*
- The paper plots HS against explanation length (§4.4). Turn that view into a length-aware score:
  - the Pareto front of (score, compressed length in bits);
  - a penalised score, score − λ·bits, reported over a range of λ rather than at one value chosen after the fact.
- Why: a dump of the weights scores high (README baselines: weights dump) but explains nothing, and a
  description-length view separates the two. For the description-length view see MDL for SAEs
  ([Ayonrinde et al.](https://arxiv.org/abs/2410.11179)) and the proof-length trade-off of compact proofs
  ([Gross et al. 2024](https://arxiv.org/abs/2406.11779)).

**D2. LLM as judge of simplicity and understandability.** *agent runs*
- Each explanation gets two rubric ratings:
  - simplicity, in the Occam sense: fewer assumptions, fewer free parameters, lower complexity;
  - how understandable it is to a human.
- Calibrate against a small human-rated subset, and randomise order and anonymise conditions. LLM judges show
  position, verbosity and self-enhancement biases ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)). Verbosity
  bias matters here because the explanation levels differ in length.

**D3. Structural complexity of an explanation.** *existing results*
- Three measures:
  - node count and depth of the abstract syntax tree of code explanations (Python's
    [`ast`](https://docs.python.org/3/library/ast.html));
  - operation count and expression-tree size of math explanations
    ([SymPy `count_ops`](https://docs.sympy.org/latest/modules/core.html));
  - syntax-tree size of a [Lean 4](https://lean-lang.org/) formalisation of the explanation.
- Inputs: the E3 code and math formats already exist (`results/explanations/*/E3-fmt-code`, `E3-fmt-math`). A Lean
  version still has to be written.

**D4. Generalisation of an explanation.** *existing results (diffs, compiler); agent runs (transfer test)*
- A good explanation of a family of networks changes little when the network changes a little. Across T0, T1, T2 and
  T-new, which have different key frequencies, ideally only the frequency list changes.
- Measure:
  1. how much of each level's explanation differs between two targets (diff size in bits), set against how much
     their weights differ. Explanations are rendered per target by `src/vmi/explainer/render.py`, so these diffs come
     from `results/explanations/` with no new runs;
  2. a transfer test: T0's E3 with only its frequency list replaced by T1's, graded against T1 next to T1's own E3.
     Run it with the agent-free reference compiler first, then with the agent.

## E. The ladder of evidence and training time

**E1. One score across the ladder of evidence.** *agent runs*
- An explanation makes predictions on several rungs. Grade each rung, and report the vector of grades and an
  aggregate:
  - correlational: does it predict activations (C2);
  - interventional:
    - ablations: does removing what it names break the behaviour it predicts;
    - steering: does adding a direction it names change outputs as predicted;
    - counterfactual concept swapping: interchange interventions
      ([Geiger et al.](https://arxiv.org/abs/2301.04709)) and causal scrubbing
      ([Chan et al. 2022](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing-interpretability-hypotheses));
  - simulation, at increasing granularity: one neuron, one attention head, the MLP layer, the residual stream, one
    layer's activations or weights (C1–C3), a few layers, the whole model (the current method).
- The aggregation weights are a design choice to preregister. The vector is always reported, so no weighting can hide
  a failing rung.

**E2. Mechanistic interpretability in time.** *agent runs*
- Explain the training dynamics, not only the end point: what the network computes during memorisation, circuit
  formation and cleanup ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)).
- The simulator rebuilds the network at several checkpoints from a time-indexed explanation, and each rebuild is
  graded against the matching checkpoint.
- Code to reuse: the H8 code (`trajectory`, `src/vmi/nonagent.py`) already grades replacements against T0's saved
  checkpoints.
- Related work: SAE features traced through grokking in a 19M-parameter transformer trained on 3-digit addition
  ([Wang et al. 2026](https://aclanthology.org/2026.aacl-long.113.pdf)).

## F. Explanation units

**F1. Explanations in other units.** *new targets, agent runs*
- So far, explanations use the network's own terms (frequencies, neurons). Render and grade explanations in other
  units:
  - neurons (E4 already does this);
  - steering vectors ([Turner et al. 2023](https://arxiv.org/abs/2308.10248));
  - SAE features (F2);
  - feature manifolds ([Engels et al. 2024](https://arxiv.org/abs/2405.14860);
    [Gurnee et al. 2026](https://arxiv.org/abs/2601.04480)). Here, each key frequency maps a token a to the point
    (cos ωa, sin ωa) on a circle, a one-dimensional manifold;
  - crosscoder features ([Lindsey et al. 2024](https://transformer-circuits.pub/2024/crosscoders/index.html));
  - attribution-graph circuits built from cross-layer transcoders
    ([Ameisen et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html));
  - block-sparse featurizers, which apply sparsity to multi-dimensional blocks so a concept can be a manifold
    ([Fel et al. 2026](https://arxiv.org/abs/2606.25234), shown on vision models).
- Question: which unit gives the best score per explanation length (D1)?

**F2. SAEs on the grokking transformer.** *new targets (training takes minutes)*
- Train sparse autoencoders on T0's residual stream and on `mlp_post`, for example with
  [SAELens](https://github.com/jbloomAus/SAELens).
- Check whether the latents recover the key-frequency Fourier features, and whether each frequency's circle is
  captured by a compact group of latents or tiled by many local ones.
  [Bhalla et al. 2026](https://arxiv.org/abs/2604.28119) report that real SAEs mix the two.
- Compare with a block-sparse featurizer (F1). Then render SAE-based explanations into the ladder and grade them.
- Prior work: a quick search on 2026-11-23 found no SAE study of this model. The nearest are the Local Interaction
  Basis, evaluated on a modular-addition model ([Bushnaq et al. 2024](https://arxiv.org/abs/2405.10928)), and the
  3-digit-addition study in E2.

## G. Simulators and targets

**G1. Claude Sonnet and Opus where Haiku failed.** *agent runs*
- Claude Haiku 5.5 failed on three points the grader depends on (README highlights):
  - the true E1–E3 explanations of the Clock target T0 were rebuilt Pizza-like (E4's rebuilds sit near the 0.4 line on the Clock side; paper §5);
  - the shuffled neuron table (W-scramble) was not followed literally;
  - explanations that name no frequencies (E0, E1) were filled in from memory of the published ones.
- Re-run exactly those conditions with Claude Sonnet 5.5, then Claude Opus 5.5, cheapest first, within the agreed
  budget. This item is already postponed in [planned-runs.md](planned-runs.md).
- Prediction to preregister: a more capable simulator follows explanations more literally (Clock-like rebuilds from
  Clock explanations, lower HS for W-scramble), but it may also fill gaps from memory more often.

**G2. Other toy models.** *new targets*
- The README's existing item: Clock/Pizza p = 59, the paper's 2-layer and LayerNorm variants, and ARENA's monthly
  problems ([plan.md](plan.md), "Later targets"). Purpose: test whether the conclusions hold on toy models with other
  mechanisms.

**G3. LLMs from millions to trillions of parameters.** *new targets*
- As targets, a size ladder:
  - GPT-2 small, with its indirect-object-identification circuit
    ([Wang et al. 2022](https://arxiv.org/abs/2211.00593));
  - the Pythia suite, 70M to 12B parameters, with training checkpoints
    ([Biderman et al. 2023](https://arxiv.org/abs/2304.01373)), which also serves E2;
  - Gemma 2 with the open Gemma Scope SAEs ([Lieberum et al. 2024](https://arxiv.org/abs/2408.05147)), which also
    serves F1;
  - open-weight models of about one trillion parameters, for example Kimi K2: 1T total and 32B activated parameters
    ([Kimi Team 2025](https://arxiv.org/abs/2507.20534)).

  Whole-network rebuilds stop being feasible early in this ladder. These targets need the component-level simulation
  of C and E1: a circuit, a layer, a head, a neuron.
- As simulators, the same ladder on the simulator axis: from small local models, which stayed at chance here
  (deviation 5), up to the largest available. This measures how the score depends on simulator capability, which is
  H5, not testable in rounds 1–2.

## H. Red teaming

**H1. Bad explanations that score high.** *agent runs*
- Search adversarially for wrong or useless explanations that score high, to map where the grader breaks:
  - compressed weight dumps posing as explanations (D1);
  - wrong mechanisms that share the right frequencies (W-alg);
  - shuffled details that the simulator repairs (W-scramble);
  - text that cues the simulator's memory of the published mechanism.
- Method: an attacker LLM proposes explanations, the grader scores them, and the loop keeps the high scorers. An
  independent check (the reference compiler and the A2 diagnostics) confirms that each kept explanation is wrong.
- Output: a catalogue of failure cases with their scores, added to the ❌ highlights (A1).

## I. Protocol fixes found by audit 8 (for a next protocol version; the current one stays frozen)

**I1. A clean self-test ablation.** *agent runs*
- The no-self-test variant removed `selftest.py` from `TASK.md` and the prompt, but the starter `build_weights.py`
  still says "re-run this script and selftest.py after each stage", and every no-self-test agent read it
  (`reports/reviews/audit-8-2026-11-23.md`). Strip that line in the variant, and add a test that no file in the
  variant's sandbox names the self-test.

**I2. A same-frequency specificity control.** *no agent*
- H7's wrong targets all use other key frequencies than T0, so it cannot show whether HS tells apart networks with the
  same frequencies. Add wrong targets that share T0's key frequencies: retrains selected for T0's frequency set, the
  Fourier-restricted T0 with permuted or re-phased neurons, and a literal Pizza build at T0's frequencies (already in
  `results/compiler/T0/W-alg.json`).

## Mapping from your list (2026-11-23)

| Your item | Entry |
|---|---|
| Try other toy models | G2 |
| Try LLMs of different sizes, from millions to trillions | G3 |
| Red-team for bad explanations that score high | H1 |
| Linear-regression HS ignores rotation; cosine would give 0, a rotated copy should get about 0.5 | B1 |
| Find more negative results and highlight them | A1 |
| See if Opus can do what Haiku couldn't | G1 |
| Evaluate the created models yourself in more ways | A2 |
| Generate layer by layer, not the full network | C1 |
| Generate activations instead of weights | C2 |
| Weight explanations by length | D1 |
| Mean-difference similarity between the two models' weights | B2 |
| Cosine similarity between the two models' weights | B2 |
| Account for noise in the modular-addition transformer | B3 |
| LLM as judge of simplicity (Occam's razor) | D2 |
| LLM as judge of human understandability | D2 |
| Length and complexity of the code AST, math-symbol tree and Lean tree | D3 |
| Generalisation of an explanation (only the frequency list should change) | D4 |
| An agent that judges whether the two models are the same | A3 |
| Mechanistic interpretability in time | E2 |
| One score across the ladder of evidence | E1 |
| Predict the next layer from the previous layer's activations and the layer's explanation | C3 |
| Neurons, steering vectors, SAE features, manifolds, crosscoder attribution graphs, block-sparse featurizers | F1 |
| Train SAEs on the grokking modular-addition transformer | F2 |
