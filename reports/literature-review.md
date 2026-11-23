# Literature review — grading mechanistic explanations by simulation

*Libor "Burny" Burian, Victor Levoso — 2026-11-21. Written before any experiment of this project was run;
citation corrections and post-freeze facts (simulator, target contamination) added 2026-11-22; second-audit
corrections (Haklay, Heap, Poché, contamination wording) the same day.*

**How this review was made.** Primary sources (arXiv HTML/PDF, LaTeX sources, official repositories, LessWrong
posts and comments) were read in full by research subagents on 2026-11-20/21 and summarised with locations; every
arXiv ID below was then re-checked against the arXiv API (title, first author, date) on 2026-11-21. Claims that
could not be confirmed from a primary source are marked *unverified*. Numbers are quoted from the cited source,
not recomputed here.

---

## 1. The target: the modular-addition transformer of Nanda et al. 2023

**Paper.** Nanda, Chan, Lieberum, Smith, Steinhardt, *Progress measures for grokking via mechanistic
interpretability*, ICLR 2023 — [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) (v3);
code: [progress-measures-paper](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)
(no licence file).

**Model.** p = 113, input `a b =`, one layer, d_model 128, 4 heads × d_head 32, d_mlp 512, ReLU, no LayerNorm,
learned positional embeddings, untied embed/unembed, biases only in the MLP (official code `transformers.py`).
Full-batch AdamW, lr 1e-3, weight decay 1, 30 % of the 12,769 pairs for training. The released code adds what the
paper does not state: betas (0.9, 0.98), a 10-step linear warm-up and 114 output classes during training ("="
included); its default is 50,000 epochs, while the paper (§3) trains and analyses 40,000 (the analysed checkpoint).

**Algorithm ("Fourier multiplication", §3.1).** Embed a and b as cos/sin(w_k a), cos/sin(w_k b) for a few key
frequencies w_k = 2πk/p; form cos/sin(w_k(a+b)) with the angle-addition identities in attention and the MLP;
read out logit(c) ≈ Σ_k α_k cos(w_k(a+b−c)) through W_L = W_U W_out, where the terms interfere constructively at
c = (a+b) mod p. Mainline key frequencies: {14, 35, 41, 42, 52} (§4.1).

**How much the paper's explanation explains (quantitative claims).**
- Logits ≈ Σ_k α_k cos(w_k(a+b−c)) explains 95 % of logit variance (§4.2).
- 433 of 512 neurons have > 85 % of variance explained by a degree-2 polynomial in one frequency (§4.3).
- W_L is effectively rank 10; its residual outside the key directions is < 0.55 % of its norm (§4.2).
- Attention: each head's pattern A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with FVE 97.9–99.1 % (Table 2).
- Ablations (§4.4): replacing the 433 neurons with their polynomial fits raises loss 3 %; restricting each neuron to
  its key-frequency cos/sin(w_k(a+b)) terms lowers loss 77 % (to 5.54×10⁻⁸; the paper's attribution — grokviz's
  reproduction gets 1.12×10⁻⁷ for this ablation and 5.55×10⁻⁸ for the restricted logits, see the state-of-the-art
  review §6; audit 8); keeping only the key-frequency components
  of the logits lowers it 70 % (to 7.24×10⁻⁸); projecting MLP activations onto the nullspace of W_L's 10 key
  directions raises loss to 5.27, worse than uniform.
- **What the explanation leaves unspecified** (and what therefore limits weight reconstruction): per-neuron
  phases and gains; how neurons are allocated to frequencies (only the k=14 cluster size, 44, is published);
  79 poorly explained neurons; a sixth embedding frequency not used downstream; positional and "=" embeddings;
  MLP biases; the remaining 5 % of logit variance; and the overall logit scale, which appears only as a plot of the
  fitted α_k over training (App. C.1.5, Fig. 14), not as numbers. The paper's appendix notes that at
  perfect accuracy the loss keeps falling by scaling the logits up, which weight decay counteracts; that the scale is
  therefore set by this balance rather than by the algorithm is our inference.

**Symmetries that make raw weights incomparable** (our derivation; consistent with Klabunde et al., *Similarity of
Neural Network Models: A Survey of Functional and Representational Measures*,
[arXiv:2305.06329](https://arxiv.org/abs/2305.06329)): without
LayerNorm, any invertible A ∈ GL(128) applied to every residual-stream writer (and A⁻¹ to every reader) preserves the
function; per head W_Q→W_Q M, W_K→W_K M⁻ᵀ and W_V→W_V N, W_O→W_O N⁻¹; MLP neurons can be permuted and positively
rescaled; heads can be permuted. A grader must compare invariants (outputs, attention patterns, neuron activations up
to permutation/scale, residual stream up to an invertible linear map).

**Seed variability.** Other seeds of the same recipe learn 3–4 different key frequencies (paper App. C.2.1:
seed 1 {2, 9, 19, 31}, seed 2 {40, 44, 53}, ...); our re-trained runs (§6) learn 3–6 per seed. An explanation that does
not name the key frequencies cannot reproduce a specific model's hidden states.

## 2. Closely related work on the same mechanism

| Work | What it adds | Relevance here |
|---|---|---|
| Zhong et al. 2023, *The Clock and the Pizza* — [arXiv:2306.17844](https://arxiv.org/abs/2306.17844) | Two algorithms reach perfect accuracy: Clock (logits ≈ cos(w(a+b−c)), needs multiplication) and Pizza (logits ≈ \|cos(w(a−b)/2)\|·cos(w(a+b−c)), needs only \|·\|). Same accuracy but **different logit patterns** (§3.2, "First Evidence for Pizza: Logit Patterns"); also separated by distance irrelevance (computed from the correct-logit matrix, i.e. output-level; the decisive factor, §4.1.3) and gradient symmetricity (input gradients). Full attention at Nanda's width typically gives Clock, but not always (a width-1024 model with full attention learned Pizza, App. H). | Existence proof that this task family admits distinct algorithms at equal accuracy, so a replacement must be checked for *which* algorithm it implements. It is **not** a "same outputs, different internals" case: output KL separates the literal builds (reference compiler on T0, W-alg vs E3: paper Appendix A), though not the agents' builds (paper §4.1). Post hoc (audit 3) we apply Zhong et al.'s distance irrelevance: every target is Clock-like, but the replacements built from our true explanations are Pizza-like (paper §5). H3's claim that output KL is insufficient rests instead on Tracr (§4) and on near-one-hot targets ([state-of-the-art review §1](state-of-the-art-review.md)). |
| Gromov 2023, *Grokking modular arithmetic* — [arXiv:2301.02679](https://arxiv.org/abs/2301.02679) | Approximate closed-form weights (exact only as width grows) for a 2-layer MLP with x² activation: W⁽¹⁾ = cos(2πkn/p + φ), W⁽²⁾ = cos(−2πkq/p − φ₃), with φ₁ + φ₂ = φ₃; works for ReLU empirically at larger width. | A hand-written-weights construction for the same algorithm; template for what a "reference compiler" can do. |
| Yip, Agrawal, Chan, Gross 2024, *Modular addition without black-boxes* — [arXiv:2412.03773](https://arxiv.org/abs/2412.03773) | ReLU neurons implement a quadrature of ∫ReLU(cos(k(a+b)/2+φ))cos(kc+2φ)dφ ∝ cos(k(a+b−c)); per-neuron phase ψ ≈ 2φ; error bounds. Pizza models, p = 59. | Shows that per-neuron phases are needed for a faithful neuron-level account (our E4 level). |
| Gross et al. 2024, *Compact proofs of model performance* — [arXiv:2406.11779](https://arxiv.org/abs/2406.11779) | Explanation quality as a trade-off between proof length (asymptotic cost, estimated FLOPs) and bound tightness; brute force = exact but the longest of their strategies. Max-of-K, not modular addition. | The degenerate "explanation = the weights" endpoint; we report every score against explanation length. |
| Wu, Jaburi, Drori, Gross 2024 — [arXiv:2410.07476](https://arxiv.org/abs/2410.07476) | Builds an idealized model from the interpretation plus trained parameters to obtain accuracy bounds for group-operation networks. | Closest construction-from-interpretation; uses trained parameters, which our simulator is forbidden to see. |
| Chughtai, Chan, Nanda 2023 — [arXiv:2302.03025](https://arxiv.org/abs/2302.03025); Stander et al. 2023 — [arXiv:2312.06581](https://arxiv.org/abs/2312.06581) | Group-representation explanation; Stander et al. (ICML 2024) argue the models implement coset circuits instead, and Wu et al. 2024 find both accounts incomplete and partly reconcile them. | Cautionary tale: plausible explanations can be wrong — motivates wrong-explanation controls (W-*). |
| Morwani et al. 2023 — [arXiv:2311.07568](https://arxiv.org/abs/2311.07568) | Max-margin solutions of one-hidden-layer quadratic-activation networks use single-frequency cosine neurons with θ_u + θ_v = θ_w (Theorem 7). | Theory for the phase constraint the explainer reports. |

A targeted search found **no prior work that writes all weights of the Nanda transformer from a text explanation
alone and measures closeness to the trained model** (searched 2026-11-20; absence is not proof).

## 3. Automated scoring of interpretability explanations

- **Bills et al. 2023 (OpenAI)**, *Language models can explain neurons in language models*
  ([paper page](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)): explainer → simulator
  → score = correlation between simulated and true activations; an *ablation score* replaces the neuron by simulated
  activations and measures output Jensen–Shannon divergence normalised by mean ablation.
- **Paulo, Mallen, Juang, Belrose 2024**, *Automatically Interpreting Millions of Features in LLMs*
  ([arXiv:2410.13928](https://arxiv.org/abs/2410.13928); code [EleutherAI/delphi](https://github.com/EleutherAI/delphi)):
  five new scorers (detection, fuzzing, surprisal, embedding, intervention), compared against the earlier simulation
  scorer of Bills et al.; recommends several scorers; explicitly notes no scorer accounts for explanation length.
- **Paulo & Belrose 2025**, *Partially Rewriting a Transformer in Natural Language*
  ([arXiv:2501.18838](https://arxiv.org/abs/2501.18838)): replaces a transcoder's encoder by an LLM that predicts each
  latent from its explanation; measures output loss. Closest earlier "replace part of the model from its
  explanation" result, but at activation level with an LLM in the loop at inference time.
- **Huang et al. 2023**, *Rigorously Assessing Natural Language Explanations of Neurons*
  ([arXiv:2309.10312](https://arxiv.org/abs/2309.10312)): precision/recall and interchange-intervention tests; on 300
  GPT-2 XL neurons that GPT-4 scored highly, F1 was 0.56 and correlated −0.1 with the simulation score.
- **Kopf et al. 2024, CoSy** ([arXiv:2405.20331](https://arxiv.org/abs/2405.20331)) and Puri et al., *FADE: Why Bad
  Descriptions Happen to Good Features*, Findings ACL 2025 ([arXiv:2502.16994](https://arxiv.org/abs/2502.16994)) —
  explanation scoring for vision neurons (CoSy) and for LLM features, both MLP neurons and SAE features (FADE).
- **Leo Gao, *Neural networks are not already clean*** (LessWrong, 2026-10-17,
  [post](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)): argues
  that the correlation-based simulation scores of Bills et al. (of which Gao is a co-author) are flawed — they penalise
  activation differences that do not matter downstream and need careful binning — and proposes scoring by
  **substituting simulated activations and measuring downstream KL**, preferring KL to loss-delta because errors can cancel. It replaces activations, not
  weights, and reports no experiments. A commenter notes "the model is the optimal explanation of itself" — the
  length problem again.

## 4. Explanations compiled to or checked against weights

- **Tracr** (Lindner et al. 2023, [arXiv:2301.05062](https://arxiv.org/abs/2301.05062)): compiles RASP programs to
  transformer weights. Its compressed models are a published case of same behaviour via different internal
  encodings, caught by layer-output comparison (cosine ≈ 0.8, `sort_unique`) despite near-perfect accuracy.
- **InterpBench / SIIT** (Gupta et al. 2024, [arXiv:2407.14494](https://arxiv.org/abs/2407.14494)): semi-synthetic
  transformers with known circuits as ground truth.
- **Causal abstraction** (Geiger et al., [arXiv:2301.04709](https://arxiv.org/abs/2301.04709)) and **causal scrubbing**
  (Chan et al. 2022, [appendix](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)):
  faithfulness via interchange interventions / resampling ablations. **Sutter et al. 2025**
  ([arXiv:2507.08802](https://arxiv.org/abs/2507.08802)): under their assumptions (countable inputs, input-injective
  and strictly output-surjective hidden layers, matchable partial orderings with at least one unused neuron per layer, a network that solves the task), with
  unrestricted alignment maps any network can be mapped to any algorithm — the alignment class must be decided case
  by case. This is why our hidden-state metric uses linear maps only (justified by the residual stream's symmetry; at
  the MLP site linear maps are wider than the neurons' permutation-and-scaling symmetry, paper §3.4).
- **MIPS** (Michaud et al. 2024, [arXiv:2402.05110](https://arxiv.org/abs/2402.05110)): distils RNNs into Python
  programs; success = program reaches 100 % accuracy.
- **MIB** (Mueller et al. 2025, [arXiv:2504.13151](https://arxiv.org/abs/2504.13151)): circuit faithfulness
  f = (m(C) − m(∅)) / (m(N) − m(∅)) and interchange-intervention accuracy.
- **Open Problems in Mechanistic Interpretability** (Sharkey et al. 2025,
  [arXiv:2501.16496](https://arxiv.org/abs/2501.16496)): descriptions should be validated by whether they predict
  counterfactual behaviour; discusses hand-crafting replacement networks, citing Cammarata et al. 2020,
  [*Curve Detectors*](https://distill.pub/2020/circuits/curve-detectors/) (Distill), which Sharkey et al. describe as
  validating its account with handcrafted replacements; *Curve Detectors* itself only mentions the hand-made
  reimplementation ("reimplemented it by hand"; a "hand-trained 'artificial artificial neural network'") without
  presenting it. That article,
  [*Curve Circuits*](https://distill.pub/2020/circuits/curve-circuits/) (Cammarata et al., Distill 2021), builds a
  hand-crafted curve network from scratch and compares its neurons with InceptionV1's — our idea, done by hand for one
  circuit.
- **Parameter decomposition:** APD ([arXiv:2501.14926](https://arxiv.org/abs/2501.14926), "…Minimizing Mechanistic
  Description Length") fits replacement components to the weights and frames this as minimising mechanistic
  description length; SPD ([arXiv:2506.20790](https://arxiv.org/abs/2506.20790)) presents itself as a more scalable
  linear parameter decomposition.

## 5. LLMs and agents that write weights — the novelty check

- **iWSP: Mehta et al. 2026, *Interpretable weight-space programming builds transparent transformers and brain
  encoding models*** ([arXiv:2611.18114](https://arxiv.org/abs/2611.18114), 2026-11-13;
  [code](https://github.com/csinva/weight-space-programming-autoresearch)): Claude Code writes the weights of small
  transformers directly; a *rebuild test* gives a fresh session a ≤ 200-word description and scores the rebuilt model
  by accuracy, parameter ratio and prediction correlation, against a no-description control. **Difference from this
  project:** applied only to agent-written models (never SGD-trained ones); no output-KL comparison, and the only
  representation comparison is linear CKA between original and rebuilt features for one model (the visual-cortex
  model, App. A.2). iWSP builds on Takehi et al. (2026-11-02), *Sparks of weight-space programming* (blog,
  [rikiyat.github.io](https://rikiyat.github.io/blog/weight-space-programming/)), where an LLM reads, edits and writes
  network weights.
- **Weight Oracles** (Kabra et al. 2026, [arXiv:2610.07334](https://arxiv.org/abs/2610.07334)): LMs read weights to
  simulate small transformers — reading, not writing.
- **Code as Explanations** (Nguyen et al. 2026, [arXiv:2611.14704](https://arxiv.org/abs/2611.14704)): a program
  explains an LLM's behaviour, scored by output agreement — behavioural only. An earlier benchmark in this direction is
  **FIND** (Schwettmann et al., NeurIPS 2023, [arXiv:2309.03886](https://arxiv.org/abs/2309.03886)): functions that
  resemble components of trained networks, paired with natural-language and code descriptions, against which
  interpretability agents' descriptions are evaluated.
- **The November 2026 weight-space-programming wave** (all fetched; claims from the abstracts, audit 8): agents edit the
  weights of trained Qwen3.5 models ([Hu et al., arXiv:2611.09382](https://arxiv.org/abs/2611.09382)); Claude Opus 5.5
  reads trained CNNs' weights and relies on dataset priors rather than tracing weight flow
  ([Ahmed, arXiv:2611.08070](https://arxiv.org/abs/2611.08070)); LLMs write all weights of a network from game rules
  given as simulator source ([Lehmann, arXiv:2611.16981](https://arxiv.org/abs/2611.16981)), of a small medical-imaging
  CNN without training data ([Lindholz et al., arXiv:2611.12066](https://arxiv.org/abs/2611.12066)), of a policy checked
  by probabilistic model checking ([Gross, arXiv:2611.11604](https://arxiv.org/abs/2611.11604)), or of a PDE solver
  from governing equations ([Sato et al., arXiv:2611.12954](https://arxiv.org/abs/2611.12954)). None rebuilds an
  SGD-trained network from an explanation of it or compares hidden states with an original, so the novelty statement
  below stands.
- **Interpretability agents and their evaluation:** MAIA ([arXiv:2404.14394](https://arxiv.org/abs/2404.14394));
  AgenticInterpBench ([arXiv:2606.24026](https://arxiv.org/abs/2606.24026)); **Haklay et al. 2026, *Pitfalls in
  Evaluating Interpretability Agents*** ([arXiv:2603.20101](https://arxiv.org/abs/2603.20101)) — asked directly, the
  LLM behind an interpretability agent (Claude Opus 4.1) listed the IOI circuit's components from memory (§4.3.1), so
  agents may recall rather than discover published circuits; Pando ([arXiv:2604.11061](https://arxiv.org/abs/2604.11061)).
- **Critiques of auto-interp scores:** Heap et al. 2025 (v2 2026, retitled) — fuzzing scores of SAE latents (detection
  in an appendix) are similar for trained and randomised transformers in many settings, with a larger gap for small
  Pythia models ([arXiv:2501.17727](https://arxiv.org/abs/2501.17727)); **Poché et al. 2026 — LLM simulators can bypass
  explanations** ([arXiv:2609.08585](https://arxiv.org/abs/2609.08585)): when LLM simulators predict a text
  classifier's outputs from concept explanations (a ConSim replication with meaningful class names), explanation gains
  over no-explanation baselines were statistically significant but small relative to their variability (in an extension with four simulators, attributions improve significantly only with Qwen, and with Llama attributions and rationales fall significantly below the no-explanation baseline, App. J–K), which the authors explain by a "shortcut hypothesis" (predictions rest mainly on task
  priors — with meaningful class names the simulator can solve the task directly — and explanations change them only
  slightly; simulators of at most 15B parameters without reasoning); they
  recommend label-leakage baselines, multiple simulators, enough seeds and novel tasks;
  Lermen et al. 2025 — explainer and simulator can collude via steganography
  ([arXiv:2504.07831](https://arxiv.org/abs/2504.07831)).

**Honest novelty statement.** Each ingredient has precedent. What we did not find is the combination: an
**SGD-trained** target, a **weights-blind, offline** agent that compiles a **mechanistic explanation** into a full
replacement of the same architecture, scored by **output KL plus symmetry-aware hidden-state similarity**, with
**explanation-length accounting** and **no-explanation, wrong-explanation, wrong-target and planted-mechanism
controls**. The contribution is that combination and a careful validation of it, not a new concept.

## 6. Reusable resources (checked for licence)

| Resource | Licence | Use |
|---|---|---|
| `BurnyCoder/grokking-modular-addition-runs` ([HF](https://huggingface.co/BurnyCoder/grokking-modular-addition-runs), created and uploaded 2026-10-21): 22 one-layer models trained deterministically — 14 with the paper's settings (11 seeds `wd1_s0`–`wd1_s10`, plus 3 initialisations on one split, `split0_init100`–`102`), 5 with weight decay 0 (`wd0_s*`), 3 random-label controls (`rand_s*`); 40,000 steps (the official code defaults to 50,000); 401 checkpoints each | MIT | Targets, retrain baselines, no-mechanism controls |
| `grokviz` ([GitHub](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer)): paper-faithful model, training, Fourier analysis, paper-checkpoint conversion | MIT | Analysis and training code (pinned commit) |
| Paper mainline checkpoint ([HF mirror](https://huggingface.co/callummcdougall/grokking_full_run_data), byte-identical to [ARENA's](https://huggingface.co/arena-education/grokking_full_run_data)) | none stated | Target T-paper; not redistributed |
| ARENA 1.5.2 code ([ARENA_materials](https://github.com/ARENA-education/ARENA_materials)) | none | Not copied; its published numbers used as test oracles |
| netrep ([GitHub](https://github.com/ahwillia/netrep)), Google CKA ([google-research](https://github.com/google-research/google-research/tree/master/representation_similarity)) | MIT / Apache-2.0 | Reference implementations considered for cross-checking; the tests use Google's CKA (re-derived) and scipy's orthogonal Procrustes, not netrep |

## 7. Methodology guidance followed

Lipton & Steinhardt, *Troubling Trends* ([arXiv:1807.03341](https://arxiv.org/abs/1807.03341)); Lones, *How to avoid
ML pitfalls* ([arXiv:2108.02497](https://arxiv.org/abs/2108.02497)); Nanda's research-process sequence
([Alignment Forum](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL)); NeurIPS checklist
([guide](https://neurips.cc/public/guides/PaperChecklist)). Concretely: preregistered hypotheses and falsifiers,
discriminating controls, explanation-free baselines, contamination controls, and reporting of negative results.
Contamination: no paper analyses targets T0–T2; their weights and an automated analysis that lists their key
frequencies were uploaded to Hugging Face only on 2026-10-21 — after Claude Haiku 5.5's training-data cutoff
(June 2026, [models overview](https://platform.claude.com/docs/en/models/overview)) and after its release on
2026-10-07 ([model page](https://platform.claude.com/docs/en/models/haiku-5-5/overview)) — so they cannot be
in its training data, and the sandbox has no internet access; a held-out target was trained after the protocol freeze. Run `wd1_s0` learned exactly the paper's
frequencies {14, 35, 41, 42, 52}, so it is not a clean target; it serves as the paper's-twin baseline, not as a target.
