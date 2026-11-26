# State-of-the-art review — metrics, tooling and models this project builds on

*Libor "Burny" Burian, Victor Levoso — 2026-11-21. Companion to [literature-review.md](literature-review.md),
which covers the scientific background; this file covers the concrete methods and tools chosen, and why. Second-audit
corrections 2026-11-22.*

## 1. Output fidelity: KL divergence conventions

- Interpretability work measures how far a modified model's output distribution is from the original's with
  **KL(original ‖ modified)**: ACDC uses KL(full ‖ subgraph) ([Conmy et al. 2023, arXiv:2304.14997](https://arxiv.org/abs/2304.14997));
  SAELens computes KL(original ‖ new) ([evals.py](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/evals.py)).
  Leo Gao argues KL is preferable to loss-delta because errors can cancel in the loss — in his footnote 5, badly
  modelled neurons are offset by other neurons
  ([post](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)).
- Normalised variants: SAELens `kl_div_score` = (KL_ablation − KL_SAE)/KL_ablation; SAEBench "loss recovered"
  ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532)); causal scrubbing "% loss recovered"
  ([Chan et al. 2022](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing-interpretability-hypotheses);
  defined in the [appendix, §2.1](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix)).
- **Choice here:** primary = mean over all 113² inputs of KL(P_M ‖ P_M̂) over the 113 number classes, in nats, float64
  (the input space is small enough to evaluate exactly, so no sampling error). A known property for grokked targets:
  P_M is nearly one-hot, so KL ≈ the replacement's cross-entropy on M's argmax labels; this is why H3 predicts that
  output KL alone cannot separate a correct-behaviour/wrong-mechanism replacement from a faithful one. (Zhong et
  al.'s Clock and Pizza, [arXiv:2306.17844](https://arxiv.org/abs/2306.17844), are not such a pair: their logit
  patterns differ.)

## 2. Hidden-state similarity: metrics, invariances, pitfalls

Formulas and invariance classes follow Kornblith et al. 2019 (Table 1), Klabunde et al., *Similarity of Neural Network Models*
([arXiv:2305.06329](https://arxiv.org/abs/2305.06329), ACM CSUR 2025) and Ding et al.
([arXiv:2108.01661](https://arxiv.org/abs/2108.01661)).

| Metric | Invariant to | Notes |
|---|---|---|
| Linear regression R² (cross-validated) | invertible linear maps of the predictor ([Kornblith et al. 2019](https://arxiv.org/abs/1905.00414), Table 1, "Y only"); orthogonal maps, isotropic scaling, translation (Klabunde et al. Table 1) | asymmetric → compute both directions |
| Linear CKA ([Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)) | orthogonal maps, isotropic scaling | dominated by top PCs ([Ding et al.](https://arxiv.org/abs/2108.01661) §5: insensitive to differences in smaller singular values); manipulable ([Davari et al.](https://arxiv.org/abs/2210.16156)) |
| Orthogonal Procrustes (Ding et al.; their eq. 7 is the squared distance) | orthogonal maps, and isotropic scaling and translation after centring and normalisation | consistently good baseline in Ding et al. |
| Hard neuron matching ([Li et al. 2016, ICLR; arXiv 2015](https://arxiv.org/abs/1511.07543)) | permutation, positive per-neuron scaling | optimal one-to-one assignment on correlations (scipy [`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), modified Jonker–Volgenant) |
| Model stitching ([Bansal et al. 2021](https://arxiv.org/abs/2106.07682)) | whatever the stitch class allows | functional (causal) test; Bansal et al. train the stitch on the task loss, this project fits an affine stitch by least squares to M's activations (the variant in Klabunde et al. §4.5) |
| Shape metrics ([Williams et al. 2021](https://arxiv.org/abs/2110.14739)) | chosen group | true metrics |

**Matching the metric to the model's symmetries.** The target has no LayerNorm, so the residual stream is defined
only up to GL(128) (literature review §1). CKA and Procrustes are *not* invariant to that group: a functionally
identical replacement written in a different residual basis would be penalised. Hence the **primary** hidden-state
score is cross-validated linear-regression variance explained in both directions; the alignment class is linear maps.
Unregularized regression absorbs invertible changes to the predictor basis. The implemented isotropic ridge
breaks this general invariance. Squared errors pooled over the response coordinates also depend on their scale.
The post-hoc whitened variant equalizes retained response directions, but the relative eigenvalue cutoff may
change the retained subspace. The strength of either dependence is empirical. [Kornblith et al., Table 1](https://proceedings.mlr.press/v97/kornblith19a.html) · [Grading code](../src/vmi/grader/metrics.py) · [Whitening code](../src/vmi/sensitivity.py).
Restricting the alignment to linear maps is this project's assumption, motivated by
[Sutter et al. 2025](https://arxiv.org/abs/2507.08802): their Theorem 1 is an existence result — with unrestricted
non-linear alignment maps, and under assumptions such as countable inputs and input-injective hidden states, any
network can be mapped to any algorithm. They leave the map class to be decided case by case, noting that linear maps
"may be advised" if one accepts the linear representation hypothesis. With N = 12,769 inputs ≫ 128 dimensions, the
regression is well-determined; 5-fold cross-validation guards against over-fitting. MLP neurons are compared by
correlation after optimal one-to-one assignment (permutation and scale invariant), attention patterns directly after
head matching. The GL(128) argument covers the residual sites only: the primary HS also regresses at `mlp_post`,
where the neurons' symmetry is permutation and positive scaling, so the linear alignment there is wider than the
symmetry group and cannot see per-neuron assignments (audit 3, deviation 10); the matched-neuron correlation is the
group-matched comparison. CKA and Procrustes are reported as secondary metrics for comparability with the literature.

**Reference implementations** for cross-checking: netrep ([MIT](https://github.com/ahwillia/netrep)) and Google's CKA
code ([Apache-2.0](https://github.com/google-research/google-research/tree/master/representation_similarity)).
The project review did not identify a maintained PyPI package covering regression R², CKA, Procrustes and neuron matching together (2026-11 check: the
READMEs of ReSi, similarity-repository and netrep install from git; a `similarity-repository` 0.0.1 and `repsim` exist
on PyPI), so the metrics are implemented in
`src/vmi/grader/` in a few dozen lines each and tested against Google's `feature_space_linear_cka` (re-derived in the
test) and `scipy.linalg.orthogonal_procrustes`; netrep was not used in the end (audit 4).

## 3. Explanation-quality scoring: where the field is

- **Per-feature activation-prediction scores** — correlation-based simulation (Bills et al. 2023) and the detection
  and fuzzing scorers of [Paulo et al. 2024](https://arxiv.org/abs/2410.13928) — are the most used automated scores
  and the most criticised: simulation scores are weak on false positives
  ([Huang et al.](https://arxiv.org/abs/2309.10312)); fuzzing scores of SAE latents (detection in an appendix) are similar
  for trained and random transformers in many settings ([Heap et al.](https://arxiv.org/abs/2501.17727), v2 2026). Relatedly, when
  LLM simulators predict a text classifier's outputs from concept explanations (not per-feature activations), the gains
  over matched no-explanation baselines are statistically significant but small relative to their variability (in an extension with four simulators, attributions improve significantly only with Qwen, and with Llama attributions and rationales fall significantly below the no-explanation baseline), which the authors explain by a "shortcut hypothesis": the simulator's
  predictions rest mainly on task priors (with meaningful class names it can solve the task directly), and explanations
  change them only slightly ([Poché et al. 2026](https://arxiv.org/abs/2609.08585)).
- **Substitution-based scores** (Bills' ablation score; [Paulo & Belrose 2025](https://arxiv.org/abs/2501.18838);
  Gao's downstream-KL proposal) replace activations by explanation-driven predictions and measure output damage.
- **Whole-model rebuilds** exist for agent-written models (iWSP, [arXiv:2611.18114](https://arxiv.org/abs/2611.18114))
  and for compiled programs (Tracr, [arXiv:2301.05062](https://arxiv.org/abs/2301.05062)), not for SGD-trained models
  from explanations.
- **Length accounting** is acknowledged as missing ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)) and handled
  in proofs by FLOPs
  ([Gross et al.](https://arxiv.org/abs/2406.11779)) and in SAEs by MDL ([Ayonrinde et al.](https://arxiv.org/abs/2410.11179)).
  Here every score is plotted against explanation length (characters, gzip bytes, number count), with the weights
  dump as the degenerate endpoint.

## 4. Agent harness: OpenCode, offline

> **Update 2026-11-22.** Preregistration amendment 2 replaced OpenCode with a minimal bash-only harness (following
> [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)) and made Claude Haiku 5.5 in Claude Code the main
> simulator; deviation 5 stopped the local Qwen runs because every run was at chance. Claude Haiku 5.5 returns an error for
> non-default `temperature`, `top_p` or `top_k` ([thinking docs](https://platform.claude.com/docs/en/build-with-claude/thinking); the
> [migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide) says to remove them), so Claude runs use default sampling. §4–5 below are kept unchanged as the pre-run review,
> apart from citation fixes.

- **OpenCode v2** (repo `anomalyco/opencode`, docs [opencode.ai/docs](https://opencode.ai/docs)): headless
  `opencode run --format json` emits an NDJSON event stream (text, reasoning with `--thinking`, tool calls with
  inputs/outputs); `--auto` approves permissions not explicitly denied (these three flags are documented only in the
  v2.3.2 source, `packages/cli/src/commands/commands.ts` and `packages/cli/src/run/noninteractive.ts`, not on the docs site);
  `--standalone` uses a private server (docs).
  Permissions are ordered `{action, resource, effect}` rules (last match wins); `webfetch`, `websearch`, MCP, subagents
  and code mode can be denied; custom agents take a system prompt. Local observations on this machine (not in the
  docs): the model catalog falls back to a built-in snapshot when offline, and ripgrep must be on PATH to avoid a
  download. Installed version: **2.3.2** (Windows and a
  native Linux binary in WSL). Config keys were documented for 2.2.x; stage 0 validates them on 2.3.2.
- **Isolation without Docker or sudo:** bubblewrap `--unshare-net` in WSL2 (tested on this machine: external HTTP
  fails, `/mnt/c` and `/home` are hidden). A similar pattern — bwrap with its own network namespace, plus a socat relay
  over Unix domain sockets to a host-side proxy — is what Claude Code's own sandbox uses on Linux/WSL2
  ([docs](https://code.claude.com/docs/en/sandboxing); [sandbox-runtime README](https://github.com/anthropics/sandbox-runtime)). The LLM endpoint is bridged into the
  network namespace through a unix socket by a stdlib Python relay (socat is not installed and there is no sudo).

## 5. Local model and server

- **Server: llama.cpp `llama-server`** (build b12600, Windows build with the CUDA 13 runtime, [releases](https://github.com/ggml-org/llama.cpp/releases)):
  OpenAI-compatible API, Jinja chat templates on by default (needed for tool calling), `--n-cpu-moe` keeps
  mixture-of-experts weights in system RAM, fixed model file (no pull endpoint, unlike Ollama's `/api/pull`),
  per-request seed. Ollama was rejected because its OpenAI-compatible endpoint does not set the context size
  ([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), so the default applies (4k on
  < 24 GiB GPUs, which breaks tool calls; [context length](https://docs.ollama.com/context-length)), and it has no
  `--n-cpu-moe` equivalent (requests for exact MoE expert-offload controls are open issues on
  [ollama/ollama](https://github.com/ollama/ollama/issues?q=is%3Aissue+expert+offload); our check, 2026-11-22).
- **Model: Qwen3.6-35B-A3B** (35 B total, 3 B active, MoE; [model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)),
  GGUF `UD-IQ4_XS` (17.7 GB, [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), pinned revision
  `a483e9e`). Chosen as the strongest agentic-coding model that fits 8 GB VRAM + 31 GB RAM: independent Terminal-Bench
  ≈ 24.6 % vs 9.2 % for Qwen3.5-9B (attributed by
  [llmrun.dev](https://llmrun.dev/benchmark/terminalbench) to Epoch AI, benchmark version not stated; vendor numbers are
  higher). A throughput of ≈ 33 tok/s on an
  8 GB RTX 3070 laptop with `--n-cpu-moe` was an unsourced third-party estimate for a different GPU than ours (RTX 5070
  Laptop); it was measured here in stage 0 instead (amendment 1). No
  usable Qwen3.8 fits this machine ([27B](https://huggingface.co/Qwen/Qwen3.8-27B) is dense;
  [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)'s smallest unpruned GGUFs are ≈ 66 GB / 62 GiB, unsloth
  UD-IQ1_S, or 58.5 GB for a third-party TQ1_0 quant ([agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)), all above 8 GB VRAM + 31 GB RAM; smaller third-party builds are
  expert-pruned (REAP, e.g. a ≈ 20 GB GGUF that also needs a separate 28.8 GB n-gram table, [anik-jha/Qwen3.8-Flash-Next-24GB-GGUF](https://huggingface.co/anik-jha/Qwen3.8-Flash-Next-24GB-GGUF)) or embedding-split, which we excluded as no longer the published model; a 4-bit
  27B would fit but is dense and too slow with CPU offload (our judgement); checked on the Hub 2026-11-22).
- **Ablation models:** Qwen3.5-9B and Qwen3.5-4B (already pulled in Ollama; served through the same llama-server for
  comparability).

## 6. Target models and training code

- Reused, not re-implemented: `grokviz` ([GitHub](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer),
  MIT, pinned commit `d52a898`) provides `GrokTransformer` with the paper's
  exact parameter layout and hooks, `RunStore`, the Fourier basis, key frequencies, neuron clusters, restricted and
  excluded loss, attention fits (paper Table 2), the W_L read-off (Table 1) and deterministic vectorised training.
  Its definitions reproduce the paper's and ARENA's published numbers (433 neurons above 0.85 FVE; loss 5.5474e-8 on
  the paper model when the logits are projected onto the 10 key-frequency cos/sin(w_k(a+b)) directions and bias-corrected,
  i.e. the restricted loss of the paper code's `calculate_trig_loss` and ARENA's `trig_loss`, agreeing with Nanda et al.
  §4.4's 5.54×10⁻⁸ to within 0.2 %; restricting each MLP neuron to its key-frequency terms instead gives 1.12e-7 and does
  not reproduce it, and §4.4's other logit-level ablation gives 7.24×10⁻⁸), per its module docstrings and its
  reproduction table (`reports/tables/exp01_rchecks.md` in grokviz at `d52a898`, row R10b; corrected in audit 6).
- Targets: `BurnyCoder/grokking-modular-addition-runs` (pinned revision `98c61de`), trained on this same GPU model
  with torch 2.14.1+cu130.
