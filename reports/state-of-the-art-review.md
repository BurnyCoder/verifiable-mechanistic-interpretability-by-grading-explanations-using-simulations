# State-of-the-art review — metrics, tooling and models this project builds on

*Libor "Burny" Burian, Victor Levoso — 2026-11-21. Companion to [literature-review.md](literature-review.md),
which covers the scientific background; this file covers the concrete methods and tools chosen, and why.*

## 1. Output fidelity: KL divergence conventions

- Interpretability work measures how far a modified model's output distribution is from the original's with
  **KL(original ‖ modified)**: ACDC uses KL(full ‖ subgraph) ([Conmy et al. 2023, arXiv:2304.14997](https://arxiv.org/abs/2304.14997));
  SAELens computes KL(original ‖ new) ([evals.py](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/evals.py)).
  Leo Gao argues KL is preferable to loss-delta because errors on different inputs can cancel in the loss
  ([post](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)).
- Normalised variants: SAELens `kl_div_score` = (KL_ablation − KL_SAE)/KL_ablation; SAEBench "loss recovered"
  ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532)); causal scrubbing "% loss recovered".
- **Choice here:** primary = mean over all 113² inputs of KL(P_M ‖ P_M̂) over the 113 number classes, in nats, float64
  (the input space is small enough to evaluate exactly, so no sampling error). A known property for grokked targets:
  P_M is nearly one-hot, so KL ≈ the replacement's cross-entropy on M's argmax labels; this is why H3 predicts that
  output KL alone cannot separate a correct-behaviour/wrong-mechanism replacement from a faithful one.

## 2. Hidden-state similarity: metrics, invariances, pitfalls

Formulas and invariance classes follow Klabunde et al., *Similarity of Neural Network Models*
([arXiv:2305.06329](https://arxiv.org/abs/2305.06329), ACM CSUR 2025) and Ding et al.
([arXiv:2108.01661](https://arxiv.org/abs/2108.01661)).

| Metric | Invariant to | Notes |
|---|---|---|
| Linear regression R² (cross-validated) | invertible linear maps of the predictor | asymmetric → compute both directions |
| Linear CKA ([Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)) | orthogonal maps, isotropic scaling | dominated by top PCs; manipulable ([Davari et al.](https://arxiv.org/abs/2210.16156)) |
| Orthogonal Procrustes (Ding et al.) | orthogonal maps (after normalisation) | consistently good baseline in Ding et al. |
| Hard neuron matching ([Li et al. 2015](https://arxiv.org/abs/1511.07543)) | permutation, per-neuron scaling | Hungarian assignment on correlations |
| Model stitching ([Bansal et al. 2021](https://arxiv.org/abs/2106.07682)) | whatever the stitch class allows | functional (causal) test |
| Shape metrics ([Williams et al. 2021](https://arxiv.org/abs/2110.14739)) | chosen group | true metrics |

**Matching the metric to the model's symmetries.** The target has no LayerNorm, so the residual stream is defined
only up to GL(128) (literature review §1). CKA and Procrustes are *not* invariant to that group: a functionally
identical replacement written in a different residual basis would be penalised. Hence the **primary** hidden-state
score is cross-validated linear-regression variance explained in both directions — invariant to exactly the
invertible linear maps the architecture allows, and no more (unrestricted non-linear maps make any network match any
algorithm: [Sutter et al. 2025](https://arxiv.org/abs/2507.08802)). With N = 12,769 inputs ≫ 128 dimensions, the
regression is well-determined; 5-fold cross-validation guards against over-fitting. MLP neurons are compared with
Hungarian-matched correlation (permutation and scale invariant), attention patterns directly after head matching.
CKA and Procrustes are reported as secondary metrics for comparability with the literature.

**Reference implementations** for cross-checking: netrep ([MIT](https://github.com/ahwillia/netrep)) and Google's CKA
code ([Apache-2.0](https://github.com/google-research/google-research/tree/master/representation_similarity)).
No maintained PyPI package covers regression R², CKA, Procrustes and neuron matching together (2026-11 check:
ReSi, similarity-repository and netrep install from git only), so the metrics are implemented in
`src/vmi/grader/` in a few dozen lines each and tested against those references.

## 3. Explanation-quality scoring: where the field is

- **Per-feature, correlation-based simulation** (Bills et al. 2023; Paulo et al. 2024) is the most used automated
  score and the most criticised: weak on false positives ([Huang et al.](https://arxiv.org/abs/2309.10312)),
  similar for trained and random models ([Heap et al.](https://arxiv.org/abs/2501.17727)), bypassable by simulators
  that ignore the explanation ([Poché et al. 2026](https://arxiv.org/abs/2609.08585)).
- **Substitution-based scores** (Bills' ablation score; Paulo & Belrose 2025; Gao's downstream-KL proposal) replace
  activations by explanation-driven predictions and measure output damage.
- **Whole-model rebuilds** exist for agent-written models (iWSP, [arXiv:2611.18114](https://arxiv.org/abs/2611.18114))
  and for compiled programs (Tracr), not for SGD-trained models from explanations.
- **Length accounting** is acknowledged as missing (Paulo et al. 2024) and handled in proofs by FLOPs
  ([Gross et al.](https://arxiv.org/abs/2406.11779)) and in SAEs by MDL ([Ayonrinde et al.](https://arxiv.org/abs/2410.11179)).
  Here every score is plotted against explanation length (characters, gzip bytes, number count), with the weights
  dump as the degenerate endpoint.

## 4. Agent harness: OpenCode, offline

- **OpenCode v2** (repo `anomalyco/opencode`, docs [opencode.ai/docs](https://opencode.ai/docs)): headless
  `opencode run --format json` emits an NDJSON event stream (text, reasoning with `--thinking`, tool calls with
  inputs/outputs); `--auto` approves permissions not explicitly denied; `--standalone` uses a private server.
  Permissions are ordered `{action, resource, effect}` rules (last match wins); `webfetch`, `websearch`, MCP, subagents
  and code mode can be denied; custom agents take a system prompt. The model catalog falls back to a built-in
  snapshot when offline; ripgrep must be on PATH to avoid a download. Installed version: **2.3.2** (Windows and a
  native Linux binary in WSL). Config keys were documented for 2.2.x; stage 0 validates them on 2.3.2.
- **Isolation without Docker or sudo:** bubblewrap `--unshare-net` in WSL2 (tested on this machine: external HTTP
  fails, `/mnt/c` and `/home` are hidden). The same pattern (bwrap + a unix-socket bridge) is what Claude Code's own
  sandbox uses on Linux/WSL2 ([docs](https://code.claude.com/docs/en/sandboxing)). The LLM endpoint is bridged into the
  network namespace through a unix socket by a stdlib Python relay (socat is not installed and there is no sudo).

## 5. Local model and server

- **Server: llama.cpp `llama-server`** (build b12600, CUDA 13.5, [releases](https://github.com/ggml-org/llama.cpp/releases)):
  OpenAI-compatible API, Jinja chat templates on by default (needed for tool calling), `--n-cpu-moe` keeps
  mixture-of-experts weights in system RAM, fixed model file (no pull endpoint, unlike Ollama's `/api/pull`),
  per-request seed. Ollama was rejected because its OpenAI endpoint ignores client context settings (default 4k on
  < 24 GiB GPUs breaks tool calls; [docs](https://docs.ollama.com/context-length)) and it has no `--n-cpu-moe`.
- **Model: Qwen3.6-35B-A3B** (35 B total, 3 B active, MoE; [model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)),
  GGUF `UD-IQ4_XS` (17.7 GB, [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), pinned revision
  `a483e9e`). Chosen as the strongest agentic-coding model that fits 8 GB VRAM + 31 GB RAM: independent Terminal-Bench
  ≈ 24.6 % vs 9.2 % for Qwen3.5-9B (reported by [llmrun.dev](https://llmrun.dev/benchmark/terminalbench); vendor
  numbers are higher). Reported ≈ 33 tok/s on an 8 GB RTX 3070 laptop with `--n-cpu-moe` (third-party report, to be
  measured here in stage 0). No usable Qwen3.8 fits this machine (27B is dense; Flash-Next's smallest GGUF ≈ 68 GiB).
- **Ablation models:** Qwen3.5-9B and Qwen3.5-4B (already pulled in Ollama; served through the same llama-server for
  comparability).

## 6. Target models and training code

- Reused, not re-implemented: `grokviz` (MIT, pinned commit `d52a898`) provides `GrokTransformer` with the paper's
  exact parameter layout and hooks, `RunStore`, the Fourier basis, key frequencies, neuron clusters, restricted and
  excluded loss, attention fits (paper Table 2), the W_L read-off (Table 1) and deterministic vectorised training.
  Its definitions reproduce the paper's and ARENA's published numbers (433 neurons above 0.85 FVE; restricted loss
  5.5474e-8 on the paper model), per its module docstrings.
- Targets: `BurnyCoder/grokking-modular-addition-runs` (pinned revision `98c61de`), trained on this same GPU model
  with torch 2.14.1+cu130.
