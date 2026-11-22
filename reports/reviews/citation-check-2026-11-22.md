# Citation check — 2026-11-22 (part of the independent audit)

**What was checked.** Every external source cited in the README, summary, paper, preregistration, literature and
state-of-the-art reviews, experiment reports, learnings, plan, docs and code docstrings was fetched on 2026-11-22.
Each was checked for three things: that it exists at the URL or ID; that its title, first author and year match; and
that it supports the specific claim attributed to it. Sources were not judged from memory, because many date from
2025–2026.

**Verdicts.** *verified* = all three hold. *fixed* = the source exists, but the text claimed more than it shows or got
its metadata wrong; the text was corrected. *marked* = no source could be confirmed, so the claim is now labelled an
estimate, our inference or a local observation. Nothing was removed outright. The corrections are recorded in
[independent-audit-2026-11-22.md](independent-audit-2026-11-22.md), row 10.

## Fixed

| Source | Was cited for | What it shows | Where fixed |
|---|---|---|---|
| [iWSP, Mehta et al. 2026](https://arxiv.org/abs/2611.18114) | rebuilds without hidden-state comparison | Agent-written models only, no output KL; but it has a no-description control and a feature-CKA comparison (App. A.2) | README, paper, novelty claims (the original approved plan in `plan.md` keeps its wording as a record) |
| [Zhong et al. 2023](https://arxiv.org/abs/2306.17844) | Pizza = same outputs, different mechanism | Same accuracy; Pizza's logits carry an extra \|cos(w(a−b)/2)\| factor; "distance irrelevance" is an output-level metric | paper §2, literature review |
| [Heap et al. 2025](https://arxiv.org/abs/2501.17727) | simulation scores fail to separate trained from random | Fuzzing and detection scores of SAE features, "in many settings" | README, paper, both reviews |
| [Poché et al. 2026](https://arxiv.org/abs/2609.08585) | simulators bypass explanations | Output simulatability of text classifiers (ConSim replication) with meaningful class names; recommends label-leakage baselines and novel tasks | README, paper, both reviews |
| [Huang et al. 2023](https://arxiv.org/abs/2309.10312) | simulation scores disagree with causal tests | High-scoring explanations show little causal efficacy; the two are not compared directly | paper §1 |
| [Paulo et al. 2024](https://arxiv.org/abs/2410.13928) | simulation correlated with activations; "five scorers incl. simulation" | Five new classification-style scorers; simulation is the earlier baseline | paper §1, literature review |
| [Haklay et al. 2026](https://arxiv.org/abs/2603.20101) | agents reproduce published circuits from memory | The agent's LLM recalled one circuit (IOI) from memory; memorisation alone does not explain the agent | paper |
| [Gross et al. 2024](https://arxiv.org/abs/2406.11779) | weights as the longest explanation | Brute-force evaluation on every input is the longest proof | paper §2 |
| [Sutter et al. 2025](https://arxiv.org/abs/2507.08802) | non-linear alignment makes any network match any algorithm; linear advised | Existence result for causal abstraction under assumptions; the map class is decided case by case | paper, SOTA review, preregistration erratum |
| [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414) | HS invariant to exactly GL(d) | Table 1: regression R² is invariant to invertible linear maps of the predictor only | finding 1 (every file) |
| [Geiger et al.](https://arxiv.org/abs/2301.04709) | causal abstraction (no year) | v4 title (2025), v1 2023 | paper |
| [Gao 2026](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean) | KL beats loss delta because errors cancel across inputs | Cancellation across neurons; Bills et al.'s ablation scoring is the precedent | SOTA review, README, paper |
| [Cammarata et al.](https://distill.pub/2020/circuits/curve-detectors/) | Sharkey et al. cite *Curve Circuits* | They cite *Curve Detectors* (2020); the hand-built network is in *Curve Circuits* (2021) | literature review |
| [APD](https://arxiv.org/abs/2501.14926) / [SPD](https://arxiv.org/abs/2506.20790) | SPD framed as MDL | The MDL framing is APD's | literature review |
| [Tracr](https://arxiv.org/abs/2301.05062) | wrong encoding caught "only" by hidden states | Caught by layer-output cosine ≈ 0.8 despite near-perfect accuracy (sort_unique) | literature review |
| [Nanda et al. 2023](https://arxiv.org/abs/2301.05217) §4.4 | restricted loss 5.54·10⁻⁸ | That is the MLP-neuron restriction; the logit restriction gives 7.24·10⁻⁸ | both reviews |
| [BurnyCoder/grokking-modular-addition-runs](https://hf.co/BurnyCoder/grokking-modular-addition-runs) | 22 runs with the official recipe; "unpublished" | 14 with the paper's settings, 5 with weight decay 0, 3 random-label; 40,000 steps; public since 2026-10-21 | literature review, every "unpublished" |
| [Claude API docs](https://platform.claude.com/docs/en/about-claude/models/overview) | hidden thinking not returned | Thinking text is omitted by default; summaries can be requested | README, learnings |
| Statistics sources | Jonckheere 1954 for J; Schönemann 1966 for the scaled Procrustes | J is Terpstra's 1952 form (equivalent to S = P − Q); the scale is Schönemann & Carroll 1970 | `stats.py`, `sensitivity.py` |
| [Epoch AI via llmrun.dev](https://llmrun.dev/benchmark/terminalbench) | Terminal-Bench numbers from llmrun.dev | llmrun re-hosts Epoch AI's scores | SOTA review |
| Causal scrubbing; Paulo & Belrose; FADE; Klabunde et al. | no link, or ambiguous | Links added: [Chan et al. 2022](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing-interpretability-hypotheses), [arXiv:2501.18838](https://arxiv.org/abs/2501.18838), [arXiv:2502.16994](https://arxiv.org/abs/2502.16994), [arXiv:2305.06329](https://arxiv.org/abs/2305.06329) | paper, both reviews |

## Marked (no source could be confirmed)

- **Third-party estimates:** the ≈ 33 tok/s for an 8 GB GPU and the ≈ 68 GiB for the smallest Qwen3.8-Flash-Next GGUF
  are now marked as unchecked estimates (state-of-the-art review).
- **Local observations:** OpenCode's offline catalog snapshot and its ripgrep download are now marked as such.
- **Our inference:** the logit scale being set by the balance of weight decay and loss.
- **Development notes:** amendment 2 item 3's dev check and some learnings figures are not produced by a committed
  script. They are now labelled as notes (deviation 8 f).
- **User-provided:** the mechinterp.com hackathon. The site exists, but the hackathon was not found on it; this is
  the authors' own statement.

## Verified

- **Papers.** All arXiv IDs cited anywhere resolve with the right title, first author and year (checked through the
  arXiv API). The attributed claims hold for:
  - [Nanda et al. 2023](https://arxiv.org/abs/2301.05217): architecture, recipe, frequencies {14, 35, 41, 42, 52},
    Tables 1–2, 433 neurons;
  - [Conmy et al. 2023](https://arxiv.org/abs/2304.14997): the KL direction;
  - [Ding et al. 2021](https://arxiv.org/abs/2108.01661): the Procrustes distance;
  - [Bansal et al. 2021](https://arxiv.org/abs/2106.07682);
  - [Klabunde et al.](https://arxiv.org/abs/2305.06329);
  - [Williams et al. 2021](https://arxiv.org/abs/2110.14739);
  - [Li et al. 2016](https://arxiv.org/abs/1511.07543);
  - [Gromov 2023](https://arxiv.org/abs/2301.02679), [Yip et al. 2024](https://arxiv.org/abs/2412.03773),
    [Morwani et al.](https://arxiv.org/abs/2311.07568), [Chughtai et al.](https://arxiv.org/abs/2302.03025),
    [Stander et al.](https://arxiv.org/abs/2312.06581), [Wu et al.](https://arxiv.org/abs/2410.07476);
  - the SIIT, MIPS, MIB, Weight Oracles, Code as Explanations, MAIA, Pando and AgenticInterpBench papers;
  - [Lipton & Steinhardt](https://arxiv.org/abs/1807.03341), [Lones](https://arxiv.org/abs/2108.02497);
  - [Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html);
  - [Cammarata et al. 2021](https://distill.pub/2020/circuits/curve-circuits/).
- **Library and tool docs.**
  - scipy: `wilcoxon` is exact for n ≤ 50 without ties or zeros; `mannwhitneyu` exact; `linear_sum_assignment`;
    `orthogonal_procrustes`.
  - [PyTorch `lstsq`](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) (the CUDA driver assumes full
    rank); [man 7 unix](https://man7.org/linux/man-pages/man7/unix.7.html) (108-byte `sun_path`).
  - The [Haiku 5.5 migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide): no sampling
    parameters; effort `medium` is the default.
  - Claude Code 2.1.330 and `--bare` (npm); llama.cpp b12600; bubblewrap 0.9.0; PyTorch 2.14 cu130;
    [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent); Ollama context defaults; the Qwen3.6 model card's
    coding preset.
- **Models and repositories.**
  - Claude Haiku 5.5: id `claude-haiku-5-5`, released 7 October 2026, training-data cutoff June 2026
    ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)).
  - Hugging Face: [results repo](https://huggingface.co/BurnyCoder/grading-explanations-by-simulation), the
    collection, the six example replacement folders, [callummcdougall/grokking_full_run_data](https://huggingface.co/callummcdougall/grokking_full_run_data)
    (same file as the ARENA mirror), the Qwen and unsloth repositories.
  - GitHub: [grokviz](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer) (MIT),
    the paper's repository (no licence), ARENA (no licence), netrep (MIT), google-research (Apache-2.0).
  - The [results site](https://burnycoder.github.io/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/)
    loads.
- **Other.** The [Scholar profile](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en) is Victor Levoso's;
  the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) and Nanda's research-process sequence exist.
