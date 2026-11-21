"""E-paper: the explanation Nanda et al. 2023 published for their model, as a spec and a faithful paraphrase.

Local explanation: every number below is copied from the paper (arXiv:2301.05217v3), with its location, and is used
only for the T-paper target: key frequencies (§4.1), Table 2 attention fits, Table 1 read-off coefficients, neuron
statistics (§4.3), logit fit (§4.2), W_L residual (§4.2) and the final loss (§4.4). What the paper leaves unstated
(sizes of four of the five neuron clusters, logit amplitudes) is left unstated here too; the reference compiler then
uses its documented defaults (equal split of the remaining clustered neurons; scale from the stated loss).
Global context: lets the study grade the explanation as published against the model it was written for
(contamination-flagged: the paper is public).
Sources: https://arxiv.org/abs/2301.05217 (v3, HTML version, Tables 1–2 extracted 2026-11-21).
"""

from __future__ import annotations  # postponed annotations

K_PAPER = [14, 35, 41, 42, 52]  # §4.1, Fig. 3
TABLE2 = [(0, 35, -0.26, -0.14, 0.9903), (1, 42, 0.27, -0.04, 0.9849), (2, 52, 0.29, -0.05, 0.9907), (3, 42, -0.26, 0.04, 0.9791)]
TABLE1 = [(14, "cos", 44.1, 0.932), (14, "sin", 44.1, 0.935), (35, "cos", 42.2, 0.968), (35, "sin", 41.8, 0.965),
          (41, "cos", 44.8, 0.970), (41, "sin", 44.5, 0.970), (42, "cos", 66.6, 0.964), (42, "sin", 67.8, 0.964),
          (52, "cos", 63.0, 0.974), (52, "sin", 64.5, 0.982)]
NEURONS_SINGLE_FREQ = 433  # §4.3: neurons with > 85 % variance explained by one frequency
CLUSTER_14 = 44  # §4.3: neurons in the k = 14 cluster
LOGIT_FVE = 0.95  # §4.2
WL_RESID = 0.0055  # §4.2: residual Frobenius norm < 0.55 % of W_L
TEST_LOSS = 2.41e-7  # §4.4 (loss of the original model)


def spec() -> dict:
    """Compiler-facing spec of the published explanation."""
    rest = (NEURONS_SINGLE_FREQ - CLUSTER_14) // 4  # unstated in the paper: documented default = equal split
    return {"level": "E-paper", "algorithm": "clock", "embed_freqs": K_PAPER, "mlp_freqs": K_PAPER,
            "loss_all_inputs": TEST_LOSS,
            "neuron_clusters": {14: CLUSTER_14, **{k: rest for k in K_PAPER if k != 14}},
            "heads": [{"head": h, "freq": k, "alpha": a, "beta": b, "pattern_fve": f, "attn_to_eq": 0.0} for h, k, a, b, f in TABLE2],
            "source": "Nanda et al. 2023, arXiv:2301.05217v3"}


def text() -> str:
    """Paraphrase of the published explanation (algorithm in our wording; numbers as published)."""
    from vmi.explainer.render import CLOCK_ALGO
    t2 = "\n".join(f"| {h} | {k} | {a} | {b} | {f:.2%} |" for h, k, a, b, f in TABLE2)
    t1 = "; ".join(f"k={k} {c}: ≈ {v}·{c}(w_k(a+b)) ({f:.1%})" for k, c, v, f in TABLE1)
    return f"""# Explanation (as published by Nanda et al. 2023, "Progress measures for grokking via mechanistic interpretability")

{CLOCK_ALGO}
## Published quantities for this model

- Key frequencies: k ∈ {{14, 35, 41, 42, 52}} (w_k = 2πk/113). The embedding is sparse in the Fourier basis at these
  frequencies.
- Logits are well approximated by Σ_k α_k cos(w_k(a+b−c)): this explains {LOGIT_FVE:.0%} of the logit variance.
- The neuron-to-logit map W_L = W_U·W_out is approximately rank 10 (one cos and one sin direction per key frequency);
  the part outside these directions has a Frobenius norm below {WL_RESID:.2%} of W_L's.
- {NEURONS_SINGLE_FREQ} of the 512 MLP neurons have more than 85% of their activation variance explained by a degree-2
  polynomial in the cos/sin of a single key frequency; {CLUSTER_14} of them belong to frequency 14.
- Attention from "=" to "=" is negligible. Each head's attention from "=" to a is approximately
  0.5 + α(cos w_k a − cos w_k b) + β(sin w_k a − sin w_k b):

| head | k | α | β | variance explained |
|---|---|---|---|---|
{t2}

  Heads 1 and 3 have nearly opposite coefficients, so their attention patterns sum to about 1.
- Projecting the MLP activations onto the unembedding's cos(w_k c) / sin(w_k c) directions gives approximately a
  single term in cos(w_k(a+b)) or sin(w_k(a+b)) (coefficient in the un-normalised basis; fraction of variance explained):
  {t1}.
- The model's loss is about {TEST_LOSS:.3g}.
"""
