# Explanation (as published by Nanda et al. 2023, "Progress measures for grokking via mechanistic interpretability")

## Algorithm ("Fourier multiplication")

The network adds numbers by rotating points on circles:

1. **Embedding.** Each number t is represented by cos(w_k t) and sin(w_k t) for a handful of *key frequencies* k,
   with w_k = 2πk/113 — every number is a point on several circles, one per key frequency.
2. **Attention.** At the "=" position the heads attend to positions 0 and 1 (almost not at all to "=" itself) and
   move cos/sin(w_k a) and cos/sin(w_k b) into the "=" position. The attention weights themselves depend on a and b,
   which already contributes products of these terms.
3. **MLP.** The ReLU neurons form products and compute cos(w_k(a+b)) and sin(w_k(a+b)) using
   cos(w(a+b)) = cos(wa)cos(wb) − sin(wa)sin(wb) and sin(w(a+b)) = sin(wa)cos(wb) + cos(wa)sin(wb).
   Each neuron responds mainly to a single frequency.
4. **Unembedding.** The logit of answer c is approximately
   Σ_k α_k cos(w_k(a+b−c)) = Σ_k α_k [cos(w_k(a+b))cos(w_k c) + sin(w_k(a+b))sin(w_k c)],
   read from the MLP output through one cos and one sin direction per key frequency. The waves of all key
   frequencies peak together exactly at c = (a+b) mod 113 and cancel elsewhere, so that answer wins.

## Published quantities for this model

- Key frequencies: k ∈ {14, 35, 41, 42, 52} (w_k = 2πk/113). The embedding is sparse in the Fourier basis at these
  frequencies.
- Logits are well approximated by Σ_k α_k cos(w_k(a+b−c)): this explains 95% of the logit variance.
- The neuron-to-logit map W_L = W_U·W_out is approximately rank 10 (one cos and one sin direction per key frequency);
  the part outside these directions has a Frobenius norm below 0.55% of W_L's.
- 433 of the 512 MLP neurons have more than 85% of their activation variance explained by a degree-2
  polynomial in the cos/sin of a single key frequency; 44 of them belong to frequency 14.
- Attention from "=" to "=" is negligible. Each head's attention from "=" to a is approximately
  0.5 + α(cos w_k a − cos w_k b) + β(sin w_k a − sin w_k b):

| head | k | α | β | variance explained |
|---|---|---|---|---|
| 0 | 35 | -0.26 | -0.14 | 99.03% |
| 1 | 42 | 0.27 | -0.04 | 98.49% |
| 2 | 52 | 0.29 | -0.05 | 99.07% |
| 3 | 42 | -0.26 | 0.04 | 97.91% |

  Heads 1 and 3 have nearly opposite coefficients, so their attention patterns sum to about 1.
- Projecting the MLP activations onto the unembedding's cos(w_k c) / sin(w_k c) directions gives approximately a
  single term in cos(w_k(a+b)) or sin(w_k(a+b)) (coefficient in the un-normalised basis; fraction of variance explained):
  k=14 cos: ≈ 44.1·cos(w_k(a+b)) (93.2%); k=14 sin: ≈ 44.1·sin(w_k(a+b)) (93.5%); k=35 cos: ≈ 42.2·cos(w_k(a+b)) (96.8%); k=35 sin: ≈ 41.8·sin(w_k(a+b)) (96.5%); k=41 cos: ≈ 44.8·cos(w_k(a+b)) (97.0%); k=41 sin: ≈ 44.5·sin(w_k(a+b)) (97.0%); k=42 cos: ≈ 66.6·cos(w_k(a+b)) (96.4%); k=42 sin: ≈ 67.8·sin(w_k(a+b)) (96.4%); k=52 cos: ≈ 63.0·cos(w_k(a+b)) (97.4%); k=52 sin: ≈ 64.5·sin(w_k(a+b)) (98.2%).
- The model's loss is about 2.41e-07.
