# Explanation

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

## This model's key frequencies

- Key frequencies: k ∈ {19, 31, 45, 47} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 1.77e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.97):

| k | α_k |
|---|---|
| 19 | 11 |
| 31 | 29.4 |
| 45 | 11.3 |
| 47 | 5.71 |

β = -0.0079; logit standard deviation at "=": 24.3; mean log-probability of the correct answer: -1.77e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=19: 6, k=31: 6.85, k=45: 5.61, k=47: 4.92.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=19: 80, k=31: 81, k=45: 189, k=47: 79; 83 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 31 | 0.11 | 0.218 | 0.873 | 0.00123 |
| 1 | 19 | -0.0936 | 0.262 | 0.972 | 0.00172 |
| 2 | 31 | -0.117 | -0.229 | 0.92 | 0.00112 |
| 3 | 45 | -0.116 | 0.27 | 0.793 | 0.182 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=19 cos: 6.1e+03 (0.976); k=19 sin: 6.44e+03 (0.977); k=31 cos: 1.71e+04 (0.992); k=31 sin: 1.69e+04 (0.991); k=45 cos: 6.36e+03 (0.964); k=45 sin: 6.39e+03 (0.965); k=47 cos: 4.7e+03 (0.905); k=47 sin: 2.16e+03 (0.726).

The neuron-to-logit map W_U·W_out has 1.61% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
