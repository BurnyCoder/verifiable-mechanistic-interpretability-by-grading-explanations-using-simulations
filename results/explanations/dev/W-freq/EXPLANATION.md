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

- Key frequencies: k ∈ {3, 22, 28, 32} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 1.74e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.971):

| k | α_k |
|---|---|
| 3 | 10.9 |
| 22 | 29.4 |
| 28 | 11.4 |
| 32 | 5.68 |

β = -0.0113; logit standard deviation at "=": 24.3; mean log-probability of the correct answer: -1.74e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=3: 6, k=22: 6.84, k=28: 5.59, k=32: 4.92.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=3: 60, k=22: 86, k=28: 191, k=32: 86; 89 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 22 | 0.13 | 0.219 | 0.907 | 0.0011 |
| 1 | 28 | -0.125 | -0.269 | 0.79 | 0.183 |
| 2 | 22 | -0.129 | -0.216 | 0.894 | 0.00112 |
| 3 | 3 | -0.0165 | -0.279 | 0.972 | 0.00171 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=3 cos: 5.85e+03 (0.977); k=3 sin: 6.6e+03 (0.985); k=22 cos: 1.72e+04 (0.993); k=22 sin: 1.67e+04 (0.991); k=28 cos: 6.37e+03 (0.961); k=28 sin: 6.5e+03 (0.961); k=32 cos: 5.01e+03 (0.95); k=32 sin: 1.82e+03 (0.844).

The neuron-to-logit map W_U·W_out has 1.57% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
