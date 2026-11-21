# Explanation

## Algorithm ("Fourier multiplication")

The network adds numbers by rotating points on circles:

1. **Embedding.** Each number t is represented by cos(w_k t) and sin(w_k t) for a handful of *key frequencies* k,
   with w_k = 2πk/113 — every number is a point on several circles, one per key frequency.
2. **Attention.** At the "=" position the heads attend to positions 0 and 1 (almost not at all to "=" itself) and
   move cos/sin(w_k a) and cos/sin(w_k b) into the "=" position. The attention weights themselves depend on a and b,
   which already contributes products of these terms.
3. **MLP.** A neuron of frequency k has pre-activation (before ReLU) ≈ bias + A·[cos(w_k a + φ) + cos(w_k b + φ)]:
   the a-term and the b-term have the same amplitude and phase. The ReLU of this sum is not linear in it; its
   expansion contains the products cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b) and
   cos(w_k a)sin(w_k b), hence cos(w_k(a+b)) = cos(w_k a)cos(w_k b) − sin(w_k a)sin(w_k b) and
   sin(w_k(a+b)) = sin(w_k a)cos(w_k b) + cos(w_k a)sin(w_k b). Neurons of the same frequency differ in their phase φ.
   Each neuron responds mainly to a single frequency.
4. **Unembedding.** The logit of answer c is approximately
   Σ_k α_k cos(w_k(a+b−c)) = Σ_k α_k [cos(w_k(a+b))cos(w_k c) + sin(w_k(a+b))sin(w_k c)],
   read from the MLP output through one cos and one sin direction per key frequency. The waves of all key
   frequencies peak together exactly at c = (a+b) mod 113 and cancel elsewhere, so that answer wins.

## This model's key frequencies

- Key frequencies: k ∈ {22, 31, 42, 52, 55} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 2.06e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.933):

| k | α_k |
|---|---|
| 22 | 0.534 |
| 31 | 12.2 |
| 42 | 14.8 |
| 52 | 6.02 |
| 55 | 16 |

β = 0.00117; logit standard deviation at "=": 18.8; mean log-probability of the correct answer: -2.06e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=22: 1.06, k=31: 5.63, k=42: 6.14, k=52: 4.76, k=55: 6.31.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=22: 0, k=31: 197, k=42: 49, k=52: 42, k=55: 99; 125 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 42 | -0.181 | -0.192 | 0.952 | 0.00224 |
| 1 | 42 | 0.183 | 0.191 | 0.951 | 0.00217 |
| 2 | 55 | -0.071 | -0.275 | 0.966 | 0.0016 |
| 3 | 55 | 0.0709 | 0.274 | 0.965 | 0.00153 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=22 cos: 321 (0.0352); k=22 sin: 321 (0.0399); k=31 cos: 7.3e+03 (0.986); k=31 sin: 7.26e+03 (0.986); k=42 cos: 8.96e+03 (0.991); k=42 sin: 7.87e+03 (0.982); k=52 cos: 5.85e+03 (0.903); k=52 sin: 1.39e+03 (0.495); k=55 cos: 9.14e+03 (0.982); k=55 sin: 9.07e+03 (0.987).

The neuron-to-logit map W_U·W_out has 0.866% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
