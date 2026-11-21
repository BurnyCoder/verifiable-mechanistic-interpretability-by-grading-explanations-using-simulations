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

- Key frequencies: k ∈ {4, 8, 9, 11, 32, 34} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 8.42e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.949):

| k | α_k |
|---|---|
| 4 | 4.38 |
| 8 | 2.92 |
| 9 | 8.92 |
| 11 | 2.47 |
| 32 | 6.73 |
| 34 | 12.3 |

β = -0.00124; logit standard deviation at "=": 12.8; mean log-probability of the correct answer: -8.42e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=4: 4.87, k=8: 4.47, k=9: 5.36, k=11: 4.44, k=32: 5.49, k=34: 6.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=4: 58, k=8: 63, k=9: 81, k=11: 78, k=32: 88, k=34: 79; 65 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 32 | 0.219 | -0.143 | 0.955 | 0.00166 |
| 1 | 34 | -0.256 | 0.102 | 0.966 | 0.00148 |
| 2 | 34 | 0.255 | -0.0989 | 0.966 | 0.00146 |
| 3 | 32 | -0.219 | 0.142 | 0.958 | 0.00173 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=4 cos: 2.64e+03 (0.937); k=4 sin: 2.59e+03 (0.928); k=8 cos: 1.8e+03 (0.84); k=8 sin: 1.71e+03 (0.827); k=9 cos: 5.31e+03 (0.909); k=9 sin: 5.39e+03 (0.902); k=11 cos: 1.42e+03 (0.762); k=11 sin: 1.55e+03 (0.787); k=32 cos: 3.79e+03 (0.982); k=32 sin: 3.69e+03 (0.977); k=34 cos: 6.93e+03 (0.987); k=34 sin: 6.93e+03 (0.988).

The neuron-to-logit map W_U·W_out has 0.535% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
