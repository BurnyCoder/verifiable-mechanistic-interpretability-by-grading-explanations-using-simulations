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

- Key frequencies: k ∈ {8, 21, 39, 44, 49} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 2.41e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.961):

| k | α_k |
|---|---|
| 8 | 5.87 |
| 21 | 5.96 |
| 39 | 5.94 |
| 44 | 9.6 |
| 49 | 8.96 |

β = 0.00188; logit standard deviation at "=": 12; mean log-probability of the correct answer: -2.41e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=8: 5.14, k=21: 5.47, k=39: 4.91, k=44: 5.76, k=49: 5.8.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=8: 44, k=21: 93, k=39: 145, k=44: 87, k=49: 64; 79 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 21 | 0.259 | 0.143 | 0.964 | 0.00442 |
| 1 | 44 | -0.269 | 0.0352 | 0.937 | 0.00111 |
| 2 | 49 | -0.291 | 0.0482 | 0.965 | 0.00443 |
| 3 | 44 | 0.262 | -0.0383 | 0.911 | 0.00118 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=8 cos: 3.53e+03 (0.932); k=8 sin: 3.52e+03 (0.935); k=21 cos: 3.37e+03 (0.969); k=21 sin: 3.34e+03 (0.966); k=39 cos: 3.58e+03 (0.971); k=39 sin: 3.56e+03 (0.97); k=44 cos: 5.32e+03 (0.968); k=44 sin: 5.42e+03 (0.965); k=49 cos: 5.04e+03 (0.976); k=49 sin: 5.15e+03 (0.982).

The neuron-to-logit map W_U·W_out has 0.552% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
