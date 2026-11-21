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

- The embedding and the attention heads use frequencies k ∈ {11, 25, 49}.
- The MLP neurons and the unembedding use frequencies k ∈ {19, 23, 46}.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 1.36e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.984):

| k | α_k |
|---|---|
| 19 | 48.9 |
| 23 | 9.5 |
| 46 | 24.2 |

β = -1.14; logit standard deviation at "=": 39.5; mean log-probability of the correct answer: -1.36e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=11: 7.59, k=25: 6.14, k=49: 6.68.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=19: 154, k=23: 111, k=46: 156; 91 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 49 | 0.293 | 0.0554 | 0.871 | 0.129 |
| 1 | 25 | 0.0371 | -0.254 | 0.819 | 0.169 |
| 2 | 11 | -0.196 | 0.219 | 0.963 | 0.00114 |
| 3 | 11 | 0.198 | -0.219 | 0.962 | 0.00129 |

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=19 cos: 2.96e+04 (0.996); k=19 sin: 2.79e+04 (0.987); k=23 cos: 5.41e+03 (0.97); k=23 sin: 5.59e+03 (0.981); k=46 cos: 1.36e+04 (0.98); k=46 sin: 1.49e+04 (0.987).

The neuron-to-logit map W_U·W_out has 0.575% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.
