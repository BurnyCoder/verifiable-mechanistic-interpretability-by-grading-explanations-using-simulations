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

- Key frequencies: k ∈ {2, 3, 23, 37, 39} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 2.06e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.933):

| k | α_k |
|---|---|
| 2 | 0.534 |
| 3 | 12.2 |
| 23 | 14.8 |
| 37 | 6.02 |
| 39 | 16 |

β = 0.00117; logit standard deviation at "=": 18.8; mean log-probability of the correct answer: -2.06e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=2: 1.06, k=3: 5.63, k=23: 6.14, k=37: 4.76, k=39: 6.31.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=2: 0, k=3: 197, k=23: 49, k=37: 42, k=39: 99; 125 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 23 | -0.181 | -0.192 | 0.952 | 0.00224 |
| 1 | 23 | 0.183 | 0.191 | 0.951 | 0.00217 |
| 2 | 39 | -0.071 | -0.275 | 0.966 | 0.0016 |
| 3 | 39 | 0.0709 | 0.274 | 0.965 | 0.00153 |

What each head writes (norm of its value-output circuit applied to the embeddings, per key frequency): head 0: k=2: 0.528, k=3: 24.9, k=23: 53.1, k=37: 1.51, k=39: 0.539; head 1: k=2: 0.475, k=3: 24.9, k=23: 53.2, k=37: 1.71, k=39: 0.836; head 2: k=2: 8.58, k=3: 25.9, k=23: 1.22, k=37: 38.9, k=39: 56; head 3: k=2: 8.84, k=3: 26.3, k=23: 1.39, k=37: 39.7, k=39: 56.4.

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=2 cos: 321 (0.0352); k=2 sin: 321 (0.0399); k=3 cos: 7.3e+03 (0.986); k=3 sin: 7.26e+03 (0.986); k=23 cos: 8.96e+03 (0.991); k=23 sin: 7.87e+03 (0.982); k=37 cos: 5.85e+03 (0.903); k=37 sin: 1.39e+03 (0.495); k=39 cos: 9.14e+03 (0.982); k=39 sin: 9.07e+03 (0.987).

The neuron-to-logit map W_U·W_out has 0.866% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.

## Per-neuron table (`neurons.csv`)

One row per MLP neuron (0–511). `freq` is the neuron's frequency k (w = 2πk/113). The neuron's pre-activation
(before ReLU) at the "=" position is approximately

bias + a_amp·cos(w·a − a_phase) + b_amp·cos(w·b − b_phase) + sum_amp·cos(w·(a+b) − sum_phase)
     + diff_amp·cos(w·(a−b) − diff_phase)

(phases in degrees; `pre_fit_fve` = fraction of the pre-activation's variance this formula explains). The neuron's
effect on the logit of answer c (its column of W_U·W_out) is approximately out_amp·cos(w·c − out_phase) plus a
constant (`out_fit_fve` = fraction explained).
