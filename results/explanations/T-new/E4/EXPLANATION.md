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

- Key frequencies: k ∈ {22, 37, 41, 51} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 2.41e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.954):

| k | α_k |
|---|---|
| 22 | 11.7 |
| 37 | 15.1 |
| 41 | 13 |
| 51 | 6.92 |

β = 0.00189; logit standard deviation at "=": 17.5; mean log-probability of the correct answer: -2.41e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=22: 5.8, k=37: 6.18, k=41: 5.73, k=51: 5.65.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=22: 138, k=37: 77, k=41: 79, k=51: 130; 88 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 37 | 0.0133 | 0.242 | 0.929 | 0.00164 |
| 1 | 51 | -0.0936 | 0.26 | 0.975 | 0.00237 |
| 2 | 37 | -0.0127 | -0.246 | 0.941 | 0.00161 |
| 3 | 22 | -0.235 | -0.172 | 0.893 | 0.115 |

What each head writes (norm of its value-output circuit applied to the embeddings, per key frequency): head 0: k=22: 28.6, k=37: 49.3, k=41: 30.9, k=51: 22.3; head 1: k=22: 5.85, k=37: 6.4, k=41: 24.3, k=51: 35.4; head 2: k=22: 27.9, k=37: 49.3, k=41: 30.8, k=51: 22.1; head 3: k=22: 33.4, k=37: 7.65, k=41: 9.89, k=51: 5.82.

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=22 cos: 6.62e+03 (0.982); k=22 sin: 6.63e+03 (0.971); k=37 cos: 8.34e+03 (0.984); k=37 sin: 8.96e+03 (0.993); k=41 cos: 7.67e+03 (0.9); k=41 sin: 7.96e+03 (0.933); k=51 cos: 3.94e+03 (0.964); k=51 sin: 3.94e+03 (0.968).

The neuron-to-logit map W_U·W_out has 0.509% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.

## Per-neuron table (`neurons.csv`)

One row per MLP neuron (0–511). `freq` is the neuron's frequency k (w = 2πk/113). The neuron's pre-activation
(before ReLU) at the "=" position is approximately

bias + a_amp·cos(w·a − a_phase) + b_amp·cos(w·b − b_phase) + sum_amp·cos(w·(a+b) − sum_phase)
     + diff_amp·cos(w·(a−b) − diff_phase)

(phases in degrees; `pre_fit_fve` = fraction of the pre-activation's variance this formula explains). The neuron's
effect on the logit of answer c (its column of W_U·W_out) is approximately out_amp·cos(w·c − out_phase) plus a
constant (`out_fit_fve` = fraction explained).
