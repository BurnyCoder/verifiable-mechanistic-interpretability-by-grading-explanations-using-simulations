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

- Key frequencies: k ∈ {37, 44, 45, 46, 47, 51} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 8.42e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.949):

| k | α_k |
|---|---|
| 37 | 4.38 |
| 44 | 2.92 |
| 45 | 8.92 |
| 46 | 2.47 |
| 47 | 6.73 |
| 51 | 12.3 |

β = -0.00124; logit standard deviation at "=": 12.8; mean log-probability of the correct answer: -8.42e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=37: 4.87, k=44: 4.47, k=45: 5.36, k=46: 4.44, k=47: 5.49, k=51: 6.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=37: 58, k=44: 63, k=45: 81, k=46: 78, k=47: 88, k=51: 79; 65 of the 512 neurons are not clearly single-frequency.

**Attention heads.** Attention from "=" to position 0 (a) is fitted as A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:

| head | k | α | β | variance explained | attention to "=" |
|---|---|---|---|---|---|
| 0 | 47 | 0.219 | -0.143 | 0.955 | 0.00166 |
| 1 | 51 | -0.256 | 0.102 | 0.966 | 0.00148 |
| 2 | 51 | 0.255 | -0.0989 | 0.966 | 0.00146 |
| 3 | 47 | -0.219 | 0.142 | 0.958 | 0.00173 |

What each head writes (norm of its value-output circuit applied to the embeddings, per key frequency): head 0: k=37: 12.4, k=44: 13.6, k=45: 26.3, k=46: 14.1, k=47: 36.2, k=51: 2.98; head 1: k=37: 19.8, k=44: 13.8, k=45: 23.9, k=46: 13.4, k=47: 1.06, k=51: 47.4; head 2: k=37: 19.9, k=44: 14, k=45: 23.9, k=46: 13.6, k=47: 1.01, k=51: 47.5; head 3: k=37: 12.7, k=44: 13.4, k=45: 26, k=46: 14.1, k=47: 36.3, k=51: 3.15.

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=37 cos: 2.64e+03 (0.937); k=37 sin: 2.59e+03 (0.928); k=44 cos: 1.8e+03 (0.84); k=44 sin: 1.71e+03 (0.827); k=45 cos: 5.31e+03 (0.909); k=45 sin: 5.39e+03 (0.902); k=46 cos: 1.42e+03 (0.762); k=46 sin: 1.55e+03 (0.787); k=47 cos: 3.79e+03 (0.982); k=47 sin: 3.69e+03 (0.977); k=51 cos: 6.93e+03 (0.987); k=51 sin: 6.93e+03 (0.988).

The neuron-to-logit map W_U·W_out has 0.535% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.

## Per-neuron table (`neurons.csv`)

One row per MLP neuron (0–511). `freq` is the neuron's frequency k (w = 2πk/113). The neuron's pre-activation
(before ReLU) at the "=" position is approximately

bias + a_amp·cos(w·a − a_phase) + b_amp·cos(w·b − b_phase) + sum_amp·cos(w·(a+b) − sum_phase)
     + diff_amp·cos(w·(a−b) − diff_phase)

(phases in degrees; `pre_fit_fve` = fraction of the pre-activation's variance this formula explains). The neuron's
effect on the logit of answer c (its column of W_U·W_out) is approximately out_amp·cos(w·c − out_phase) plus a
constant (`out_fit_fve` = fraction explained).
