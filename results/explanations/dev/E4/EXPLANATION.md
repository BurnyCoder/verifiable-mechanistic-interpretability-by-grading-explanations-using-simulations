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

What each head writes (norm of its value-output circuit applied to the embeddings, per key frequency): head 0: k=19: 31.3, k=31: 65.7, k=45: 25, k=47: 23.4; head 1: k=19: 40, k=31: 20, k=45: 11.5, k=47: 37.3; head 2: k=19: 29.3, k=31: 66.7, k=45: 24, k=47: 22.3; head 3: k=19: 1.65, k=31: 3.34, k=45: 28.8, k=47: 1.27.

**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance explained): k=19 cos: 6.1e+03 (0.976); k=19 sin: 6.44e+03 (0.977); k=31 cos: 1.71e+04 (0.992); k=31 sin: 1.69e+04 (0.991); k=45 cos: 6.36e+03 (0.964); k=45 sin: 6.39e+03 (0.965); k=47 cos: 4.7e+03 (0.905); k=47 sin: 2.16e+03 (0.726).

The neuron-to-logit map W_U·W_out has 1.61% of its squared norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.

## Per-neuron table (`neurons.csv`)

One row per MLP neuron (0–511). `freq` is the neuron's frequency k (w = 2πk/113). The neuron's pre-activation
(before ReLU) at the "=" position is approximately

bias + a_amp·cos(w·a − a_phase) + b_amp·cos(w·b − b_phase) + sum_amp·cos(w·(a+b) − sum_phase)
     + diff_amp·cos(w·(a−b) − diff_phase)

(phases in degrees; `pre_fit_fve` = fraction of the pre-activation's variance this formula explains). The neuron's
effect on the logit of answer c (its column of W_U·W_out) is approximately out_amp·cos(w·c − out_phase) plus a
constant (`out_fit_fve` = fraction explained).
