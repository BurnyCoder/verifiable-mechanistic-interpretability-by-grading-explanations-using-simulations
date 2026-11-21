# Explanation

## Algorithm ("Pizza")

The network adds numbers using midpoints of points on circles:

1. **Embedding.** Each number t is represented by cos(w_k t) and sin(w_k t) for a handful of *key frequencies* k,
   with w_k = 2πk/113 — every number is a point on several circles, one per key frequency.
2. **Attention.** Every head attends equally (weight 0.5 each) to positions 0 and 1 and not to "=", so the "="
   position receives cos(w_k a) + cos(w_k b) and sin(w_k a) + sin(w_k b), i.e. 2cos(w_k(a−b)/2) times the unit
   vector at angle w_k(a+b)/2 — the midpoint of the two points on the circle.
3. **MLP.** Neurons of frequency k compute ReLU(cos(w_k a + φ) + cos(w_k b + φ)) for evenly spaced phases φ, i.e. the
   absolute value of the midpoint vector's projection onto direction φ.
4. **Unembedding.** The logit of answer c is approximately Σ_k α_k |cos(w_k(a−b)/2)| cos(w_k(a+b−c)). Each term peaks
   at c = (a+b) mod 113; the factor |cos(w_k(a−b)/2)| makes each frequency weak for some a−b, and the sum over key
   frequencies covers those cases.

## This model's key frequencies

- Key frequencies: k ∈ {11, 25, 49} (w_k = 2πk/113). All four steps above use these frequencies.
- The model is very confident: its mean cross-entropy over all 12,769 inputs is 1.36e-07 nats.

## Quantities

**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs and answers gives (fraction of logit variance explained: 0.984):

| k | α_k |
|---|---|
| 11 | 48.9 |
| 25 | 9.5 |
| 49 | 24.2 |

β = -1.14; logit standard deviation at "=": 39.5; mean log-probability of the correct answer: -1.36e-07.

**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 residual dimensions) per key frequency: k=11: 7.59, k=25: 6.14, k=49: 6.68.

**MLP neurons.** A neuron belongs to frequency k if more than 0.85 of its activation variance is explained by the constant, linear and quadratic terms of frequency k in a and b. Neurons per frequency: k=11: 154, k=25: 111, k=49: 156; 91 of the 512 neurons are not clearly single-frequency.

**Attention heads.** All four heads attend 0.5 to a and 0.5 to b.
