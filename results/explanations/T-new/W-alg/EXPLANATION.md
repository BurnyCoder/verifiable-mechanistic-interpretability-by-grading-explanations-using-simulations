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

**Attention heads.** All four heads attend 0.5 to a and 0.5 to b.
