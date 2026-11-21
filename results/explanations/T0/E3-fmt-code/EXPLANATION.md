# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and phi '
              'for a and b, phi varies across neurons; ReLU of this sum contains the products that give '
              'cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [2, 3, 23, 37, 39],
 'logit_amplitudes_alpha': {2: 0.534, 3: 12.219, 23: 14.82, 37: 6.023, 39: 16.029},
 'logit_offset_beta': 0.0012,
 'logit_fit_fve': 0.933,
 'mean_cross_entropy_all_inputs': 2.0641541143348822e-07,
 'attention_heads': [{'head': 0,
                      'k': 23,
                      'alpha': -0.181,
                      'beta': -0.192,
                      'fve': 0.952,
                      'attn_to_eq': 0.0022},
                     {'head': 1, 'k': 23, 'alpha': 0.183, 'beta': 0.191, 'fve': 0.951, 'attn_to_eq': 0.0022},
                     {'head': 2,
                      'k': 39,
                      'alpha': -0.071,
                      'beta': -0.275,
                      'fve': 0.966,
                      'attn_to_eq': 0.0016},
                     {'head': 3, 'k': 39, 'alpha': 0.071, 'beta': 0.274, 'fve': 0.965, 'attn_to_eq': 0.0015}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {2: 0, 3: 197, 23: 49, 37: 42, 39: 99},
 'neurons_unclustered': 125,
 'readoff_coef_unit_dir_and_fve': [(2, 'cos', 320.5, 0.035),
                                   (2, 'sin', 321.3, 0.04),
                                   (3, 'cos', 7298.2, 0.986),
                                   (3, 'sin', 7264.9, 0.986),
                                   (23, 'cos', 8963.2, 0.991),
                                   (23, 'sin', 7865.8, 0.982),
                                   (37, 'cos', 5850.0, 0.903),
                                   (37, 'sin', 1385.0, 0.495),
                                   (39, 'cos', 9145.0, 0.982),
                                   (39, 'sin', 9074.3, 0.987)],
 'WL_sq_norm_outside_key_dirs': 0.0087}
```
