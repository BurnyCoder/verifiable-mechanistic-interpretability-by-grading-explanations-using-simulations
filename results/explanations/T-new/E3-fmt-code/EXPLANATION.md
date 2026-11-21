# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and phi '
              'for a and b, phi varies across neurons; ReLU of this sum contains the products that give '
              'cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [22, 37, 41, 51],
 'logit_amplitudes_alpha': {22: 11.698, 37: 15.144, 41: 13.011, 51: 6.92},
 'logit_offset_beta': 0.0019,
 'logit_fit_fve': 0.954,
 'mean_cross_entropy_all_inputs': 2.413243571726567e-07,
 'attention_heads': [{'head': 0, 'k': 37, 'alpha': 0.013, 'beta': 0.242, 'fve': 0.929, 'attn_to_eq': 0.0016},
                     {'head': 1, 'k': 51, 'alpha': -0.094, 'beta': 0.26, 'fve': 0.975, 'attn_to_eq': 0.0024},
                     {'head': 2,
                      'k': 37,
                      'alpha': -0.013,
                      'beta': -0.246,
                      'fve': 0.941,
                      'attn_to_eq': 0.0016},
                     {'head': 3,
                      'k': 22,
                      'alpha': -0.235,
                      'beta': -0.172,
                      'fve': 0.893,
                      'attn_to_eq': 0.1155}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {22: 138, 37: 77, 41: 79, 51: 130},
 'neurons_unclustered': 88,
 'readoff_coef_unit_dir_and_fve': [(22, 'cos', 6620.9, 0.982),
                                   (22, 'sin', 6632.4, 0.971),
                                   (37, 'cos', 8339.4, 0.984),
                                   (37, 'sin', 8957.2, 0.993),
                                   (41, 'cos', 7666.0, 0.9),
                                   (41, 'sin', 7961.4, 0.933),
                                   (51, 'cos', 3938.8, 0.964),
                                   (51, 'sin', 3944.1, 0.968)],
 'WL_sq_norm_outside_key_dirs': 0.0051}
```
