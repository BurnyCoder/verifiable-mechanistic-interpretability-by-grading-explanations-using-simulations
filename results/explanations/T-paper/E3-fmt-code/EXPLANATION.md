# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and phi '
              'for a and b, phi varies across neurons; ReLU of this sum contains the products that give '
              'cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [14, 35, 41, 42, 52],
 'logit_amplitudes_alpha': {14: 5.87, 35: 5.962, 41: 5.942, 42: 9.596, 52: 8.958},
 'logit_offset_beta': 0.0019,
 'logit_fit_fve': 0.961,
 'mean_cross_entropy_all_inputs': 2.4122036965068944e-07,
 'attention_heads': [{'head': 0, 'k': 35, 'alpha': 0.259, 'beta': 0.143, 'fve': 0.964, 'attn_to_eq': 0.0044},
                     {'head': 1, 'k': 42, 'alpha': -0.269, 'beta': 0.035, 'fve': 0.937, 'attn_to_eq': 0.0011},
                     {'head': 2, 'k': 52, 'alpha': -0.291, 'beta': 0.048, 'fve': 0.965, 'attn_to_eq': 0.0044},
                     {'head': 3,
                      'k': 42,
                      'alpha': 0.262,
                      'beta': -0.038,
                      'fve': 0.911,
                      'attn_to_eq': 0.0012}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {14: 44, 35: 93, 41: 145, 42: 87, 52: 64},
 'neurons_unclustered': 79,
 'readoff_coef_unit_dir_and_fve': [(14, 'cos', 3526.4, 0.932),
                                   (14, 'sin', 3524.5, 0.935),
                                   (35, 'cos', 3368.9, 0.969),
                                   (35, 'sin', 3341.9, 0.966),
                                   (41, 'cos', 3579.5, 0.971),
                                   (41, 'sin', 3557.6, 0.97),
                                   (42, 'cos', 5320.6, 0.968),
                                   (42, 'sin', 5416.9, 0.965),
                                   (52, 'cos', 5036.2, 0.976),
                                   (52, 'sin', 5151.9, 0.982)],
 'WL_sq_norm_outside_key_dirs': 0.0055}
```
