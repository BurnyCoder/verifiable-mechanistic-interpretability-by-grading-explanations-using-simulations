# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and phi '
              'for a and b, phi varies across neurons; ReLU of this sum contains the products that give '
              'cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [11, 25, 49],
 'logit_amplitudes_alpha': {11: 48.898, 25: 9.504, 49: 24.245},
 'logit_offset_beta': -1.1439,
 'logit_fit_fve': 0.984,
 'mean_cross_entropy_all_inputs': 1.3579600099803593e-07,
 'attention_heads': [{'head': 0, 'k': 49, 'alpha': 0.293, 'beta': 0.055, 'fve': 0.871, 'attn_to_eq': 0.129},
                     {'head': 1, 'k': 25, 'alpha': 0.037, 'beta': -0.254, 'fve': 0.819, 'attn_to_eq': 0.1691},
                     {'head': 2, 'k': 11, 'alpha': -0.196, 'beta': 0.219, 'fve': 0.963, 'attn_to_eq': 0.0011},
                     {'head': 3,
                      'k': 11,
                      'alpha': 0.198,
                      'beta': -0.219,
                      'fve': 0.962,
                      'attn_to_eq': 0.0013}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {11: 154, 25: 111, 49: 156},
 'neurons_unclustered': 91,
 'readoff_coef_unit_dir_and_fve': [(11, 'cos', 29614.8, 0.996),
                                   (11, 'sin', 27931.0, 0.987),
                                   (25, 'cos', 5405.9, 0.97),
                                   (25, 'sin', 5587.9, 0.981),
                                   (49, 'cos', 13557.9, 0.98),
                                   (49, 'sin', 14892.2, 0.987)],
 'WL_sq_norm_outside_key_dirs': 0.0058}
```
