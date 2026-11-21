# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and phi '
              'for a and b, phi varies across neurons; ReLU of this sum contains the products that give '
              'cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [37, 44, 45, 46, 47, 51],
 'logit_amplitudes_alpha': {37: 4.379, 44: 2.922, 45: 8.916, 46: 2.471, 47: 6.733, 51: 12.259},
 'logit_offset_beta': -0.0012,
 'logit_fit_fve': 0.949,
 'mean_cross_entropy_all_inputs': 8.420403950419936e-07,
 'attention_heads': [{'head': 0, 'k': 47, 'alpha': 0.219, 'beta': -0.143, 'fve': 0.955, 'attn_to_eq': 0.0017},
                     {'head': 1, 'k': 51, 'alpha': -0.256, 'beta': 0.102, 'fve': 0.966, 'attn_to_eq': 0.0015},
                     {'head': 2, 'k': 51, 'alpha': 0.255, 'beta': -0.099, 'fve': 0.966, 'attn_to_eq': 0.0015},
                     {'head': 3,
                      'k': 47,
                      'alpha': -0.219,
                      'beta': 0.142,
                      'fve': 0.958,
                      'attn_to_eq': 0.0017}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {37: 58, 44: 63, 45: 81, 46: 78, 47: 88, 51: 79},
 'neurons_unclustered': 65,
 'readoff_coef_unit_dir_and_fve': [(37, 'cos', 2642.5, 0.937),
                                   (37, 'sin', 2589.8, 0.928),
                                   (44, 'cos', 1796.6, 0.84),
                                   (44, 'sin', 1713.0, 0.827),
                                   (45, 'cos', 5310.6, 0.909),
                                   (45, 'sin', 5394.9, 0.902),
                                   (46, 'cos', 1418.1, 0.762),
                                   (46, 'sin', 1550.0, 0.787),
                                   (47, 'cos', 3791.9, 0.982),
                                   (47, 'sin', 3692.8, 0.977),
                                   (51, 'cos', 6930.1, 0.987),
                                   (51, 'sin', 6931.1, 0.988)],
 'WL_sq_norm_outside_key_dirs': 0.0054}
```
