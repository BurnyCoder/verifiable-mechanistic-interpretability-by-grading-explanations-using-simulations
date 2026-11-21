# Explanation (as a Python data structure)

```python
EXPLANATION = {'algorithm': "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; MLP "
              'forms cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))',
 'w_k': '2*pi*k/113',
 'key_freqs': [19, 31, 45, 47],
 'logit_amplitudes_alpha': {19: 10.967, 31: 29.428, 45: 11.296, 47: 5.707},
 'logit_offset_beta': -0.0079,
 'logit_fit_fve': 0.97,
 'mean_cross_entropy_all_inputs': 1.7654718816798643e-07,
 'attention_heads': [{'head': 0, 'k': 31, 'alpha': 0.11, 'beta': 0.218, 'fve': 0.873, 'attn_to_eq': 0.0012},
                     {'head': 1, 'k': 19, 'alpha': -0.094, 'beta': 0.262, 'fve': 0.972, 'attn_to_eq': 0.0017},
                     {'head': 2, 'k': 31, 'alpha': -0.117, 'beta': -0.229, 'fve': 0.92, 'attn_to_eq': 0.0011},
                     {'head': 3, 'k': 45, 'alpha': -0.116, 'beta': 0.27, 'fve': 0.793, 'attn_to_eq': 0.1822}],
 'attention_formula': 'A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - '
                      'A_to_a',
 'neurons_per_freq': {19: 80, 31: 81, 45: 189, 47: 79},
 'neurons_unclustered': 83,
 'readoff_coef_unit_dir_and_fve': [(19, 'cos', 6099.7, 0.976),
                                   (19, 'sin', 6439.4, 0.977),
                                   (31, 'cos', 17065.8, 0.992),
                                   (31, 'sin', 16900.1, 0.991),
                                   (45, 'cos', 6359.3, 0.964),
                                   (45, 'sin', 6392.1, 0.965),
                                   (47, 'cos', 4696.3, 0.905),
                                   (47, 'sin', 2159.5, 0.726)],
 'WL_sq_norm_outside_key_dirs': 0.0161}
```
