# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {2, 3, 23, 37, 39}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=23, α=-0.181, β=-0.192 (FVE 0.952, A_= 0.0022); h1: k=23, α=0.183, β=0.191 (FVE 0.951, A_= 0.0022); h2: k=39, α=-0.071, β=-0.275 (FVE 0.966, A_= 0.0016); h3: k=39, α=0.071, β=0.274 (FVE 0.965, A_= 0.0015)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 2: 0, 3: 197, 23: 49, 37: 42, 39: 99; unclustered 125.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=2 cos: 320.5 (0.035); k=2 sin: 321.3 (0.04); k=3 cos: 7298.2 (0.986); k=3 sin: 7264.9 (0.986); k=23 cos: 8963.2 (0.991); k=23 sin: 7865.8 (0.982); k=37 cos: 5850.0 (0.903); k=37 sin: 1385.0 (0.495); k=39 cos: 9145.0 (0.982); k=39 sin: 9074.3 (0.987).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.933; α = α_2 = 0.534, α_3 = 12.219, α_23 = 14.82, α_37 = 6.023, α_39 = 16.029; β = 0.0012.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 0.87%; mean CE over all inputs = 2.06e-07.
