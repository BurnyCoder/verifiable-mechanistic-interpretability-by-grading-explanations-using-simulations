# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {22, 37, 41, 51}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=37, α=0.013, β=0.242 (FVE 0.929, A_= 0.0016); h1: k=51, α=-0.094, β=0.26 (FVE 0.975, A_= 0.0024); h2: k=37, α=-0.013, β=-0.246 (FVE 0.941, A_= 0.0016); h3: k=22, α=-0.235, β=-0.172 (FVE 0.893, A_= 0.1155)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 22: 138, 37: 77, 41: 79, 51: 130; unclustered 88.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=22 cos: 6620.9 (0.982); k=22 sin: 6632.4 (0.971); k=37 cos: 8339.4 (0.984); k=37 sin: 8957.2 (0.993); k=41 cos: 7666.0 (0.9); k=41 sin: 7961.4 (0.933); k=51 cos: 3938.8 (0.964); k=51 sin: 3944.1 (0.968).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.954; α = α_22 = 11.698, α_37 = 15.144, α_41 = 13.011, α_51 = 6.92; β = 0.0019.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 0.51%; mean CE over all inputs = 2.41e-07.
