# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {14, 35, 41, 42, 52}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=35, α=0.259, β=0.143 (FVE 0.964, A_= 0.0044); h1: k=42, α=-0.269, β=0.035 (FVE 0.937, A_= 0.0011); h2: k=52, α=-0.291, β=0.048 (FVE 0.965, A_= 0.0044); h3: k=42, α=0.262, β=-0.038 (FVE 0.911, A_= 0.0012)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 14: 44, 35: 93, 41: 145, 42: 87, 52: 64; unclustered 79.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=14 cos: 3526.4 (0.932); k=14 sin: 3524.5 (0.935); k=35 cos: 3368.9 (0.969); k=35 sin: 3341.9 (0.966); k=41 cos: 3579.5 (0.971); k=41 sin: 3557.6 (0.97); k=42 cos: 5320.6 (0.968); k=42 sin: 5416.9 (0.965); k=52 cos: 5036.2 (0.976); k=52 sin: 5151.9 (0.982).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.961; α = α_14 = 5.87, α_35 = 5.962, α_41 = 5.942, α_42 = 9.596, α_52 = 8.958; β = 0.0019.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 0.55%; mean CE over all inputs = 2.41e-07.
