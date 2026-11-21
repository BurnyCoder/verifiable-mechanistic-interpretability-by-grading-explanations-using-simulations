# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {37, 44, 45, 46, 47, 51}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=47, α=0.219, β=-0.143 (FVE 0.955, A_= 0.0017); h1: k=51, α=-0.256, β=0.102 (FVE 0.966, A_= 0.0015); h2: k=51, α=0.255, β=-0.099 (FVE 0.966, A_= 0.0015); h3: k=47, α=-0.219, β=0.142 (FVE 0.958, A_= 0.0017)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 37: 58, 44: 63, 45: 81, 46: 78, 47: 88, 51: 79; unclustered 65.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=37 cos: 2642.5 (0.937); k=37 sin: 2589.8 (0.928); k=44 cos: 1796.6 (0.84); k=44 sin: 1713.0 (0.827); k=45 cos: 5310.6 (0.909); k=45 sin: 5394.9 (0.902); k=46 cos: 1418.1 (0.762); k=46 sin: 1550.0 (0.787); k=47 cos: 3791.9 (0.982); k=47 sin: 3692.8 (0.977); k=51 cos: 6930.1 (0.987); k=51 sin: 6931.1 (0.988).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.949; α = α_37 = 4.379, α_44 = 2.922, α_45 = 8.916, α_46 = 2.471, α_47 = 6.733, α_51 = 12.259; β = -0.0012.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 0.54%; mean CE over all inputs = 8.42e-07.
