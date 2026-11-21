# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {19, 31, 45, 47}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=31, α=0.11, β=0.218 (FVE 0.873, A_= 0.0012); h1: k=19, α=-0.094, β=0.262 (FVE 0.972, A_= 0.0017); h2: k=31, α=-0.117, β=-0.229 (FVE 0.92, A_= 0.0011); h3: k=45, α=-0.116, β=0.27 (FVE 0.793, A_= 0.1822)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 19: 80, 31: 81, 45: 189, 47: 79; unclustered 83.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=19 cos: 6099.7 (0.976); k=19 sin: 6439.4 (0.977); k=31 cos: 17065.8 (0.992); k=31 sin: 16900.1 (0.991); k=45 cos: 6359.3 (0.964); k=45 sin: 6392.1 (0.965); k=47 cos: 4696.3 (0.905); k=47 sin: 2159.5 (0.726).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.97; α = α_19 = 10.967, α_31 = 29.428, α_45 = 11.296, α_47 = 5.707; β = -0.0079.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 1.61%; mean CE over all inputs = 1.77e-07.
