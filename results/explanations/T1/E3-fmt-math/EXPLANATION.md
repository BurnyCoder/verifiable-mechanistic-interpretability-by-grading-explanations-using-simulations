# Explanation (mathematical summary)

Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from "=" to position 0.

- K = {11, 25, 49}.
- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.
- Attention ("=" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); to b ≈ 1 − A_h; to "=" ≈ 0.
  h0: k=49, α=0.293, β=0.055 (FVE 0.871, A_= 0.129); h1: k=25, α=0.037, β=-0.254 (FVE 0.819, A_= 0.1691); h2: k=11, α=-0.196, β=0.219 (FVE 0.963, A_= 0.0011); h3: k=11, α=0.198, β=-0.219 (FVE 0.962, A_= 0.0013)
- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; |cluster_k| = 11: 154, 25: 111, 49: 156; unclustered 91.
- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.
- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.
- Read-out coefficient (FVE): k=11 cos: 29614.8 (0.996); k=11 sin: 27931.0 (0.987); k=25 cos: 5405.9 (0.97); k=25 sin: 5587.9 (0.981); k=49 cos: 13557.9 (0.98); k=49 sin: 14892.2 (0.987).
- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE 0.984; α = α_11 = 48.898, α_25 = 9.504, α_49 = 24.245; β = -1.1439.
- ‖W_L − proj_K W_L‖² / ‖W_L‖² = 0.58%; mean CE over all inputs = 1.36e-07.
