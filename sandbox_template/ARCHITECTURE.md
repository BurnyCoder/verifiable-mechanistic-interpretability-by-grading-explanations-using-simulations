# Architecture of the network you must write weights for

A one-layer transformer that reads three tokens `a b =` and predicts `(a + b) mod 113` at the last position.

## Tokens and positions

- Vocabulary: 114 tokens. Tokens 0–112 are the numbers; token 113 is `=`.
- Every input is the sequence `[a, b, 113]` (positions 0, 1, 2), for a, b in 0..112.
- The prediction is read from the **last position (2)**. Only the first 113 logits (the numbers) are used.
- There is no LayerNorm anywhere. Attention is causal (position i sees positions ≤ i).

## Weights file: `weights.npz` with exactly these 11 float arrays

| Name | Shape | Meaning |
|---|---|---|
| `W_E` | (128, 114) | column t is the embedding of token t |
| `W_pos` | (3, 128) | row i is added to the residual stream at position i |
| `W_Q` | (4, 32, 128) | query of head h: `W_Q[h] @ x` |
| `W_K` | (4, 32, 128) | key of head h: `W_K[h] @ x` |
| `W_V` | (4, 32, 128) | value of head h: `W_V[h] @ x` |
| `W_O` | (128, 128) | attention output: `W_O @ concat(z_0, z_1, z_2, z_3)` (head-major, 4 × 32) |
| `W_in` | (512, 128) | MLP input weights |
| `b_in` | (512,) | MLP input bias |
| `W_out` | (128, 512) | MLP output weights |
| `b_out` | (128,) | MLP output bias |
| `W_U` | (128, 114) | unembedding: `logits = x @ W_U` |

## Forward pass (exactly what `forward.py` computes)

For a sequence with residual vectors x_0, x_1, x_2 ∈ R^128:

1. `x_i = W_E[:, token_i] + W_pos[i]`
2. For each head h: `q_i = W_Q[h] x_i`, `k_j = W_K[h] x_j`, `v_j = W_V[h] x_j`;
   `score_ij = (q_i · k_j) / sqrt(32)` for j ≤ i; `A_ij = softmax_j(score_ij)`; `z_i^h = Σ_j A_ij v_j`.
3. `resid_mid_i = x_i + W_O @ concat_h(z_i^h)`
4. `mlp_pre_i = W_in @ resid_mid_i + b_in`; `mlp_post_i = ReLU(mlp_pre_i)`
5. `resid_post_i = resid_mid_i + W_out @ mlp_post_i + b_out`
6. `logits_i = resid_post_i @ W_U` — the answer is `argmax(logits_2[0:113])`.

There are 4 heads, 512 MLP neurons and a 128-dimensional residual stream.
