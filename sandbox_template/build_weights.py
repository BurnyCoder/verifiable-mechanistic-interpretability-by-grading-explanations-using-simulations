"""Starter build script: creates weights.npz with the 11 arrays of ARCHITECTURE.md, all zeros for now.

Replace the zeros with your construction, one section at a time (small edits), and re-run this script and
selftest.py after each change.
"""

import numpy as np

P, D_VOCAB, D_MODEL, N_HEADS, D_HEAD, D_MLP, N_CTX = 113, 114, 128, 4, 32, 512, 3

# 1. Embeddings: column t of W_E is token t (113 is "="); row i of W_pos is position i.
W_E = np.zeros((D_MODEL, D_VOCAB))
W_pos = np.zeros((N_CTX, D_MODEL))

# 2. Attention: head h uses W_Q[h], W_K[h], W_V[h]; W_O maps the 4 concatenated head outputs back.
W_Q = np.zeros((N_HEADS, D_HEAD, D_MODEL))
W_K = np.zeros((N_HEADS, D_HEAD, D_MODEL))
W_V = np.zeros((N_HEADS, D_HEAD, D_MODEL))
W_O = np.zeros((D_MODEL, N_HEADS * D_HEAD))

# 3. MLP: mlp_post = ReLU(W_in @ resid_mid + b_in); resid_post = resid_mid + W_out @ mlp_post + b_out.
W_in = np.zeros((D_MLP, D_MODEL))
b_in = np.zeros(D_MLP)
W_out = np.zeros((D_MODEL, D_MLP))
b_out = np.zeros(D_MODEL)

# 4. Unembedding: logits = resid_post @ W_U.
W_U = np.zeros((D_MODEL, D_VOCAB))

np.savez("weights.npz", W_E=W_E, W_pos=W_pos, W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O,
         W_in=W_in, b_in=b_in, W_out=W_out, b_out=b_out, W_U=W_U)
print("wrote weights.npz")
