"""Reference forward pass of the network, in numpy. You may import and use it; do not edit it.

forward(weights, tokens) -> dict of activations. `weights` is a dict with the 11 arrays described in ARCHITECTURE.md;
`tokens` is an int array [batch, 3] of (a, b, 113) rows. Everything is computed in float64.
"""

import numpy as np

P = 113  # modulus; token 113 is '='
D_HEAD = 32  # per-head query/key/value size


def all_inputs() -> np.ndarray:
    """All 113*113 inputs as rows (a, b, 113); row index = a*113 + b."""
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    return np.stack([a, b, np.full(P * P, P)], axis=1)


def labels() -> np.ndarray:
    """Correct answers (a + b) mod 113 for the rows of all_inputs()."""
    t = all_inputs()
    return (t[:, 0] + t[:, 1]) % P


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward(w: dict, tokens: np.ndarray) -> dict:
    """Run the 1-layer transformer. Returns activations at every position (see keys below)."""
    W_E, W_pos = w["W_E"], w["W_pos"]
    x = W_E[:, tokens].transpose(1, 2, 0) + W_pos[None, : tokens.shape[1]]  # [batch, pos, 128]
    q = np.einsum("hdm,bpm->bhpd", w["W_Q"], x)  # [batch, head, pos, 32]
    k = np.einsum("hdm,bpm->bhpd", w["W_K"], x)
    v = np.einsum("hdm,bpm->bhpd", w["W_V"], x)
    n = tokens.shape[1]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(D_HEAD)  # [batch, head, query, key]
    causal = np.tril(np.ones((n, n), dtype=bool))  # query position q may attend to key positions <= q
    scores = np.where(causal, scores, -1e10)
    pattern = _softmax(scores, axis=-1)  # attention weights, rows sum to 1
    z = np.einsum("bhqk,bhkd->bhqd", pattern, v)  # [batch, head, pos, 32]
    z_flat = z.transpose(0, 2, 1, 3).reshape(z.shape[0], n, -1)  # [batch, pos, 4*32], head-major
    attn_out = z_flat @ w["W_O"].T  # [batch, pos, 128]
    resid_mid = x + attn_out
    mlp_pre = resid_mid @ w["W_in"].T + w["b_in"]  # [batch, pos, 512]
    mlp_post = np.maximum(mlp_pre, 0.0)  # ReLU
    resid_post = resid_mid + mlp_post @ w["W_out"].T + w["b_out"]
    logits = resid_post @ w["W_U"]  # [batch, pos, 114]
    return {"resid_pre": x, "pattern": pattern, "attn_out": attn_out, "resid_mid": resid_mid, "mlp_pre": mlp_pre,
            "mlp_post": mlp_post, "resid_post": resid_post, "logits": logits}
