"""Compile an explanation spec into weights of the paper architecture — the agent-free reference point (baseline; not a ceiling: the agent exceeds it on some targets).

Local explanation: the spec (built by `explainer/ladder.py`, the same dict the explanation text is rendered from)
says which frequencies, attention patterns, neuron allocations, per-neuron functions and logit amplitudes the
explanation states. The compiler builds, in order:
1. **Embedding**: number token t → (cos w_k t, sin w_k t) for each stated embedding frequency, plus a constant
   "is-number" dimension; '=' → an "is-equals" dimension. Positional embeddings are zero.
2. **Attention**: the '=' query reads "is-equals"; keys = G·is-number + 4α cos(w t) + 4β sin(w t) for a head with a
   stated pattern A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) (σ(x) ≈ 0.5 + x/4 in the linear regime; G = 30
   removes attention to '='), else G only (uniform over a and b). Each head's OV copies the Fourier dimensions into
   its own residual block.
3. **MLP**: single-frequency neurons. With a per-neuron table (E4), each neuron's input weights are the least-squares
   fit, over all inputs, from this model's own resid_mid to the stated pre-activation function; otherwise neurons
   of frequency k with phases φ_j = 2πj/n_k read cos(wa+φ) + cos(wb+φ) from the head blocks (n_k from the stated
   cluster sizes, else an equal split).
4. **Read-out**: least squares, over all inputs, from this model's own neuron activations to the stated logit formula
   (Clock: Σ α_k cos(w_k(a+b−c)); Pizza: Σ α_k |cos(w_k(a−b)/2)| cos(w_k(a+b−c))), factorised by SVD into W_out and
   W_U through dedicated output dimensions (the paper's W_L = W_U W_out structure).
The logit scale comes from the stated amplitudes, else from the stated loss (1-D search), else a default loss 1e-6.
Global context: no target weight or activation is used — only the spec — so the compiler measures what the
explanation alone determines, separating explanation quality from agent skill (preregistration H5).
E0 (audit 5): E0's spec says `algorithm: "none"`, which this compiler does not treat specially; it builds its default
Clock network at DEFAULT_FREQS, exactly as for E1, so the compiler's E0 and E1 scores are identical and its E0 score is
the compiler's own default build, not a reading of the E0 text.
Sources: Fourier-multiplication algorithm and W_L structure — Nanda et al. 2023 §3–4 (https://arxiv.org/abs/2301.05217);
Pizza algorithm — Zhong et al. 2023 (https://arxiv.org/abs/2306.17844); quadrature view of single-frequency ReLU
neurons with uniformly spaced phases — Yip et al. 2024 (https://arxiv.org/abs/2412.03773); closed-form cosine weights —
Gromov 2023 (https://arxiv.org/abs/2301.02679).
"""

from __future__ import annotations  # postponed annotations

import math  # pi, sqrt

import numpy as np  # construction is plain numpy, like the agent's

from vmi.config import D_HEAD, D_MLP, D_MODEL, D_VOCAB, N_HEADS, P  # architecture

DEFAULT_FREQS = sorted((np.random.default_rng(0).choice(56, size=5, replace=False) + 1).tolist())  # used when unstated
DEFAULT_LOSS = 1e-6  # target cross-entropy when neither amplitudes nor loss are stated
G_KEY = 30.0  # key offset for number tokens: attention to '=' ≈ e^-30


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    """a and b for every row a·p + b."""
    return np.repeat(np.arange(P), P), np.tile(np.arange(P), P)


def _forward_to_mlp(W: dict) -> tuple[np.ndarray, np.ndarray]:
    """resid_mid and mlp_post at the '=' position for all inputs (numpy re-implementation of the forward pass)."""
    a, b = _inputs()
    tokens = np.stack([a, b, np.full(P * P, P)], 1)
    x = W["W_E"][:, tokens].transpose(1, 2, 0) + W["W_pos"][None]
    q = np.einsum("hdm,bm->bhd", W["W_Q"], x[:, 2])  # only the '=' query matters for the read-out
    k = np.einsum("hdm,bpm->bhpd", W["W_K"], x)
    v = np.einsum("hdm,bpm->bhpd", W["W_V"], x)
    s = np.einsum("bhd,bhpd->bhp", q, k) / math.sqrt(D_HEAD)
    s = s - s.max(-1, keepdims=True)
    A = np.exp(s) / np.exp(s).sum(-1, keepdims=True)  # [batch, head, 3]
    z = np.einsum("bhp,bhpd->bhd", A, v).reshape(len(a), -1)  # head-major concat
    resid_mid = x[:, 2] + z @ W["W_O"].T
    post = np.maximum(resid_mid @ W["W_in"].T + W["b_in"], 0.0)
    return resid_mid, post


def _ridge(X: np.ndarray, Y: np.ndarray, rel: float = 1e-8) -> np.ndarray:
    """Ridge least squares (XᵀX + λI)⁻¹XᵀY with λ = rel·tr(XᵀX)/d: like lstsq, but keeps weights moderate when
    features are nearly collinear (huge cancelling weights would be fragile)."""
    G = X.T @ X
    lam = rel * np.trace(G) / G.shape[0] + 1e-12
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), X.T @ Y)


def _logit_target(spec: dict, freqs: list[int], amps: dict) -> np.ndarray:
    """The stated logit formula on all inputs: [p², 113]."""
    a, b = _inputs()
    c = np.arange(P)
    T = np.zeros((P * P, P))
    for k in freqs:
        w = 2 * math.pi * k / P
        wave = np.cos(w * ((a + b)[:, None] - c[None, :]))
        if spec.get("algorithm") == "pizza":
            wave = np.abs(np.cos(w * (a - b) / 2))[:, None] * wave
        T += float(amps.get(k, amps.get(str(k), 1.0))) * wave
    return T


def _loss(logits: np.ndarray) -> float:
    """Mean cross-entropy of (a+b) mod p under the given logits."""
    a, b = _inputs()
    y = (a + b) % P
    z = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1))
    return float((lse - z[np.arange(len(y)), y]).mean())


def compile_spec(spec: dict) -> dict[str, np.ndarray]:
    """Build the 11 weight arrays from a spec (see module docstring)."""
    K_e = list(spec.get("embed_freqs") or DEFAULT_FREQS)  # frequencies present in the embedding
    K_m = list(spec.get("mlp_freqs") or K_e)  # frequencies the MLP and read-out use (differ only for W-role)
    nK = len(K_e)
    tok = {k: (2 * i, 2 * i + 1) for i, k in enumerate(K_e)}  # residual dims holding cos/sin of each frequency
    D_ONE, D_EQ = 2 * nK, 2 * nK + 1
    block = [2 * nK + 2 + h * 2 * nK for h in range(N_HEADS)]  # start of each head's copy of the Fourier dims
    out0 = 2 * nK + 2 + N_HEADS * 2 * nK  # first read-out dimension
    assert out0 < D_MODEL and 2 * nK <= D_HEAD, "too many frequencies for this construction"
    W = {"W_E": np.zeros((D_MODEL, D_VOCAB)), "W_pos": np.zeros((3, D_MODEL)), "W_Q": np.zeros((N_HEADS, D_HEAD, D_MODEL)),
         "W_K": np.zeros((N_HEADS, D_HEAD, D_MODEL)), "W_V": np.zeros((N_HEADS, D_HEAD, D_MODEL)),
         "W_O": np.zeros((D_MODEL, N_HEADS * D_HEAD)), "W_in": np.zeros((D_MLP, D_MODEL)), "b_in": np.zeros(D_MLP),
         "W_out": np.zeros((D_MODEL, D_MLP)), "b_out": np.zeros(D_MODEL), "W_U": np.zeros((D_MODEL, D_VOCAB))}
    t = np.arange(P)
    # 1. embedding
    for k, (ic, is_) in tok.items():
        w = 2 * math.pi * k / P
        W["W_E"][ic, :P], W["W_E"][is_, :P] = np.cos(w * t), np.sin(w * t)
    W["W_E"][D_ONE, :P] = 1.0
    W["W_E"][D_EQ, P] = 1.0
    # 2. attention
    heads = list(spec.get("heads") or [])
    for h in range(N_HEADS):
        W["W_Q"][h, 0, D_EQ] = math.sqrt(D_HEAD)  # '=' query; ×1/√d_head in the forward pass cancels this
        W["W_K"][h, 0, D_ONE] = G_KEY
        if h < len(heads) and int(heads[h]["freq"]) in tok:
            ic, is_ = tok[int(heads[h]["freq"])]
            W["W_K"][h, 0, ic] = 4 * float(heads[h]["alpha"])  # σ(x) ≈ 0.5 + x/4 → slope 4
            W["W_K"][h, 0, is_] = 4 * float(heads[h]["beta"])
        for j in range(2 * nK):  # OV: copy all Fourier dims into this head's block
            W["W_V"][h, j, j] = 1.0
            W["W_O"][block[h] + j, h * D_HEAD + j] = 1.0
    # 3. MLP
    neurons = spec.get("neurons")
    if neurons:  # per-neuron table: least-squares fit from this model's resid_mid to each stated pre-activation
        resid_mid, _ = _forward_to_mlp(W)
        a, b = _inputs()
        X = np.concatenate([resid_mid, np.ones((P * P, 1))], 1)
        targets = np.zeros((P * P, D_MLP))
        for row in neurons[:D_MLP]:
            k = int(row["freq"])
            w = 2 * math.pi * k / P
            f = row["bias"] + row["a_amp"] * np.cos(w * a - math.radians(row["a_phase"])) \
                + row["b_amp"] * np.cos(w * b - math.radians(row["b_phase"])) \
                + row["sum_amp"] * np.cos(w * (a + b) - math.radians(row["sum_phase"])) \
                + row["diff_amp"] * np.cos(w * (a - b) - math.radians(row["diff_phase"]))
            targets[:, int(row["neuron"])] = f if k in K_m else row["bias"]  # W-role: frequency absent → constant
        sol = _ridge(X, targets)  # [129, 512]
        W["W_in"] = sol[:-1].T
        W["b_in"] = sol[-1]
    else:  # neurons of frequency k, evenly spaced phases φ, target pre-activation cos(wa+φ) + cos(wb+φ)
        clusters = spec.get("neuron_clusters") or {k: D_MLP // len(K_m) for k in K_m}
        counts = {int(k): int(n) for k, n in clusters.items() if int(k) in K_m}
        total = sum(counts.values())
        if total > D_MLP:  # rescale proportionally if the stated sizes exceed the layer
            counts = {k: n * D_MLP // total for k, n in counts.items()}
        resid_mid, _ = _forward_to_mlp(W)
        a, b = _inputs()
        X = np.concatenate([resid_mid, np.ones((P * P, 1))], 1)
        targets = np.full((P * P, D_MLP), -1.0)  # unused neurons: constant −1 → dead after ReLU
        i = 0
        for k, n in counts.items():
            w = 2 * math.pi * k / P
            for j in range(n):
                phi = 2 * math.pi * j / max(n, 1)
                # least squares picks the head-block combination that best yields the symmetric linear form, so
                # attention patterns that depend on a and b (Clock heads) are used rather than ignored
                targets[:, i] = np.cos(w * a + phi) + np.cos(w * b + phi) if k in tok else 0.0
                i += 1
        sol = _ridge(X, targets)
        W["W_in"] = sol[:-1].T
        W["b_in"] = sol[-1]
    # 4. read-out: least squares from neuron activations (plus the residual stream the unembedding also sees) to the
    #    stated logit formula; the neuron part is factorised by SVD into W_out and W_U through dedicated output dims
    resid_mid, post = _forward_to_mlp(W)
    used = slice(0, out0)  # residual dims in use before the MLP (token, flags, head blocks)
    amps = spec.get("logit_amplitudes") or {}
    T = _logit_target(spec, K_m, amps)
    Xp = np.concatenate([post, resid_mid[:, used], np.ones((P * P, 1))], 1)
    L = _ridge(Xp, T - T.mean(1, keepdims=True))  # [512 + out0 + 1, 113]
    L_post = np.concatenate([L[:D_MLP], L[-1:]], 0)  # neuron weights + bias row → through W_out
    U, S, Vt = np.linalg.svd(L_post, full_matrices=False)
    r = int(min((S > S[0] * 1e-8).sum(), D_MODEL - out0))  # rank that fits the free dimensions
    W["W_out"][out0:out0 + r] = (U[:-1, :r] * S[:r]).T
    W["b_out"][out0:out0 + r] = U[-1, :r] * S[:r]
    W["W_U"][out0:out0 + r, :P] = Vt[:r]
    W["W_U"][used, :P] = L[D_MLP:D_MLP + out0]  # direct path: resid_mid dims read by the unembedding
    if not amps:  # scale from the stated loss (or the default) by bisection on log-scale
        base = Xp @ L
        target = float(spec.get("loss_all_inputs") or DEFAULT_LOSS)
        lo, hi = -5.0, 8.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _loss(base * math.exp(mid)) > target else (lo, mid)
        W["W_U"][:, :P] *= math.exp((lo + hi) / 2)
    return W
