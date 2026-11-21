"""Comparison metrics between a target M and a replacement M̂ (preregistration §4).

Local explanation:
- Output fidelity: `kl_out` = mean_x KL(P_M(·|x) ‖ P_M̂(·|x)) over all 113² inputs (primary 1), plus argmax agreement,
  task accuracy, centred-logit FVU and temperature-calibrated KL (secondary).
- Hidden states: `cv_r2` = 5-fold cross-validated ridge-regression variance explained; `site_score` = harmonic mean of
  both directions; HS = mean over the three sites (primary 2). Secondary: linear CKA, orthogonal Procrustes distance,
  Hungarian-matched neuron correlation, head-matched attention distance, stitching KL.
Global context: the residual stream of a LayerNorm-free transformer is only defined up to an invertible linear map,
so the primary alignment class is linear regression (invariant to exactly that group); CKA/Procrustes are invariant
only to orthogonal maps and are reported for comparability.
Sources: KL convention KL(original ‖ modified) — Conmy et al. 2023 (https://arxiv.org/abs/2304.14997); linear CKA —
Kornblith et al. 2019 eq. (https://arxiv.org/abs/1905.00414); Procrustes distance 2 − 2‖XᵀY‖_* after centring and
Frobenius normalisation — Ding et al. 2021 (https://arxiv.org/abs/2108.01661); invariances — Klabunde et al.
(https://arxiv.org/abs/2305.06329); Hungarian matching — Li et al. 2015 (https://arxiv.org/abs/1511.07543), scipy
linear_sum_assignment (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html);
model stitching — Bansal et al. 2021 (https://arxiv.org/abs/2106.07682).
"""

from __future__ import annotations  # postponed annotations

import math  # log, golden ratio

import numpy as np  # fold permutation, Hungarian input
import torch  # all heavy linear algebra on the grading device
from scipy.optimize import linear_sum_assignment  # optimal one-to-one matching

N_FOLDS = 5  # preregistered cross-validation folds
RIDGE_REL = 1e-6  # preregistered ridge strength relative to trace(XᵀX)/d
FOLD_SEED = 0  # preregistered fold assignment seed


# ------------------------------------------------------------------------------------------------
# Output fidelity
# ------------------------------------------------------------------------------------------------
def kl_out(logits_m: torch.Tensor, logits_r: torch.Tensor) -> float:
    """Primary 1: mean over inputs of KL(softmax(logits_m) ‖ softmax(logits_r)) in nats (float64)."""
    lp = torch.log_softmax(logits_m.double(), -1)  # log P_M
    lq = torch.log_softmax(logits_r.double(), -1)  # log P_M̂
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def argmax_agreement(logits_m: torch.Tensor, logits_r: torch.Tensor) -> float:
    """Fraction of inputs on which M̂ predicts the same class as M."""
    return float((logits_m.argmax(-1) == logits_r.argmax(-1)).double().mean())


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of inputs on which the model's argmax equals the task label."""
    return float((logits.argmax(-1) == labels).double().mean())


def centred_logit_fvu(logits_m: torch.Tensor, logits_r: torch.Tensor) -> float:
    """Σ‖L_M − L_M̂‖² / Σ‖L_M − mean‖² after removing each input's mean logit (softmax-invariant shift)."""
    lm = logits_m - logits_m.mean(-1, keepdim=True)
    lr = logits_r - logits_r.mean(-1, keepdim=True)
    return float(((lm - lr) ** 2).sum() / (lm**2).sum().clamp_min(1e-300))


def calibrated_kl(logits_m: torch.Tensor, logits_r: torch.Tensor) -> tuple[float, float]:
    """KL after rescaling M̂'s logits by the temperature that minimises it (golden-section search on log scale)."""
    f = lambda log_s: kl_out(logits_m, logits_r * math.exp(log_s))  # noqa: E731 — KL as a function of log-scale
    lo, hi = -8.0, 8.0  # scales e^-8 … e^8 cover every plausible logit scale
    g = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(60):  # 60 iterations shrink the bracket by 0.618^60 ≈ 3e-13
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = f(b)
    best = (a + b) / 2
    return f(best), math.exp(best)


# ------------------------------------------------------------------------------------------------
# Hidden-state similarity: cross-validated ridge regression (primary)
# ------------------------------------------------------------------------------------------------
def folds(n: int, k: int = N_FOLDS, seed: int = FOLD_SEED) -> list[np.ndarray]:
    """Fixed random partition of n rows into k folds (numpy default_rng(seed))."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


def cv_r2(X: torch.Tensor, Y: torch.Tensor, k: int = N_FOLDS, ridge_rel: float = RIDGE_REL, seed: int = FOLD_SEED) -> float:
    """Held-out variance of Y explained by a ridge-regularised affine map from X, pooled over folds and dims, clipped to [0, 1].

    Each fold: centre X and Y with the training-fold means, solve (XᵀX + λI)W = XᵀY with λ = ridge_rel·tr(XᵀX)/d,
    predict the held-out rows. R² = 1 − ΣSSE/ΣSST with SST about the training-fold mean of Y. A Y with no variance
    (e.g. a dead layer) gets R² = 0 by definition.
    """
    X, Y = X.double(), Y.double()
    sse, sst = 0.0, 0.0
    for test_idx in folds(X.shape[0], k, seed):
        test = torch.as_tensor(test_idx, device=X.device)
        train_mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        train_mask[test] = False
        Xtr, Ytr = X[train_mask], Y[train_mask]
        mx, my = Xtr.mean(0), Ytr.mean(0)  # training-fold means (affine map)
        Xc, Yc = Xtr - mx, Ytr - my
        G = Xc.T @ Xc  # Gram matrix [dx, dx]
        lam = ridge_rel * float(torch.trace(G)) / max(G.shape[0], 1) + 1e-12  # tiny floor keeps a zero X solvable
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), Xc.T @ Yc)
        pred = (X[test] - mx) @ W + my
        sse += float(((Y[test] - pred) ** 2).sum())
        sst += float(((Y[test] - my) ** 2).sum())
    if sst <= 1e-12 * X.shape[0]:
        return 0.0
    return float(min(max(1.0 - sse / sst, 0.0), 1.0))


def site_score(x_m: torch.Tensor, x_r: torch.Tensor, **kw) -> dict[str, float]:
    """Completeness R²(M̂→M), soundness R²(M→M̂) and their harmonic mean for one site."""
    comp = cv_r2(x_r, x_m, **kw)  # how much of M's state is linearly recoverable from M̂'s
    sound = cv_r2(x_m, x_r, **kw)  # how much of M̂'s state is linearly recoverable from M's
    hm = 0.0 if comp + sound == 0 else 2 * comp * sound / (comp + sound)
    return {"completeness": comp, "soundness": sound, "harmonic": hm}


# ------------------------------------------------------------------------------------------------
# Secondary hidden-state metrics
# ------------------------------------------------------------------------------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA = ‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F) on column-centred matrices (Kornblith et al. 2019)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    den = float(torch.linalg.norm(Xc.T @ Xc) * torch.linalg.norm(Yc.T @ Yc))
    return 0.0 if den == 0 else float(torch.linalg.norm(Yc.T @ Xc) ** 2) / den


def procrustes_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Orthogonal Procrustes distance 2 − 2‖XᵀY‖_* after centring and unit-Frobenius normalisation (Ding et al. 2021); in [0, 2]."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    nx, ny = torch.linalg.norm(Xc), torch.linalg.norm(Yc)
    if nx == 0 or ny == 0:
        return 2.0
    nuc = torch.linalg.svdvals((Xc / nx).T @ (Yc / ny)).sum()  # nuclear norm
    return float(2 - 2 * nuc)


def _corr_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pearson correlations between every column of A and every column of B; zero-variance columns give 0."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    sa, sb = Ac.norm(dim=0), Bc.norm(dim=0)
    C = (Ac.T @ Bc) / (sa[:, None] * sb[None, :]).clamp_min(1e-300)
    C[sa == 0, :] = 0.0
    C[:, sb == 0] = 0.0
    return C


def matched_neuron_corr(post_m: torch.Tensor, post_r: torch.Tensor) -> float:
    """Mean correlation over the target's live neurons after the one-to-one matching that maximises total correlation.

    Permutation- and positive-scale-invariant. Target neurons that never vary (dead on every input) carry no mechanism
    and are excluded; a live target neuron matched to a dead replacement neuron counts as correlation 0.
    """
    live = (post_m - post_m.mean(0)).norm(dim=0) > 0  # target neurons with any variance
    if not bool(live.any()):
        return 0.0
    C = _corr_matrix(post_m[:, live], post_r).cpu().numpy()
    rows, cols = linear_sum_assignment(-C)  # maximise Σ correlation
    return float(C[rows, cols].sum() / int(live.sum()))


def matched_attention_distance(pat_m: torch.Tensor, pat_r: torch.Tensor) -> float:
    """Mean |A_M − A_M̂| of attention from '=' after matching heads one-to-one (head order is arbitrary)."""
    H = pat_m.shape[1]
    D = torch.stack([torch.stack([(pat_m[:, i] - pat_r[:, j]).abs().mean() for j in range(H)]) for i in range(H)]).cpu().numpy()
    rows, cols = linear_sum_assignment(D)
    return float(D[rows, cols].mean())


def stitching_kl(target_model, resid_r: torch.Tensor, resid_m: torch.Tensor, logits_m: torch.Tensor, k: int = N_FOLDS) -> float:
    """Causal test: map M̂'s resid_mid linearly into M's (fit on training folds), run M's MLP + unembedding, KL to M on held-out rows."""
    mlp = target_model.blocks[0].mlp
    W_in, b_in = mlp.W_in.double(), mlp.b_in.double()
    W_out, b_out = mlp.W_out.double(), mlp.b_out.double()
    W_U = target_model.unembed.W_U.double()[:, : logits_m.shape[1]]
    out = torch.empty_like(logits_m)
    for test_idx in folds(resid_r.shape[0], k):
        test = torch.as_tensor(test_idx, device=resid_r.device)
        mask = torch.ones(resid_r.shape[0], dtype=torch.bool, device=resid_r.device)
        mask[test] = False
        # Affine stitching map by the same ridge solve as cv_r2 (centred, λ = RIDGE_REL·tr(G)/d): torch.linalg.lstsq's
        # only CUDA driver ('gels') assumes full rank and returned NaN for rank-deficient replacement residuals
        # (https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).
        Xtr, Ytr = resid_r[mask].double(), resid_m[mask].double()
        mx, my = Xtr.mean(0), Ytr.mean(0)
        Xc = Xtr - mx
        G = Xc.T @ Xc
        lam = RIDGE_REL * float(torch.trace(G)) / max(G.shape[0], 1) + 1e-12
        A = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), Xc.T @ (Ytr - my))
        r = (resid_r[test].double() - mx) @ A + my  # stitched residual in M's basis
        post = torch.relu(r @ W_in.T + b_in)
        out[test] = (r + post @ W_out.T + b_out) @ W_U
    return kl_out(logits_m, out)


def embedding_spectrum_cosine(W_E_m: torch.Tensor, W_E_r: torch.Tensor, F: torch.Tensor) -> float:
    """Cosine similarity of the per-frequency norms of the two token embeddings (number tokens only)."""
    def spectrum(W_E):
        c = W_E[:, : F.shape[0]].double() @ F.T  # [d_model, p] Fourier coefficients along the token axis
        n = c.norm(dim=0)
        return torch.sqrt(n[1::2] ** 2 + n[2::2] ** 2)  # combine cos/sin per frequency → [(p-1)/2]
    a, b = spectrum(W_E_m), spectrum(W_E_r)
    den = float(a.norm() * b.norm())
    return 0.0 if den == 0 else float(a @ b) / den
