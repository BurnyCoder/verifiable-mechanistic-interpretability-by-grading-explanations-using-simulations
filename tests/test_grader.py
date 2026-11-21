"""Tests for the grader: identity, symmetry invariance, failure handling and agreement with reference formulas.

Global context: the grader is the measuring instrument of the whole study; these tests check the properties the
preregistration relies on (KL = 0 and HS = 1 for M vs M; invariance to the architecture's symmetries; failed runs
scored as the zero model) before any agent run is graded.
Sources: Google representation_similarity `feature_space_linear_cka` (Apache-2.0,
https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb) re-derived here as an
independent reference; scipy orthogonal_procrustes for the Procrustes check.
"""

import numpy as np
import pytest
import torch
from grokviz.model import GrokTransformer
from scipy.linalg import orthogonal_procrustes

from vmi import weights
from vmi.grader import metrics as mt
from vmi.grader.score import grade_models, grade_npz


def _random_model(seed: int) -> GrokTransformer:
    """A randomly initialised paper-layout model (no download needed)."""
    return GrokTransformer(seed=seed).eval()


def _linear_change_of_basis(arrays: dict, seed: int = 1) -> dict:
    """Apply an invertible map A to the residual stream (writers ×A, readers ×A⁻¹): the function is unchanged."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(128, 128)) + 3 * np.eye(128)  # well-conditioned, not orthogonal
    Ai = np.linalg.inv(A)
    w = {k: v.copy() for k, v in arrays.items()}
    w["W_E"] = A @ arrays["W_E"]
    w["W_pos"] = arrays["W_pos"] @ A.T
    for k in ("W_Q", "W_K", "W_V"):
        w[k] = arrays[k] @ Ai  # [h, d_head, d_model] @ [d_model, d_model]
    w["W_O"] = A @ arrays["W_O"]
    w["W_in"] = arrays["W_in"] @ Ai
    w["W_out"] = A @ arrays["W_out"]
    w["b_out"] = A @ arrays["b_out"]
    w["W_U"] = Ai.T @ arrays["W_U"]
    return w


def _permute_and_scale_neurons(arrays: dict, seed: int = 2) -> dict:
    """Permute MLP neurons and rescale each by a positive factor (ReLU is positively homogeneous)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(512)
    s = rng.uniform(0.5, 2.0, size=512)
    w = {k: v.copy() for k, v in arrays.items()}
    w["W_in"] = s[:, None] * arrays["W_in"][perm]
    w["b_in"] = s * arrays["b_in"][perm]
    w["W_out"] = arrays["W_out"][:, perm] / s[None, :]
    return w


@pytest.fixture(scope="module")
def base():
    """A random model with non-zero MLP biases so every parameter matters."""
    m = _random_model(0)
    with torch.no_grad():
        m.blocks[0].mlp.b_in.normal_(0, 0.3, generator=torch.Generator().manual_seed(5))
        m.blocks[0].mlp.b_out.normal_(0, 0.3, generator=torch.Generator().manual_seed(6))
    return m


def test_identity_gives_perfect_scores(base):
    res = grade_models(base, weights.to_model(weights.from_model(base)))
    assert res["kl_out"] < 1e-9
    assert res["hs"] > 0.999
    assert res["argmax_agreement"] == 1.0
    assert res["matched_neuron_corr"] > 0.999


def test_invariant_to_residual_basis_change(base):
    res = grade_models(base, weights.to_model(_linear_change_of_basis(weights.from_model(base))))
    assert res["kl_out"] < 1e-5  # float32 forward pass: tiny numerical differences only
    assert res["hs"] > 0.999  # linear-regression alignment absorbs any invertible map
    assert res["cka"]["resid_mid"] < 0.999  # CKA is NOT invariant to a non-orthogonal map (why it is secondary)


def test_invariant_to_neuron_permutation_and_scaling(base):
    res = grade_models(base, weights.to_model(_permute_and_scale_neurons(weights.from_model(base))))
    assert res["kl_out"] < 1e-5
    assert res["hs"] > 0.999
    assert res["matched_neuron_corr"] > 0.999


def test_zero_model_scores_zero_hs(base):
    res = grade_models(base, weights.to_model(weights.zeros()))
    assert res["hs"] == 0.0
    assert res["kl_out"] > 0.0


def test_missing_file_is_failed_zero_model(base, tmp_path):
    res = grade_npz(base, tmp_path / "nope.npz")
    assert res["failed"] and res["hs"] == 0.0


def test_wrong_shape_is_failed(base, tmp_path):
    arrays = weights.from_model(base)
    arrays["W_U"] = arrays["W_U"][:, :113]  # common agent mistake: drop the '=' column
    np.savez(tmp_path / "w.npz", **arrays)
    res = grade_npz(base, tmp_path / "w.npz")
    assert res["failed"] and "W_U" in res["error"]


def _google_feature_space_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Independent reference: Google's feature_space_linear_cka (non-debiased) re-derived from its demo."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    dot = np.linalg.norm(x.T @ y) ** 2
    return dot / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))


def test_cka_and_procrustes_match_references():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 20))
    y = x @ rng.normal(size=(20, 20)) + 0.5 * rng.normal(size=(500, 20))
    assert abs(mt.linear_cka(torch.tensor(x), torch.tensor(y)) - _google_feature_space_linear_cka(x, y)) < 1e-10
    xc, yc = x - x.mean(0), y - y.mean(0)
    xc, yc = xc / np.linalg.norm(xc), yc / np.linalg.norm(yc)
    R, _ = orthogonal_procrustes(yc, xc)  # rotation aligning y to x
    ref = np.linalg.norm(xc - yc @ R) ** 2  # squared distance after optimal rotation = 2 − 2‖XᵀY‖_*
    assert abs(mt.procrustes_distance(torch.tensor(x), torch.tensor(y)) - ref) < 1e-10


def test_cv_r2_detects_unrelated_states():
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(2000, 30)))
    assert mt.cv_r2(x, torch.tensor(rng.normal(size=(2000, 30)))) < 0.05  # independent noise: nothing explained
    assert mt.cv_r2(x, x @ torch.tensor(rng.normal(size=(30, 10)))) > 0.999  # exact linear function: everything


def test_stitching_kl_is_finite_for_rank_deficient_replacements(base):
    """Stitching must work when the replacement's residual stream spans only a subspace (agent builds often do);
    torch.linalg.lstsq's only CUDA driver ('gels') assumes full rank and returned NaN there."""
    from vmi.grader.activations import collect
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = base.double().to(dev)
    acts = collect(m, dev)
    deficient = acts.resid_mid.clone()
    deficient[:, 64:] = 0.0  # half the residual dimensions unused
    kl_self = mt.stitching_kl(m, acts.resid_mid, acts.resid_mid, acts.logits)
    kl_def = mt.stitching_kl(m, deficient, acts.resid_mid, acts.logits)
    assert np.isfinite(kl_self) and kl_self < 1e-6  # M stitched into itself reproduces M
    assert np.isfinite(kl_def) and kl_def >= kl_self


def test_orthogonal_cv_r2_is_one_for_rotations_and_lower_for_general_maps():
    """Orthogonal (similarity) alignment recovers a scaled rotation exactly but not a general invertible linear map."""
    from vmi.sensitivity import cv_r2_orth
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 16, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=torch.float64))
    assert cv_r2_orth(X, 3.0 * X @ Q + 1.0) > 0.999
    A = torch.randn(16, 16, generator=g, dtype=torch.float64) + 3 * torch.diag(torch.linspace(0.1, 5, 16, dtype=torch.float64))
    assert mt.cv_r2(X, X @ A) > 0.999 and cv_r2_orth(X, X @ A) < 0.99
