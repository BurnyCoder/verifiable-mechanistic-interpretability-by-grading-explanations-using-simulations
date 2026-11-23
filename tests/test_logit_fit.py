"""Tests for the post-hoc logit-level comparison and the logit-matching loss (audit 9, deviation 16).

Global context: `calibrated_logit_fvu` is read as "how much of the target's centred-logit pattern the replacement
reproduces up to one positive scale"; these tests pin the properties that reading needs (0 for a rescaled copy, 1 for
an unrelated or sign-flipped pattern, never above the raw FVU, blind to a per-input constant), and that the training
loss of the logit students is zero exactly when the centred logits agree.
Sources: closed-form least-squares scale; softmax invariance to a per-input constant.
"""

import torch

from vmi.analysis.logit_fit import calibrated_logit_fvu, centred_logits
from vmi.grader.metrics import centred_logit_fvu
from vmi.nonagent import retrain_ids
from vmi.training import centred, logit_mse


def _logits(seed: int, n: int = 200, c: int = 113) -> torch.Tensor:
    return torch.randn(n, c, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_rescaled_copy_is_fully_explained():
    L = _logits(0)
    fvu, scale = calibrated_logit_fvu(L, 3.0 * L)
    assert abs(fvu) < 1e-12 and abs(scale - 1 / 3) < 1e-12


def test_unrelated_and_sign_flipped_patterns_explain_nothing():
    L = _logits(0)
    assert calibrated_logit_fvu(L, -L)[0] == 1.0  # only positive scales count, like a temperature
    assert calibrated_logit_fvu(L, _logits(1))[0] > 0.98  # independent random tables share almost nothing


def test_calibrated_never_exceeds_raw_and_ignores_per_input_constants():
    L, R = _logits(0), 0.5 * _logits(0) + 0.3 * _logits(2)
    shift = torch.randn(L.shape[0], 1, dtype=torch.float64) * 10
    cal = calibrated_logit_fvu(L, R)[0]
    assert cal <= centred_logit_fvu(L, R) + 1e-12
    assert abs(calibrated_logit_fvu(L, R + shift)[0] - cal) < 1e-12
    assert torch.allclose(centred_logits(R + shift), centred_logits(R))


def test_logit_mse_is_zero_only_for_matching_centred_logits():
    L = _logits(3)
    assert float(logit_mse(L + 5.0, centred(L))) < 1e-24  # a per-input constant does not count
    assert float(logit_mse(2 * L, centred(L))) > 0


def test_retrain_ids_exclude_target_and_dev():
    ids = retrain_ids("T0")
    assert len(ids) == 9 and "wd1_s1" not in ids and "wd1_s10" not in ids
