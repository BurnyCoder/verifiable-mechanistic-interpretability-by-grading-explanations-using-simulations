"""Tests for the ninth (independent) audit's re-derivation script, scripts/audit9/verify.py.

Local explanation: the script uses neither scipy nor the project's analysis code, so its exact tests are pinned here
to values that follow from counting alone: complete separation in a 3-vs-3 one-sided Mann–Whitney gives
p = 1/C(6,3) = 0.05 and two-sided 0.1; a perfectly ordered Jonckheere–Terpstra over 3 levels × 2 runs gives
p = 1/(6!/(2!)^3) = 1/90; six positive differences give a one-sided Wilcoxon p = 1/2^6; Holm multiplies the i-th
smallest of m p-values by (m − i + 1) and keeps the running maximum.
Global context: audit 9 (2026-11-23), reports/reviews/audit-9-2026-11-23.md.
Sources: exact permutation nulls of the three rank tests; Holm 1979.
"""

from __future__ import annotations  # postponed annotations

import importlib.util  # load the script by path (scripts/ is not a package)

from vmi.config import REPO_ROOT  # repository root


def _verify():
    spec = importlib.util.spec_from_file_location("verify9", REPO_ROOT / "scripts" / "audit9" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_rank_tests_match_counting():
    v = _verify()
    assert abs(v.mw_p([4, 5, 6], [1, 2, 3])[1] - 0.05) < 1e-12
    assert abs(v.mw_p([4, 5, 6], [1, 2, 3], two_sided=True)[1] - 0.1) < 1e-12
    assert abs(v.jt_p([[1, 2], [3, 4], [5, 6]])[1] - 1 / 90) < 1e-12
    assert abs(v.wilcoxon_p([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) - 1 / 64) < 1e-12


def test_holm_step_down():
    v = _verify()
    adj = v.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["c"] - 0.06) < 1e-12 and abs(adj["b"] - 0.06) < 1e-12
