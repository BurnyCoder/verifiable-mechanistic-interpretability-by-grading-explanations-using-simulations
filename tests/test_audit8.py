"""Tests for the eighth (independent) audit's re-derivation script, scripts/audit8/verify.py.

Local explanation: the script avoids scipy and the project's own analysis code, so its exact tests are checked here
against values that follow from counting alone: a 3-vs-3 one-sided Mann–Whitney with complete separation has
p = 1/C(6,3) = 0.05; a Jonckheere–Terpstra test over 4 levels × 3 runs in perfect order has p = 1/369,600 (the number
of distinct assignments, 12!/(3!)^4); a one-sided Wilcoxon signed-rank test with 6 positive differences has
p = 1/2^6; Holm multiplies the i-th smallest of m p-values by (m − i + 1) and keeps the running maximum. The two
verdict readings are checked on the case that motivated the audit (a defined test whose smallest p is α).
Global context: audit 8 (2026-11-23), reports/reviews/audit-8-2026-11-23.md; preregistration §5–§6 define the verdicts.
Sources: Mann–Whitney exact null (all splits equally likely); Jonckheere 1954, Terpstra 1952; Wilcoxon exact null (all
sign patterns equally likely); Holm 1979.
"""

from __future__ import annotations  # postponed annotations

import importlib.util  # load the script by path (scripts/ is not a package)
import math  # factorials

import numpy as np  # samples
import pytest  # approx
from scipy import stats  # reference implementation, used only here

from vmi.config import REPO_ROOT  # repository root


def _verify():
    """Import scripts/audit8/verify.py as a module."""
    spec = importlib.util.spec_from_file_location("verify", REPO_ROOT / "scripts" / "audit8" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _verify()


def test_mann_whitney_complete_separation_is_one_in_twenty():
    assert V.mw_greater([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) == 0.05


def test_mann_whitney_matches_scipy_exact():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=3), rng.normal(size=9)
    assert math.isclose(V.mw_greater(x, y), stats.mannwhitneyu(x, y, alternative="greater", method="exact").pvalue)


def test_jonckheere_perfect_order_is_one_over_all_assignments():
    groups = [np.arange(3) + 3 * i for i in range(4)]
    n = math.factorial(12) // math.factorial(3) ** 4
    assert n == 369_600
    assert math.isclose(V.jt_greater(groups), 1 / n)


def test_wilcoxon_six_positive_is_one_in_sixty_four():
    assert V.wilcoxon_greater([0.6, 0.7, 0.65, 0.68, 0.72, 0.71]) == 1 / 64


def test_holm_step_down():
    assert V.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_verdict_readings_differ_only_for_a_powerless_defined_test():
    # Preregistration §5: "cannot tell" only when the test is undefined; deviation 1: also when min p ≥ α.
    assert V.verdicts(0.05, True, 0.05) == ("not supported", "cannot tell at this N")
    assert V.verdicts(None, True, 1.0) == ("cannot tell at this N", "cannot tell at this N")
    assert V.verdicts(0.004, True, 0.004) == ("supported", "supported")
