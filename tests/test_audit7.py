"""Tests for the seventh audit's scripts (deviation 14).

Local explanation: (1) the enumeration-based tests in scripts/audit7/verdicts.py agree with scipy's exact tests on
small samples; (2) the loop detector in scripts/audit7/build_script_searches.py flags keep-the-best and fixed-count
calibration loops but not ordinary construction loops; (3) the inline values the paper quotes are generated from the
audit-7 results files.
Global context: verdicts.py deliberately avoids scipy so that an error there cannot carry over; this test is where the
two are compared. The paper's `v:a7-*` markers are filled by `vmi.analysis.tables._audit7_values`.
Sources: scipy.stats mannwhitneyu / wilcoxon / spearmanr (https://docs.scipy.org/doc/scipy/reference/stats.html).
"""

from __future__ import annotations  # postponed annotations

import importlib.util  # load the audit scripts by path (scripts/ is not a package)

import numpy as np  # samples
import pytest  # skips
from scipy import stats  # reference tests

from vmi.config import REPO_ROOT  # repository root


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "audit7" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "audit7"))  # build_script_searches imports verdicts
    spec.loader.exec_module(mod)
    return mod


def test_enumerated_tests_match_scipy():
    v = _load("verdicts")
    rng = np.random.default_rng(1)
    x, y = rng.normal(1, 1, 3), rng.normal(0, 1, 4)
    assert v.mw_greater(x, y) == pytest.approx(stats.mannwhitneyu(x, y, alternative="greater", method="exact").pvalue)
    assert v.mw_two_sided(x, y) == pytest.approx(stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue)
    d = rng.normal(0.5, 1, 6)
    assert v.wilcoxon_greater(d) == pytest.approx(stats.wilcoxon(d, alternative="greater").pvalue)
    a, b = rng.normal(size=9), rng.normal(size=9)
    assert v.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)
    assert v.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_jt_exact_extremes():
    v = _load("verdicts")
    J, p = v.jt([np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])])
    assert J == 12 and p == pytest.approx(1 / 90)  # perfect order: 1 of 6!/(2!·2!·2!) assignments


def test_loop_detector():
    s = _load("build_script_searches")
    keep_best = "best = None\nfor c in [0.5, 1.0]:\n    err = f(c)\n    if best is None or err < best[0]:\n        best = (err, c)\n"
    calib = "for _ in range(8):\n    s2 += np.log(t / a)\n"
    plain = "for h in range(4):\n    W_Q[h] = q\nfor _ in range(D_MLP - n):  # padding\n    rows.append(0)\n"
    assert s.BEST.search(keep_best) and s.FIXED_ITER.search(calib)
    assert not s.BEST.search(plain) and not s.FIXED_ITER.search(plain)


def test_inline_values_come_from_results():
    if not (REPO_ROOT / "results" / "audit7" / "build_script_searches.json").exists():
        pytest.skip("audit-7 results not generated on this machine")
    from vmi.analysis.tables import _audit7_values
    v = _audit7_values()
    assert v["a7-search"].endswith(" of 96") and "explanation states" in v["a7-search-kinds"]
    assert v["a7-verdicts"].split(" of ")[0] == v["a7-verdicts"].split(" of ")[1]  # every check agrees
