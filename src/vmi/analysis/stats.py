"""Statistical tests used by the preregistered hypotheses (preregistration §5).

Local explanation: Jonckheere–Terpstra trend test (exact permutation when there are ≤ 400,000 distinct assignments of
runs to levels, Monte-Carlo permutation otherwise), Spearman ρ with a bootstrap CI that resamples runs within groups, one-sided
Mann–Whitney U, Holm correction, Kendall's W with a permutation p-value, Wilcoxon signed-rank, rank-biserial effect size.
Global context: `hypotheses.py` applies these to per-run scores; every function returns plain floats for JSON.
Sources: Jonckheere–Terpstra trend test — Terpstra 1952 (Indagationes Math. 14:327–333) statistic J = Σ_{i<j} #{(x, y):
x ∈ G_i, y ∈ G_j, x < y} + ½ ties, equivalent to Jonckheere's 1954 (Biometrika 41:133–145) S = P − Q = 2J − Σ_{i<j} n_i n_j,
so permutation p-values are identical;
Holm 1979 (Scand. J. Stat. 6:65–70); Kendall's W (Kendall & Babington Smith 1939; no tie correction, negligible for
continuous means); rank-biserial r = 2U/(n₁n₂) − 1 = (U₁ − U₂)/(n₁n₂) (Kerby 2014, Comprehensive Psychology 3,
doi:10.2466/11.IT.3.1); percentile bootstrap — with 3 runs per group the 95 % interval of a mean is the range of the runs
(true coverage ≈ 75 %; Hesterberg 2015, https://arxiv.org/abs/1411.5279); scipy.stats mannwhitneyu (method='exact'
ignores ties: audit 4 found no tied values in any reported comparison),
wilcoxon, spearmanr (https://docs.scipy.org/doc/scipy/reference/stats.html).
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact permutations for tiny samples
import math  # factorials

import numpy as np  # arrays, RNG
from scipy import stats  # reference tests

RNG_SEED = 0  # all resampling uses this seed (reproducibility)


def jt_statistic(groups: list[np.ndarray]) -> float:
    """Jonckheere–Terpstra J for groups in their hypothesised increasing order."""
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            x, y = np.asarray(groups[i])[:, None], np.asarray(groups[j])[None, :]
            J += float((x < y).sum() + 0.5 * (x == y).sum())
    return J


def _splits(items: list[int], sizes: list[int]):
    """Every distinct way to put `items` into ordered groups of the given sizes (multinomial many), as index tuples.

    Local explanation: choose the first group's members with itertools.combinations, then recurse on the rest; each
    assignment of runs to explanation levels appears exactly once, which is the permutation null of the JT test.
    Source: itertools.combinations (https://docs.python.org/3/library/itertools.html#itertools.combinations)."""
    if not sizes:
        yield []
        return
    for first in itertools.combinations(items, sizes[0]):
        rest = [i for i in items if i not in first]
        for tail in _splits(rest, sizes[1:]):
            yield [first, *tail]


def _jt_all(pooled: np.ndarray, sizes: list[int]) -> np.ndarray:
    """J for every distinct assignment of the pooled values to groups of the given sizes (vectorised).

    Local explanation: one label row per assignment (group index of each value, from `_splits`); with
    C[u, v] = 1 if pooled[u] < pooled[v] (½ if equal), J = Σ_{u, v} C[u, v]·[label_u < label_v], the same count as
    `jt_statistic`. Rows are processed in chunks to bound memory."""
    n = len(pooled)
    rows = []
    for split in _splits(list(range(n)), sizes):
        lab = np.empty(n, dtype=np.int8)
        for g, idx in enumerate(split):
            lab[list(idx)] = g
        rows.append(lab)
    L = np.array(rows)
    C = (pooled[:, None] < pooled[None, :]) + 0.5 * (pooled[:, None] == pooled[None, :])
    out = [(C * (L[i:i + 20000, :, None] < L[i:i + 20000, None, :])).sum(axis=(1, 2)) for i in range(0, len(L), 20000)]
    return np.concatenate(out)


def jonckheere(groups: list[np.ndarray], n_perm: int = 20000, max_exact: int = 400000) -> dict:
    """One-sided JT test for an increasing trend; p-value by exact or Monte-Carlo permutation of group labels.

    Exact (every distinct assignment of the pooled values to groups) when there are at most `max_exact` assignments,
    e.g. 3 levels × 3 runs = 1,680 and 4 levels × 3 runs = 369,600; Monte-Carlo with n_perm label shuffles otherwise.
    Audit 2026-11-22 (second pass): the exact branch used to enumerate index permutations only for ≤ 8 values, so the
    9-value case fell back to Monte-Carlo despite this docstring; it now enumerates assignments with `_splits`.
    Audit 3 (deviation 10): the exact limit was 50,000, so H1's 4 × 3 design used Monte-Carlo (p = 1.0e-4, one hit in
    20,000); the vectorised `_jt_all` makes it exact."""
    sizes = [len(g) for g in groups]
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    obs = jt_statistic(groups)
    n_orderings = math.factorial(len(pooled)) // math.prod(math.factorial(s) for s in sizes)
    cuts = np.cumsum(sizes)[:-1]
    if n_orderings <= max_exact:  # exact: enumerate distinct assignments of values to groups
        J = _jt_all(pooled, sizes)
        return {"J": obs, "p": float((J >= obs - 1e-12).mean()), "method": "exact", "orderings": n_orderings}
    rng = np.random.default_rng(RNG_SEED)
    hits = sum(jt_statistic(np.split(rng.permutation(pooled), cuts)) >= obs - 1e-12 for _ in range(n_perm))
    return {"J": obs, "p": (hits + 1) / (n_perm + 1), "method": f"monte-carlo {n_perm}", "orderings": n_orderings}


def spearman_ci(levels: np.ndarray, values: np.ndarray, groups: list[np.ndarray] | None = None, n_boot: int = 10000) -> dict:
    """Spearman ρ between level index and value; 95 % bootstrap CI resampling runs within each level (needs ≥ 2 per level)."""
    rho = float(stats.spearmanr(levels, values).statistic)
    if groups is None or min(len(g) for g in groups) < 2:
        return {"rho": rho, "ci": None}
    rng = np.random.default_rng(RNG_SEED)
    lv = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    boots = []
    for _ in range(n_boot):
        vals = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        r = stats.spearmanr(lv, vals).statistic
        if np.isfinite(r):
            boots.append(r)
    return {"rho": rho, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def mann_whitney_greater(x: np.ndarray, y: np.ndarray) -> dict:
    """One-sided Mann–Whitney U (x > y) with rank-biserial effect size; None if x has < 2 runs or y has < 1 (the callers
    report 'cannot tell' whenever the smallest attainable p is ≥ α). scipy's 'exact' method assumes no ties."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 1:
        return {"p": None, "U": None, "r_rb": None}
    res = stats.mannwhitneyu(x, y, alternative="greater", method="exact" if len(x) * len(y) <= 400 else "asymptotic")
    r_rb = 2 * res.statistic / (len(x) * len(y)) - 1  # rank-biserial correlation
    return {"p": float(res.pvalue), "U": float(res.statistic), "r_rb": float(r_rb)}


def mann_whitney_two_sided(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided Mann–Whitney U (no predicted direction; exact for small groups) with the rank-biserial effect size of x
    over y and the smallest p-value these group sizes allow (complete separation either way)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 1 or len(y) < 1:
        return {"p": None, "U": None, "r_rb": None, "min_p": 1.0}
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact" if len(x) * len(y) <= 400 else "asymptotic")
    return {"p": float(res.pvalue), "U": float(res.statistic), "r_rb": float(2 * res.statistic / (len(x) * len(y)) - 1),
            "min_p": min_p_mann_whitney(len(x), len(y), two_sided=True)}


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values (None entries are kept as None and excluded)."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m, out, running = len(items), {k: None for k in pvals}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def bootstrap_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000) -> dict:
    """Difference of means x − y with a 95 % bootstrap CI (None if either group has < 2 runs)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = float(x.mean() - y.mean())
    if len(x) < 2 or len(y) < 2:
        return {"diff": d, "ci": None}
    rng = np.random.default_rng(RNG_SEED)
    b = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    return {"diff": d, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def kendalls_w(rank_matrix: np.ndarray, n_perm: int = 20000) -> dict:
    """Kendall's W for m raters × n items (rows = raters, values = scores); permutation p-value shuffling within raters."""
    R = np.apply_along_axis(stats.rankdata, 1, np.asarray(rank_matrix, float))
    m, n = R.shape

    def w(r):
        s = r.sum(0)
        return 12 * ((s - s.mean()) ** 2).sum() / (m**2 * (n**3 - n))
    obs = w(R)
    rng = np.random.default_rng(RNG_SEED)
    hits = sum(w(np.array([rng.permutation(row) for row in R])) >= obs - 1e-12 for _ in range(n_perm))
    return {"W": float(obs), "p": (hits + 1) / (n_perm + 1)}


def wilcoxon_greater(d: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank test that d > 0 (None if fewer than 2 values); min_p = smallest p this n allows."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return {"p": None, "median": float(np.median(d)) if len(d) else None, "min_p": 1.0}
    return {"p": float(stats.wilcoxon(d, alternative="greater").pvalue), "median": float(np.median(d)),
            "min_p": min_p_wilcoxon(int((d != 0).sum()))}  # scipy drops zero differences (zero_method='wilcox')


def min_p_wilcoxon(n: int) -> float:
    """Smallest one-sided exact Wilcoxon signed-rank p-value with n non-zero differences: all positive → 1/2^n."""
    return 0.5 ** n


def min_p_mann_whitney(n1: int, n2: int, two_sided: bool = False) -> float:
    """Smallest exact Mann–Whitney p-value for group sizes n1, n2 (complete separation): 1/C(n1+n2, n1) one-sided,
    twice that two-sided (capped at 1)."""
    from math import comb
    p = 1.0 / comb(n1 + n2, n1) if n1 and n2 else 1.0
    return min(1.0, 2 * p) if two_sided else p


def verdict(p: float | None, effect_positive: bool, min_p: float, alpha: float = 0.05) -> str:
    """Preregistered verdict wording: a test whose smallest attainable p is ≥ alpha cannot decide at this N."""
    if p is None or min_p >= alpha:
        return "cannot tell at this N"
    return "supported" if p < alpha and effect_positive else "not supported"
