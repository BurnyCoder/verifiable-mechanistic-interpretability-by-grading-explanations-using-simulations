"""Statistical tests used by the preregistered hypotheses (preregistration §5).

Local explanation: Jonckheere–Terpstra trend test (exact permutation when the number of distinct orderings is small,
Monte-Carlo permutation otherwise), Spearman ρ with a bootstrap CI that resamples runs within groups, one-sided
Mann–Whitney U, Holm correction, Kendall's W with a permutation p-value, Wilcoxon signed-rank, rank-biserial effect size.
Global context: `hypotheses.py` applies these to per-run scores; every function returns plain floats for JSON.
Sources: Jonckheere–Terpstra trend test — Terpstra 1952 (Indagationes Math. 14:327–333) statistic J = Σ_{i<j} #{(x, y):
x ∈ G_i, y ∈ G_j, x < y} + ½ ties, equivalent to Jonckheere's 1954 (Biometrika 41:133–145) S = P − Q = 2J − Σ_{i<j} n_i n_j,
so permutation p-values are identical;
Holm 1979 (Scand. J. Stat. 6:65–70); Kendall's W (Kendall & Babington Smith 1939); scipy.stats mannwhitneyu,
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


def jonckheere(groups: list[np.ndarray], n_perm: int = 20000) -> dict:
    """One-sided JT test for an increasing trend; p-value by exact or Monte-Carlo permutation of group labels."""
    sizes = [len(g) for g in groups]
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    obs = jt_statistic(groups)
    n_orderings = math.factorial(len(pooled)) // math.prod(math.factorial(s) for s in sizes)
    cuts = np.cumsum(sizes)[:-1]
    if n_orderings <= 50000:  # exact: enumerate distinct assignments of values to groups
        count = total = 0
        for perm in set(itertools.permutations(range(len(pooled)))) if len(pooled) <= 8 else []:
            total += 1
            count += jt_statistic(np.split(pooled[list(perm)], cuts)) >= obs - 1e-12
        if total:
            return {"J": obs, "p": count / total, "method": "exact", "orderings": n_orderings}
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
    """One-sided Mann–Whitney U (x > y) with rank-biserial effect size; undefined (None) if a group has < 2 runs... unless
    both have ≥ 1 and the exact test is requested — with 1 vs 1 the minimum p is 0.5, reported as 'cannot tell'."""
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
