"""Independent arithmetic for the 2026-11-26 audit. Inputs are saved grades; this module has no model dependencies."""

from collections import Counter
from functools import cache
from itertools import combinations, product
from math import comb

import numpy as np
from scipy.stats import rankdata


@cache
def inversion_counts(sizes):
    """Count level-label orderings by Jonckheere statistic, adding the last observation in ascending score order."""
    if not sum(sizes):
        return {0: 1}
    out = Counter()
    for level, count in enumerate(sizes):
        if count:
            left = tuple(n - (j == level) for j, n in enumerate(sizes))
            for score, ways in inversion_counts(left).items():
                out[score + sum(left[:level])] += ways
    return out


def rank_correlation(x, y):
    """Pearson correlation of average ranks, as in the definition of Spearman's coefficient."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def trend(groups):
    """Exact upper-tail JT and registered within-level percentile interval, independently of the reporting implementation."""
    groups = [np.asarray(group, dtype=float) for group in groups]
    sizes = tuple(map(len, groups))
    values = np.concatenate(groups)
    if len(set(values)) != len(values):
        raise ValueError("This audit's untied JT recurrence requires distinct recorded observations")
    j = sum(int((a[:, None] < b).sum()) for i, a in enumerate(groups) for b in groups[i + 1 :])
    distribution = inversion_counts(sizes)
    tail = sum(count for score, count in distribution.items() if score >= j)
    total = sum(distribution.values())
    levels = np.repeat(np.arange(len(groups)), sizes)
    rho = rank_correlation(levels, values)
    ci = None
    if min(sizes) > 1:
        # All eligible study arms have equal replicate counts; draw one index per original observation.
        if len(set(sizes)) != 1:
            raise ValueError("Unbalanced replication in archived trend test")
        n = sizes[0]
        indices = np.random.default_rng(0).integers(n, size=(10000, len(groups), n))
        samples = np.asarray(groups)[np.arange(len(groups))[None, :, None], indices].reshape(10000, -1)
        ranks = rankdata(samples, axis=1)
        ranks -= ranks.mean(axis=1, keepdims=True)
        fixed = rankdata(levels) - rankdata(levels).mean()
        ci = np.percentile(
            (ranks @ fixed) / (np.linalg.norm(ranks, axis=1) * np.linalg.norm(fixed)), [2.5, 97.5]
        ).tolist()
    return {
        "J": j,
        "upper_tail": tail,
        "assignments": total,
        "p": tail / total,
        "rho": rho,
        "ci": ci,
        "supported": tail / total < 0.05 and rho > 0 and (ci is None or ci[0] > 0),
    }


def ranksum(x, y, two_sided=False):
    """Enumerate all label allocations of the observed pooled ranks, including tied values."""
    n, m = len(x), len(y)
    ranks = rankdata(list(x) + list(y))
    observed = sum(ranks[:n])
    sums = np.array([sum(ranks[list(s)]) for s in combinations(range(n + m), n)])
    upper = float(np.mean(sums >= observed))
    p = min(1.0, 2 * min(upper, float(np.mean(sums <= observed)))) if two_sided else upper
    u = float(observed - n * (n + 1) / 2)
    return {
        "p": p,
        "U": u,
        "rank_biserial": 2 * u / (n * m) - 1,
        "min_p_untied": min(1.0, (1 + two_sided) / comb(n + m, n)),
    }


def holm(raw):
    """Holm adjusted probabilities: enforce monotonicity of the ordered sequential products."""
    keys = sorted(raw, key=raw.get)
    adjusted = np.maximum.accumulate([raw[k] * (len(keys) - i) for i, k in enumerate(keys)])
    return {k: min(1.0, float(p)) for k, p in zip(keys, adjusted)}


def difference(x, y):
    """Empirical mean difference and its exact percentile bootstrap at the recorded small arm sizes."""
    d = float(np.mean(x) - np.mean(y))
    if min(len(x), len(y)) == 1:
        return {"mean_difference": d, "ci": None}
    means = lambda g: np.asarray([np.mean(sample) for sample in product(g, repeat=len(g))])
    bs = (means(x)[:, None] - means(y)).ravel()
    return {"mean_difference": d, "ci": np.quantile(bs, [0.025, 0.975], method="inverted_cdf").tolist()}


def signed_rank(margins):
    """Enumerate all signs of the nonzero paired margins; exact one-sided Wilcoxon null under sign symmetry."""
    d = np.asarray([v for v in margins if v])
    ranks = rankdata(np.abs(d))
    w = ranks[d > 0].sum()
    return float(
        np.mean(
            [sum(r for r, sign in zip(ranks, signs) if sign) >= w for signs in product((False, True), repeat=len(d))]
        )
    )


def bootstrap_coverage(repetitions=20000):
    """Reproducible simulation of exhaustive three-observation percentile intervals, under independent N(0,1) arms.

    The probability that the sample range contains a fixed θ is 1−F(θ)^3−(1−F(θ))^3 for any continuous F.
    For a difference, enumerate the 27×27 resampled pairs for every simulated experiment.
    """
    rng = np.random.default_rng(20261126)
    choices = np.array(list(product(range(3), repeat=3)))
    hits = np.zeros(2, dtype=int)
    for offset in range(0, repetitions, 200):
        a, b = rng.normal(size=(2, min(200, repetitions - offset), 3))
        hits[0] += np.sum((a.min(1) <= 0) & (a.max(1) >= 0))
        boot = a[:, choices].mean(2)[:, :, None] - b[:, choices].mean(2)[:, None, :]
        lo, hi = np.quantile(boot.reshape(len(a), -1), [0.025, 0.975], axis=1, method="inverted_cdf")
        hits[1] += np.sum((lo <= 0) & (hi >= 0))
    rates = hits / repetitions
    return {
        "distribution": "two independent arms of three iid standard-normal draws",
        "seed": 20261126,
        "repetitions": repetitions,
        "mean": rates[0],
        "difference": rates[1],
        "monte_carlo_se": np.sqrt(rates * (1 - rates) / repetitions).tolist(),
    }
