"""Audit 7: re-derive every round-2 verdict from the raw records with numpy and the standard library only.

Local explanation: reads results/runs/*.json, results/baselines/T0.json, results/compiler/*/*.json and
results/controls/*.json, recomputes each preregistered test (§5 of reports/preregistration.md) by full enumeration
(Jonckheere–Terpstra over all distinct assignments, Mann–Whitney over all splits, Wilcoxon over all sign flips,
bootstrap intervals from the exact bootstrap distribution), and compares each result with
results/hypotheses_round2.json. No project code and no scipy are imported, so an error in either cannot carry over.
Global context: seventh audit (2026-11-22), written without reading the earlier audit scripts; the output
results/audit7/verdicts.json is cited by reports/reviews/audit-7-2026-11-22.md.
Sources: Jonckheere–Terpstra statistic J = Σ_{i<j} #{x ∈ G_i, y ∈ G_j: x < y} (Terpstra 1952; Jonckheere 1954);
Mann–Whitney U (exact null = all C(n1+n2, n1) splits equally likely); Wilcoxon signed-rank (exact null = all 2^n sign
patterns); Holm 1979 step-down; Spearman ρ = Pearson correlation of ranks; percentile bootstrap (Efron & Tibshirani
1993, ch. 13).
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact enumeration of assignments, splits, sign patterns
import json  # results files
import math  # binomial coefficients
from pathlib import Path  # paths

import numpy as np  # arithmetic

ROOT = Path(__file__).resolve().parents[2]  # repository root (scripts/audit7/ → ../..)
RES = ROOT / "results"  # raw records
MAIN = "claude-haiku-5-5"  # main simulator (amendment 2)


# ---------------------------------------------------------------------------------------------------------------
# Loading: main-condition runs = main simulator, default sampling, no variant, round 1 or 2
# ---------------------------------------------------------------------------------------------------------------
def runs() -> list[dict]:
    """Every graded agent run of rounds 1–2 (stage-0 dev runs have stage 'dev' and round 0)."""
    out = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]
    return [r for r in out if r["stage"] != "dev" and r["round"] in (1, 2)]


def cond(rs: list[dict], target: str, level: str, key: str = "hs") -> np.ndarray:
    """Scores of one main-simulator condition (no variant) on one target."""
    sel = [r[key] for r in rs if r["target"] == target and r["level"] == level and r["model"] == MAIN
           and r.get("variant", "") == "" and r.get("temperature") is None]
    return np.array(sel, float)


# ---------------------------------------------------------------------------------------------------------------
# Exact tests by enumeration
# ---------------------------------------------------------------------------------------------------------------
def jt(groups: list[np.ndarray]) -> tuple[float, float]:
    """One-sided JT test for an increasing trend: (J, exact p) over every distinct assignment of values to groups."""
    def J(gs):  # count of ordered pairs across groups that follow the hypothesised order (ties ½)
        return sum(((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
                   for i, a in enumerate(gs) for b in gs[i + 1:])
    obs = J(groups)
    pooled, sizes = np.concatenate(groups), [len(g) for g in groups]
    hits = tot = 0
    def rec(idx, sz, acc):  # recursive choice of each group's members from the remaining indices
        nonlocal hits, tot
        if not sz:
            tot += 1
            hits += J([pooled[list(g)] for g in acc]) >= obs - 1e-12
            return
        for first in itertools.combinations(idx, sz[0]):
            rec([i for i in idx if i not in first], sz[1:], acc + [first])
    rec(list(range(len(pooled))), sizes, [])
    return float(obs), hits / tot


def mw_greater(x: np.ndarray, y: np.ndarray) -> float:
    """Exact one-sided Mann–Whitney p (x > y): share of all splits whose U is at least the observed U."""
    pooled, n = np.concatenate([x, y]), len(x)
    U = lambda a, b: float((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())  # noqa: E731
    obs, hits, tot = U(x, y), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
        rest = [i for i in range(len(pooled)) if i not in idx]
        hits += U(pooled[list(idx)], pooled[rest]) >= obs - 1e-12
        tot += 1
    return hits / tot


def mw_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Exact two-sided Mann–Whitney p = min(1, 2·min(one-sided p in each direction))."""
    return min(1.0, 2 * min(mw_greater(x, y), mw_greater(y, x)))


def wilcoxon_greater(d: np.ndarray) -> float:
    """Exact one-sided Wilcoxon signed-rank p (d > 0) over all 2^n sign patterns (no zeros or ties expected)."""
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    obs = ranks[d > 0].sum()
    pats = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=len(d))]
    return float(np.mean(np.array(pats) >= obs - 1e-12))


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order, out, run = sorted(ps, key=ps.get), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = Pearson correlation of average ranks."""
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(1, len(v) + 1)
        for val in np.unique(v):  # average ranks over ties
            m = v == val
            r[m] = r[m].mean()
        return r
    a, b = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else float("nan")


def boot_diff_exact(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Exact 2.5 % / 97.5 % percentiles of the bootstrap distribution of mean(x*) − mean(y*) (all n^n resamples each)."""
    mx = [np.mean(c) for c in itertools.product(x, repeat=len(x))]
    my = [np.mean(c) for c in itertools.product(y, repeat=len(y))]
    d = np.sort(np.subtract.outer(mx, my).ravel())
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def spearman_boot_min(groups: list[np.ndarray]) -> float:
    """Smallest Spearman ρ(level, value) over every within-level bootstrap resample (a bound on the CI's lower end)."""
    lv = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    per = [list(itertools.product(g, repeat=len(g))) for g in groups]
    return min(spearman(lv, np.concatenate(c)) for c in itertools.product(*per))


# ---------------------------------------------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------------------------------------------
def main() -> dict:
    rs = runs()
    stored = json.loads((RES / "hypotheses_round2.json").read_text(encoding="utf-8"))["hypotheses"]
    out, checks = {}, []

    def check(name, mine, theirs, tol=1e-9):
        ok = (mine == theirs) if isinstance(mine, str) else abs(float(mine) - float(theirs)) <= tol
        checks.append({"check": name, "audit7": mine, "stored": theirs, "agree": bool(ok)})

    # H1 — HS and −KL trends over E1 < E2 < E3 < E4 on T0
    L = ["E1", "E2", "E3", "E4"]
    g_hs = [cond(rs, "T0", lv) for lv in L]
    g_kl = [-cond(rs, "T0", lv, "kl_out") for lv in L]
    J_hs, p_hs = jt(g_hs)
    J_kl, p_kl = jt(g_kl)
    lv_idx = np.concatenate([np.full(3, i) for i in range(4)])
    rho_hs, rho_kl = spearman(lv_idx, np.concatenate(g_hs)), spearman(lv_idx, np.concatenate(g_kl))
    rho_hs_min = spearman_boot_min(g_hs)
    out["H1"] = {"n_per_level": [len(g) for g in g_hs], "J_hs": J_hs, "p_hs": p_hs, "rho_hs": rho_hs,
                 "rho_hs_min_over_all_resamples": rho_hs_min, "J_neg_kl": J_kl, "p_neg_kl": p_kl, "rho_neg_kl": rho_kl,
                 "verdict_hs": "supported" if p_hs < 0.05 and rho_hs_min > 0 else "not supported",
                 "verdict_kl": "supported" if p_kl < 0.05 and rho_kl > 0 else "not supported",
                 "e0_mean": float(cond(rs, "T0", "E0").mean()), "e1_mean": float(g_hs[0].mean())}
    check("H1 JT p (HS)", p_hs, stored["H1"]["jt_hs"]["p"])
    check("H1 JT p (−KL)", p_kl, stored["H1"]["jt_neg_kl"]["p"])
    check("H1 Spearman ρ (HS)", rho_hs, stored["H1"]["spearman_hs"]["rho"])
    check("H1 verdict HS", out["H1"]["verdict_hs"], stored["H1"]["verdicts"]["HS trend E1→E4"])
    check("H1 verdict KL", out["H1"]["verdict_kl"], stored["H1"]["verdicts"]["KL_out"])

    # H2 — four one-sided Mann–Whitney tests, Holm; 3 vs 3 can reach at best p = 1/20
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"),
             "E4>W-scramble": ("E4", "W-scramble")}
    p2 = {k: mw_greater(cond(rs, "T0", a), cond(rs, "T0", b)) for k, (a, b) in pairs.items()}
    a2 = holm(p2)
    out["H2"] = {}
    for k, (a, b) in pairs.items():
        x, y = cond(rs, "T0", a), cond(rs, "T0", b)
        min_p = 1 / math.comb(len(x) + len(y), len(x))
        out["H2"][k] = {"p": p2[k], "holm_p": a2[k], "min_p": min_p, "diff": float(x.mean() - y.mean()),
                        "exact_boot_ci": boot_diff_exact(x, y),
                        "verdict_dev1": "cannot tell at this N" if min_p >= 0.05 else ("supported" if a2[k] < 0.05 else "not supported"),
                        "verdict_literal_s5": "supported" if a2[k] < 0.05 else "not supported"}
        check(f"H2 {k} p", p2[k], stored["H2"]["tests"][k]["p"])
        check(f"H2 {k} Holm p", a2[k], stored["H2"]["holm_p"][k])
        check(f"H2 {k} verdict", out["H2"][k]["verdict_dev1"], stored["H2"]["verdicts"][k])

    # H3 — condition means (no test preregistered)
    wf_hs, wf_kl, wf_ag = cond(rs, "T0", "W-freq"), cond(rs, "T0", "W-freq", "kl_out"), cond(rs, "T0", "W-freq", "argmax_agreement")
    tm_kl = {lv: float(cond(rs, "T0", lv, "kl_out").mean()) for lv in ("E2", "E3", "E4")}
    tm_hs = {lv: float(cond(rs, "T0", lv).mean()) for lv in ("E2", "E3", "E4")}
    ok3 = wf_ag.mean() >= 0.99 and wf_kl.mean() <= max(tm_kl.values()) and all(wf_hs.mean() < v for v in tm_hs.values())
    out["H3"] = {"wfreq_agreement": float(wf_ag.mean()), "wfreq_kl": float(wf_kl.mean()), "wfreq_hs": float(wf_hs.mean()),
                 "true_kl_means": tm_kl, "true_hs_means": tm_hs,
                 "kl_within_min_max": min(tm_kl.values()) <= wf_kl.mean() <= max(tm_kl.values()),
                 "verdict": "supported" if ok3 else "not supported"}
    check("H3 verdict", out["H3"]["verdict"], stored["H3"]["verdict"])

    # H4 — E3/E4 vs nine retrains, one-sided, Holm over {E3, E4}; vs three students two-sided
    base = json.loads((RES / "baselines" / "T0.json").read_text())
    retr = np.array([x["hs"] for x in base["retrains"]], float)
    stud = np.array([x["hs"] for x in base["students"]], float)
    p4 = {lv: mw_greater(cond(rs, "T0", lv), retr) for lv in ("E3", "E4")}
    a4 = holm(p4)
    s4 = {lv: mw_two_sided(cond(rs, "T0", lv), stud) for lv in ("E3", "E4")}
    out["H4"] = {"retrain_runs": [x["run"] for x in base["retrains"]], "p": p4, "holm_p": a4,
                 "students_two_sided_p": s4, "students_holm": holm(s4),
                 "verdicts": {lv: "supported" if a4[lv] < 0.05 else "not supported" for lv in a4}}
    for lv in ("E3", "E4"):
        check(f"H4 {lv} Holm p", a4[lv], stored["H4"]["holm_p"][lv])
        check(f"H4 {lv} verdict", out["H4"]["verdicts"][lv], stored["H4"]["verdicts"][lv])
        check(f"H4 {lv} students p", s4[lv], stored["H4"]["students"][lv]["p"])

    # H4b — deterministic monotonicity on the planted series
    rows = json.loads((RES / "controls" / "planted_calibration_T0.json").read_text())["rows"]
    hs, kl = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
    ok4b = abs(hs[0] - 1) < 1e-6 and kl[0] < 1e-9 and all(a > b for a, b in zip(hs, hs[1:])) and all(a < b for a, b in zip(kl, kl[1:]))
    out["H4b"] = {"hs": hs, "kl": kl, "verdict": "supported" if ok4b else "not supported"}
    check("H4b verdict", out["H4b"]["verdict"], stored["H4b"]["verdict"])

    # H5 (main-simulator half, T0) — agent mean minus compiler HS, with exact bootstrap range of the agent mean
    out["H5"] = {}
    for lv in ("E2", "E3", "E4", "W-freq"):
        x = cond(rs, "T0", lv)
        c = json.loads((RES / "compiler" / "T0" / f"{lv}.json").read_text())["hs"]
        out["H5"][lv] = {"agent_minus_compiler": float(x.mean() - c), "agent_runs": x.tolist(), "compiler": c,
                         "compiler_inside_agent_range": bool(x.min() <= c <= x.max()) or x.mean() <= c}
        check(f"H5 {lv} agent−compiler", out["H5"][lv]["agent_minus_compiler"], stored["H5"]["agent_minus_compiler_main"][lv])

    # H7 — d = HS(T0) − max HS(wrong), exact Wilcoxon over 2^6 sign patterns
    wt = json.loads((RES / "controls" / "wrong_targets_T0.json").read_text())
    d = np.array([r["hs_T0"] - max(r["hs_wrong"].values()) for r in wt["rows"]])
    pw = wilcoxon_greater(d)
    wrong_names = sorted(wt["rows"][0]["hs_wrong"])
    out["H7"] = {"wrong_targets": wrong_names, "n_wrong": len(wrong_names), "d": d.tolist(), "p": pw,
                 "verdict": "supported" if pw < 0.05 and np.median(d) > 0 else "not supported"}
    check("H7 Wilcoxon p", pw, stored["H7"]["wilcoxon"]["p"])
    check("H7 verdict", out["H7"]["verdict"], stored["H7"]["verdict"])
    for r, dd in zip(wt["rows"], d):
        check(f"H7 d {r['run_id'][-10:]}", dd, r["d"])

    # H8 — Spearman ρ(HS(t), −log restricted loss(t)) per replacement, from the stored per-checkpoint values
    tr = json.loads((RES / "controls" / "trajectory_T0.json").read_text())
    nl = -np.log(np.array(tr["restricted_loss"], float))
    rhos = {r["run_id"]: spearman(np.array(r["hs"]), nl) for r in tr["replacements"]}
    neg = {r["run_id"]: spearman(np.array(r["hs"]), nl) for r in tr["negative_controls"]}
    out["H8"] = {"rho": rhos, "min": min(rhos.values()), "median": float(np.median(list(rhos.values()))),
                 "negative_controls": neg, "verdict": "supported" if min(rhos.values()) > 0 else "not supported"}
    check("H8 min ρ", out["H8"]["min"], tr["spearman_min"])
    check("H8 median ρ", out["H8"]["median"], tr["spearman_hs_vs_neglog_restricted_loss"])
    check("H8 verdict", out["H8"]["verdict"], stored["H8"]["verdict"])

    # Replications on the seed targets (H1 over the ladder levels each target ran; H2 E3 > W-freq; H3 means)
    rep = {}
    for t, lvls in (("T0", ["E2", "E3", "E4"]), ("T1", ["E2", "E3", "E4"]), ("T2", ["E2", "E3", "E4"]),
                    ("T-new", ["E3", "E4"]), ("T-paper", ["E1", "E3", "E4"])):
        gs = [cond(rs, t, lv) for lv in lvls]
        Jt, pt = jt(gs)
        wf = cond(rs, t, "W-freq")
        trues = [lv for lv in ("E2", "E3", "E4") if len(cond(rs, t, lv))]
        kl_m = {lv: float(cond(rs, t, lv, "kl_out").mean()) for lv in trues}
        hs_m = {lv: float(cond(rs, t, lv).mean()) for lv in trues}
        ag = float(cond(rs, t, "W-freq", "argmax_agreement").mean())
        rep[t] = {"levels": lvls, "jt_p": pt, "rho": spearman(np.concatenate([np.full(len(g), i) for i, g in enumerate(gs)]), np.concatenate(gs)),
                  "e3_minus_wfreq": float(cond(rs, t, "E3").mean() - wf.mean()),
                  "mw_p_e3_wfreq": mw_greater(cond(rs, t, "E3"), wf),
                  "h3": {"agreement": ag, "kl": float(cond(rs, t, "W-freq", "kl_out").mean()), "true_kl_max": max(kl_m.values()),
                         "hs": float(wf.mean()), "holds": ag >= 0.99 and cond(rs, t, "W-freq", "kl_out").mean() <= max(kl_m.values())
                         and all(wf.mean() < v for v in hs_m.values())},
                  # agreement of the true replacements (relevant when H3 fails on agreement)
                  "true_agreement": {lv: float(cond(rs, t, lv, "argmax_agreement").mean()) for lv in trues}}
    out["replications"] = rep
    out["checks"] = checks
    out["n_checks"], out["n_agree"] = len(checks), sum(c["agree"] for c in checks)
    return out


if __name__ == "__main__":
    res = main()
    (RES / "audit7").mkdir(exist_ok=True)
    (RES / "audit7" / "verdicts.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8", newline="\n")
    print(f"{res['n_agree']} of {res['n_checks']} checks agree")
    for c in res["checks"]:
        if not c["agree"]:
            print("DISAGREE", c)
