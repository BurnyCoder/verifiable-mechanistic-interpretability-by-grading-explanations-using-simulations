"""Audit 4 (2026-11-22): independent recomputation of every preregistered hypothesis test and verdict.

Local explanation: reads the per-run score records (`results/runs/*.json`) and the no-agent result files, and recomputes
H1–H8 and the seed-target replications with code that shares nothing with `vmi.analysis.stats`:
- Jonckheere–Terpstra p-values from the exact null distribution of J, which without ties is the coefficient sequence of
  the Gaussian (q-)multinomial coefficient [N; n_1, …, n_k]_q (J counts cross-group pairs in increasing order, i.e. the
  non-inversions of a random multiset permutation);
- Mann–Whitney and Wilcoxon p-values by enumerating every relabelling / sign flip;
- Holm by its step-down definition; bootstrap with an independent RNG stream.
It then applies the verdict rules of preregistration §5–6 as amended by deviations 1, 7, 8 and 10, and compares every
statistic and verdict with `results/hypotheses_round2.json` and the replication table generated from the same runs.
Global context: output `results/audit4/hypotheses.json`, read by `reports/reviews/audit-4-2026-11-22.md`.
Sources: JT statistic — Terpstra 1952 (Indag. Math. 14:327–333), Jonckheere 1954 (Biometrika 41:133–145); null
distribution of inversions of multiset permutations = Gaussian multinomial coefficients — Andrews, *The Theory of
Partitions* (1976), Thm 3.6 (q-binomial as an inversion generating function; the multinomial is a product of
q-binomials); Holm 1979 (Scand. J. Stat. 6:65–70); exact rank tests by complete enumeration — Lehmann, *Nonparametrics*
(1975), ch. 1–3; scipy.stats.rankdata for Spearman's average ranks
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).
Run: `.venv/Scripts/python.exe scripts/audit4/hypotheses.py`.
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact enumeration of relabellings and sign flips
import json  # results files
from pathlib import Path  # paths

import numpy as np  # arrays
from scipy.stats import rankdata  # average ranks for Spearman ρ

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "audit4" / "hypotheses.json"
ALPHA = 0.05
MAIN = "claude-haiku-5-5"


# ------------------------------------------------------------------------------------------------
# Exact tests, written from their definitions
# ------------------------------------------------------------------------------------------------
def q_binomial(n: int, k: int) -> np.ndarray:
    """Coefficients of the Gaussian binomial [n choose k]_q (index = power of q), by Pascal's q-recurrence
    [n,k] = [n−1,k−1] + q^k [n−1,k]."""
    table = {(0, 0): np.array([1], dtype=object)}
    for m in range(1, n + 1):
        for j in range(0, min(m, k) + 1):
            a = table.get((m - 1, j - 1), np.array([0], dtype=object)) if j >= 1 else np.array([0], dtype=object)
            b = table.get((m - 1, j), np.array([0], dtype=object)) if j <= m - 1 else np.array([0], dtype=object)
            b = np.concatenate([np.zeros(j, dtype=object), b])  # multiply by q^j
            size = max(len(a), len(b))
            table[(m, j)] = np.pad(a, (0, size - len(a))) + np.pad(b, (0, size - len(b)))
    return table[(n, k)]


def jt_null(sizes: list[int]) -> np.ndarray:
    """Exact null counts of J (index = J) for group sizes, no ties: [N; n_1..n_k]_q as a product of q-binomials."""
    poly, left = np.array([1], dtype=object), sum(sizes)
    for s in sizes[:-1]:
        poly = np.convolve(poly, q_binomial(left, s))
        left -= s
    return poly


def jt(groups: list[np.ndarray]) -> dict:
    """One-sided JT test for an increasing trend; exact p = P(J ≥ J_obs) under random assignment (ties: ½ each)."""
    J = sum(float((np.asarray(groups[i])[:, None] < np.asarray(groups[j])[None, :]).sum()
                  + 0.5 * (np.asarray(groups[i])[:, None] == np.asarray(groups[j])[None, :]).sum())
            for i in range(len(groups)) for j in range(i + 1, len(groups)))
    pooled = np.concatenate(groups)
    counts = jt_null([len(g) for g in groups])
    ties = len(np.unique(pooled)) < len(pooled)
    p = float(sum(counts[int(np.ceil(J - 1e-9)):]) / sum(counts))
    return {"J": J, "p": p, "ties": ties, "orderings": int(sum(counts)), "min_p": float(1 / sum(counts))}


def spearman(levels: np.ndarray, values: np.ndarray) -> float:
    """Spearman ρ = Pearson correlation of average ranks."""
    a, b = rankdata(levels), rankdata(values)
    return float(np.corrcoef(a, b)[0, 1])


def spearman_boot_ci(groups: list[np.ndarray], n_boot: int = 10000, seed: int = 20261122) -> list[float]:
    """95 % percentile CI of ρ, resampling runs with replacement within each level (§5)."""
    rng = np.random.default_rng(seed)
    lv = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    rs = []
    for _ in range(n_boot):
        v = np.concatenate([rng.choice(g, len(g)) for g in groups])
        if np.ptp(v) > 0:
            rs.append(spearman(lv, v))
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]


def mw_exact(x: np.ndarray, y: np.ndarray, two_sided: bool = False) -> dict:
    """Mann–Whitney by enumeration of all C(n1+n2, n1) relabellings; U = #{x_i > y_j} + ½ ties; one-sided P(U ≥ U_obs)."""
    pooled = np.concatenate([x, y])
    n1 = len(x)
    U = lambda a, b: float((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())  # noqa: E731
    u_obs = U(x, y)
    us = []
    for idx in itertools.combinations(range(len(pooled)), n1):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        us.append(U(pooled[m], pooled[~m]))
    us = np.array(us)
    mean = n1 * len(y) / 2
    p = float(np.mean(np.abs(us - mean) >= abs(u_obs - mean) - 1e-9)) if two_sided else float(np.mean(us >= u_obs - 1e-9))
    return {"U": u_obs, "p": p, "r_rb": 2 * u_obs / (n1 * len(y)) - 1, "min_p": float((2 if two_sided else 1) / len(us))}


def wilcoxon_exact(d: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank (d > 0) by enumerating all 2^n sign flips of the ranked |d| (zeros dropped)."""
    d = d[d != 0]
    r = rankdata(np.abs(d))
    w_obs = float(r[d > 0].sum())
    ws = [float(sum(r[i] for i in range(len(d)) if s[i])) for s in itertools.product([0, 1], repeat=len(d))]
    return {"W+": w_obs, "p": float(np.mean(np.array(ws) >= w_obs - 1e-9)), "min_p": 0.5 ** len(d), "median": float(np.median(d))}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm adjusted p: sort ascending, multiply the i-th smallest by (m − i), enforce monotonicity, cap at 1."""
    order = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[k]))
        out[k] = run
    return out


def boot_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 20261122) -> list[float]:
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def rule(p: float, positive: bool, min_p: float) -> str:
    """§6 / deviation 1: 'cannot tell' when the smallest attainable p ≥ α; else supported iff p < α and direction holds."""
    return "cannot tell at this N" if min_p >= ALPHA else ("supported" if (p < ALPHA and positive) else "not supported")


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
RUNS = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]
RUNS = [r for r in RUNS if r.get("stage") != "dev" and 1 <= int(r.get("round", 0)) <= 2]


def sel(level: str, target: str = "T0", key: str = "hs") -> np.ndarray:
    """Main-simulator runs of one unvaried condition (Claude Haiku 5.5, default sampling, no variant)."""
    return np.array([r[key] for r in RUNS if r["target"] == target and r["level"] == level and r["model"] == MAIN
                     and r.get("temperature") is None and r.get("variant", "") == ""], float)


def main() -> None:
    stored = json.loads((RES / "hypotheses_round2.json").read_text(encoding="utf-8"))["hypotheses"]
    A, cmp = {}, []

    def check(name: str, mine, theirs, tol: float = 1e-9) -> None:
        ok = (mine == theirs) if isinstance(mine, str) else (theirs is not None and abs(float(mine) - float(theirs)) <= tol)
        cmp.append({"item": name, "audit": mine, "stored": theirs, "match": bool(ok)})

    # H1
    lv = ["E1", "E2", "E3", "E4"]
    g = [sel(x) for x in lv]
    gkl = [-sel(x, key="kl_out") for x in lv]
    lvi = np.concatenate([np.full(len(x), i) for i, x in enumerate(g)])
    j_hs, j_kl = jt(g), jt(gkl)
    r_hs, r_kl = spearman(lvi, np.concatenate(g)), spearman(lvi, np.concatenate(gkl))
    ci_hs, ci_kl = spearman_boot_ci(g), spearman_boot_ci(gkl)
    v_hs = "supported" if (j_hs["p"] < ALPHA and r_hs > 0 and ci_hs[0] > 0) else "not supported"
    v_kl = "supported" if (j_kl["p"] < ALPHA and r_kl > 0 and ci_kl[0] > 0) else "not supported"
    e0 = sel("E0")
    A["H1"] = {"jt_hs": j_hs, "jt_neg_kl": j_kl, "rho_hs": r_hs, "ci_hs": ci_hs, "rho_neg_kl": r_kl, "ci_neg_kl": ci_kl,
               "verdict_hs": v_hs, "verdict_kl": v_kl, "e0_mean": float(e0.mean()), "e1_mean": float(g[0].mean()),
               "e0_below_e1": bool(e0.mean() < g[0].mean()), "n_per_level": [len(x) for x in g]}
    h1 = stored["H1"]
    check("H1 JT p (HS)", j_hs["p"], h1["jt_hs"]["p"], 1e-12)
    check("H1 JT J (HS)", j_hs["J"], h1["jt_hs"]["J"])
    check("H1 JT p (−KL)", j_kl["p"], h1["jt_neg_kl"]["p"], 1e-12)
    check("H1 Spearman ρ (HS)", r_hs, h1["spearman_hs"]["rho"], 1e-12)
    check("H1 Spearman ρ (−KL)", r_kl, h1["spearman_neg_kl"]["rho"], 1e-12)
    check("H1 ρ CI lower (HS), Monte-Carlo, tol 0.03", ci_hs[0], h1["spearman_hs"]["ci"][0], 0.03)
    check("H1 ρ CI (−KL) contains 0 in both", str(ci_kl[0] < 0 < ci_kl[1]), str(h1["spearman_neg_kl"]["ci"][0] < 0 < h1["spearman_neg_kl"]["ci"][1]))
    check("H1 verdict HS", v_hs, h1["verdicts"]["HS trend E1→E4"])
    check("H1 verdict KL", v_kl, h1["verdicts"]["KL_out"])
    check("H1 E0 clause", "not observed" if not A["H1"]["e0_below_e1"] else "observed", h1["verdicts"]["E0 below E1"].split(" (")[0])

    # H2
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"), "E4>W-scramble": ("E4", "W-scramble")}
    t2 = {k: mw_exact(sel(a), sel(b)) for k, (a, b) in pairs.items()}
    adj = holm({k: v["p"] for k, v in t2.items()})
    A["H2"] = {}
    for k, (a, b) in pairs.items():
        d = float(sel(a).mean() - sel(b).mean())
        v = rule(adj[k], d > 0, t2[k]["min_p"])
        A["H2"][k] = {**t2[k], "holm_p": adj[k], "diff": d, "boot_ci": boot_diff(sel(a), sel(b)), "verdict": v}
        check(f"H2 {k} p", t2[k]["p"], stored["H2"]["tests"][k]["p"], 1e-12)
        check(f"H2 {k} Holm p", adj[k], stored["H2"]["holm_p"][k], 1e-12)
        check(f"H2 {k} r_rb", t2[k]["r_rb"], stored["H2"]["tests"][k]["r_rb"], 1e-12)
        check(f"H2 {k} verdict", v, stored["H2"]["verdicts"][k])

    # H3
    wf_hs, wf_kl, wf_ag = sel("W-freq").mean(), sel("W-freq", key="kl_out").mean(), sel("W-freq", key="argmax_agreement").mean()
    tk = {x: sel(x, key="kl_out").mean() for x in ("E2", "E3", "E4")}
    th = {x: sel(x).mean() for x in ("E2", "E3", "E4")}
    v3 = "supported" if (wf_ag >= 0.99 and wf_kl <= max(tk.values()) and all(wf_hs < v for v in th.values())) else "not supported"
    A["H3"] = {"wf": {"hs": wf_hs, "kl": wf_kl, "agreement": wf_ag}, "true_kl": tk, "true_hs": th, "verdict": v3,
               "kl_within_range": bool(min(tk.values()) <= wf_kl <= max(tk.values()))}
    check("H3 verdict", v3, stored["H3"]["verdict"])
    check("H3 W-freq agreement", wf_ag, stored["H3"]["wfreq"]["argmax_agreement"], 1e-12)

    # H4
    base = json.loads((RES / "baselines" / "T0.json").read_text(encoding="utf-8"))
    retr = np.array([x["hs"] for x in base["retrains"]])
    stud = np.array([x["hs"] for x in base["students"]])
    t4 = {x: mw_exact(sel(x), retr) for x in ("E3", "E4")}
    adj4 = holm({k: v["p"] for k, v in t4.items()})
    t4s = {x: mw_exact(sel(x), stud, two_sided=True) for x in ("E3", "E4")}
    A["H4"] = {x: {**t4[x], "holm_p": adj4[x], "verdict": rule(adj4[x], True, t4[x]["min_p"]),
                   "students_two_sided": t4s[x]} for x in ("E3", "E4")}
    A["H4"]["n_retrains"] = len(retr)
    for x in ("E3", "E4"):
        check(f"H4 {x} p", t4[x]["p"], stored["H4"]["tests"][x]["p"], 1e-12)
        check(f"H4 {x} verdict", A["H4"][x]["verdict"], stored["H4"]["verdicts"][x])
        check(f"H4 {x} students p (two-sided)", t4s[x]["p"], stored["H4"]["students"][x]["p"], 1e-12)

    # H4b
    rows = json.loads((RES / "controls" / "planted_calibration_T0.json").read_text(encoding="utf-8"))["rows"]
    hs4b, kl4b = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
    ok = abs(hs4b[0] - 1) < 1e-6 and kl4b[0] < 1e-9 and all(np.diff(hs4b) < 0) and all(np.diff(kl4b) > 0)
    A["H4b"] = {"hs": hs4b, "kl": kl4b, "snapshot_accuracy": [r["snapshot_accuracy"] for r in rows],
                "hs_strictly_decreasing": bool(all(np.diff(hs4b) < 0)), "kl_strictly_increasing": bool(all(np.diff(kl4b) > 0)),
                "verdict": "supported" if ok else "not supported"}
    check("H4b verdict", A["H4b"]["verdict"], stored["H4b"]["verdict"])

    # H5 (descriptive half on T0): agent mean − compiler HS
    A["H5"] = {}
    for x in ("E2", "E3", "E4", "W-freq"):
        c = json.loads((RES / "compiler" / "T0" / f"{x}.json").read_text(encoding="utf-8"))["hs"]
        A["H5"][x] = {"agent_minus_compiler": float(sel(x).mean() - c), "agent_runs": sel(x).tolist(), "compiler": c,
                      "agent_range_contains_compiler": bool(sel(x).min() <= c <= sel(x).max())}
        check(f"H5 {x} agent − compiler", A["H5"][x]["agent_minus_compiler"], stored["H5"]["agent_minus_compiler_main"][x], 1e-12)

    # H7
    wt = json.loads((RES / "controls" / "wrong_targets_T0.json").read_text(encoding="utf-8"))
    d = np.array([r["hs_T0"] - max(r["hs_wrong"].values()) for r in wt["rows"]])
    w7 = wilcoxon_exact(d)
    v7 = rule(w7["p"], w7["median"] > 0, w7["min_p"])
    A["H7"] = {"d": d.tolist(), **w7, "verdict": v7, "runs": [r["run_id"] for r in wt["rows"]]}
    check("H7 Wilcoxon p", w7["p"], stored["H7"]["wilcoxon"]["p"], 1e-12)
    check("H7 verdict", v7, stored["H7"]["verdict"])
    check("H7 d recomputed from stored hs", float(np.abs(d - np.array([r["d"] for r in wt["rows"]])).max()), 0.0, 1e-12)

    # H8
    tr = json.loads((RES / "controls" / "trajectory_T0.json").read_text(encoding="utf-8"))
    nl = -np.log(np.array(tr["restricted_loss"]))
    rhos = [spearman(np.array(r["hs"]), nl) for r in tr["replacements"]]
    ctrl = {c["run_id"]: spearman(np.array(c["hs"]), nl) for c in tr.get("negative_controls", [])}
    A["H8"] = {"rhos": rhos, "min": min(rhos), "median": float(np.median(rhos)), "verdict": "supported" if min(rhos) > 0 else "not supported",
               "negative_controls": ctrl, "n_checkpoints": len(nl), "steps_first_last": [tr["steps"][0], tr["steps"][-1]]}
    check("H8 min ρ", min(rhos), stored["H8"]["spearman_min"], 1e-12)
    check("H8 verdict", A["H8"]["verdict"], stored["H8"]["verdict"])

    # Replications (§5: H1–H3 per seed target, same tests)
    A["replications"] = {}
    for t in ("T0", "T1", "T2", "T-new", "T-paper"):
        levels = [x for x in (("E2", "E3", "E4") if t == "T0" else ("E1", "E2", "E3", "E4")) if len(sel(x, t))]
        gs = [sel(x, t) for x in levels]
        jj = jt(gs)
        lvi = np.concatenate([np.full(len(x), i) for i, x in enumerate(gs)])
        rho, ci = spearman(lvi, np.concatenate(gs)), spearman_boot_ci(gs)
        v1 = "cannot tell at this N" if jj["min_p"] >= ALPHA else ("supported" if (jj["p"] < ALPHA and rho > 0 and ci[0] > 0) else "not supported")
        e3, wf = sel("E3", t), sel("W-freq", t)
        m2 = mw_exact(e3, wf)
        true = [x for x in ("E2", "E3", "E4") if len(sel(x, t))]
        h3 = {"agreement": float(sel("W-freq", t, "argmax_agreement").mean()), "kl": float(sel("W-freq", t, "kl_out").mean()),
              "true_kl_max": float(max(sel(x, t, "kl_out").mean() for x in true)), "hs": float(wf.mean()),
              "true_hs_min": float(min(sel(x, t).mean() for x in true))}
        h3["holds"] = bool(h3["agreement"] >= 0.99 and h3["kl"] <= h3["true_kl_max"] and h3["hs"] < h3["true_hs_min"])
        A["replications"][t] = {"levels": levels, "jt": jj, "rho": rho, "ci": ci, "h1": v1, "h2_p": m2["p"],
                                "h2": rule(m2["p"], e3.mean() > wf.mean(), m2["min_p"]), "h2_diff": float(e3.mean() - wf.mean()), "h3": h3}
    A["comparison"] = cmp
    A["all_match"] = all(c["match"] for c in cmp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(A, indent=1, default=float), encoding="utf-8", newline="\n")
    for c in cmp:
        print(("OK  " if c["match"] else "DIFF"), c["item"], "| audit:", c["audit"], "| stored:", c["stored"])
    for t, r in A["replications"].items():
        print(t, r["levels"], "JT p", round(r["jt"]["p"], 4), "ρ", round(r["rho"], 2), r["h1"], "| H2 p", round(r["h2_p"], 3), r["h2"], "| H3", r["h3"]["holds"])
    print("all stored statistics and verdicts reproduced:", A["all_match"])


if __name__ == "__main__":
    main()
