"""Audit 5 (2026-11-22): independent re-evaluation of every preregistered hypothesis from the raw run records.

Local explanation: reads only JSON files under results/ (never the project's analysis code) and re-derives every
statistic and verdict of preregistration §5–§6 and its deviations 1, 5, 7, 8 and 10 with exact methods:
  - Jonckheere–Terpstra p by enumerating every distinct assignment of the pooled runs to the ordered levels;
  - Mann–Whitney and Wilcoxon signed-rank p by full enumeration (rank sums over all subsets; all 2^n sign patterns);
  - percentile-bootstrap intervals from the *exact* bootstrap distribution (every equally likely resample of three
    runs per group, 27 per group), so no random seed is involved; Holm by its step-down definition;
then compares each result with `results/hypotheses_round{r}.json` and with the replication rules, and writes
`results/audit5/hypotheses_check.json` (every comparison, its stored and recomputed value, and whether they agree).
Global context: post hoc audit; it changes no verdict. Selection of runs: main simulator Claude Haiku 5.5 at default
sampling, no variant, stage ≠ dev, round ≤ r (preregistration amendment 2, deviation 1).
Sources: Jonckheere 1954 (Biometrika 41:133–145) / Terpstra 1952; Mann & Whitney 1947 (Ann. Math. Stat. 18:50–60);
Wilcoxon 1945 (Biometrics Bulletin 1:80–83); Holm 1979 (Scand. J. Stat. 6:65–70); exact bootstrap — Efron &
Tibshirani 1993, *An Introduction to the Bootstrap*, §13; Spearman ρ via scipy.stats.rankdata
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).
Run: `.venv/Scripts/python.exe scripts/audit5/hypotheses_check.py [round]` (CPU, about one minute).
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact enumeration
import json  # results files
import sys  # round argument
import time  # timestamped log
from math import comb  # minimum attainable p-values
from pathlib import Path  # paths

import numpy as np  # arrays
from scipy.stats import rankdata  # average ranks for Spearman ρ

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "audit5" / "hypotheses_check.json"
ALPHA = 0.05  # preregistration §5
MAIN = "claude-haiku-5-5"  # amendment 2: main simulator


def log(msg: str) -> None:
    """Timestamped progress line on the terminal."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_runs(r: int) -> list[dict]:
    """Graded agent runs of rounds 1 … r, development runs excluded."""
    out = []
    for p in sorted((RES / "runs").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("stage") != "dev" and 1 <= int(d.get("round", 0)) <= r:
            out.append(d)
    return out


def vals(runs, target, level, key="hs") -> np.ndarray:
    """Per-run values of one main-simulator condition (no variant, default sampling)."""
    return np.array([d[key] for d in runs if d["target"] == target and d["level"] == level and d["model"] == MAIN
                     and d.get("temperature") is None and d.get("variant", "") == ""], float)


# ---------------------------------------------------------------------------------------------------------------
# Exact tests
# ---------------------------------------------------------------------------------------------------------------
def jt_stat(groups) -> float:
    """J = Σ_{i<j} #{x ∈ G_i, y ∈ G_j: x < y} + ½ #{x = y} (Terpstra 1952)."""
    return float(sum(((np.asarray(a)[:, None] < np.asarray(b)[None, :]).sum()
                      + 0.5 * (np.asarray(a)[:, None] == np.asarray(b)[None, :]).sum())
                     for i, a in enumerate(groups) for b in groups[i + 1:]))


def assignments(n: int, sizes: list[int]):
    """Every distinct split of indices 0 … n−1 into ordered groups of the given sizes."""
    if not sizes:
        yield []
        return
    rest_all = list(range(n)) if not isinstance(n, list) else n
    for first in itertools.combinations(rest_all, sizes[0]):
        rest = [i for i in rest_all if i not in first]
        for tail in assignments(rest, sizes[1:]):
            yield [list(first), *tail]


def jt_exact(groups) -> dict:
    """One-sided exact JT p (increasing order) over all distinct assignments."""
    pooled = np.concatenate([np.asarray(g, float) for g in groups])
    sizes = [len(g) for g in groups]
    obs = jt_stat(groups)
    js = [jt_stat([pooled[idx] for idx in split]) for split in assignments(list(range(len(pooled))), sizes)]
    js = np.array(js)
    return {"J": obs, "p": float((js >= obs - 1e-12).mean()), "orderings": int(len(js))}


def mw_exact_greater(x, y) -> dict:
    """One-sided exact Mann–Whitney p for x > y by enumerating every subset of ranks (no ties in these data)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    U = float((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum())
    pooled = np.concatenate([x, y])
    n1 = len(x)
    us = []
    for idx in itertools.combinations(range(len(pooled)), n1):
        a = pooled[list(idx)]
        b = np.delete(pooled, list(idx))
        us.append((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
    us = np.array(us)
    return {"U": U, "p": float((us >= U - 1e-12).mean()), "p_two_sided": float(min(1.0, 2 * min((us >= U - 1e-12).mean(),
                                                                                                    (us <= U + 1e-12).mean()))),
            "min_p": 1 / comb(len(pooled), n1), "r_rb": 2 * U / (len(x) * len(y)) - 1,
            "ties": int(len(np.unique(pooled)) < len(pooled))}


def wilcoxon_exact_greater(d) -> dict:
    """One-sided exact Wilcoxon signed-rank p that d > 0 over all 2^n sign patterns (zeros dropped)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    r = rankdata(np.abs(d))
    obs = float(r[d > 0].sum())
    ws = np.array([sum(rr for rr, s in zip(r, signs) if s) for signs in itertools.product([0, 1], repeat=len(d))])
    return {"W+": obs, "p": float((ws >= obs - 1e-12).mean()), "min_p": 0.5 ** len(d), "median": float(np.median(d))}


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def resamples(g) -> np.ndarray:
    """Every equally likely bootstrap resample (with replacement, ordered draws) of a group: n^n rows."""
    g = np.asarray(g, float)
    return g[np.array(list(itertools.product(range(len(g)), repeat=len(g))))]


def wquantile(v: np.ndarray, q: float) -> float:
    """Quantile of an equally weighted discrete distribution (linear interpolation, like numpy.percentile)."""
    return float(np.percentile(v, 100 * q))


def boot_mean_diff_exact(x, y) -> dict:
    """Exact percentile-bootstrap 95 % interval of mean(x) − mean(y)."""
    mx, my = resamples(x).mean(1), resamples(y).mean(1)
    d = (mx[:, None] - my[None, :]).ravel()
    return {"diff": float(np.mean(x) - np.mean(y)), "ci": [wquantile(d, 0.025), wquantile(d, 0.975)]}


def spearman(levels, values) -> float:
    """Spearman ρ = Pearson correlation of average ranks."""
    a, b = rankdata(levels), rankdata(values)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_exact_ci(groups) -> dict:
    """Spearman ρ(level index, value) with the exact percentile-bootstrap 95 % interval (resampling runs within level)."""
    lv = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    rho = spearman(lv, np.concatenate(groups))
    per = [resamples(g) for g in groups]
    rhos = []
    for combo in itertools.product(*[range(len(p)) for p in per]):
        v = np.concatenate([per[i][j] for i, j in enumerate(combo)])
        r = spearman(lv, v)
        if np.isfinite(r):
            rhos.append(r)
    rhos = np.array(rhos)
    return {"rho": rho, "ci": [wquantile(rhos, 0.025), wquantile(rhos, 0.975)], "n_resamples": int(len(rhos))}


def trend_verdict(groups) -> dict:
    """§5 H1 rule: one-sided JT p < α and Spearman ρ > 0 with its bootstrap CI above 0; 'cannot tell' when the smallest
    attainable p (1 / number of orderings) is ≥ α (§6, deviation 1)."""
    jt, sp = jt_exact(groups), spearman_exact_ci(groups)
    if 1 / jt["orderings"] >= ALPHA:
        v = "cannot tell at this N"
    else:
        v = "supported" if jt["p"] < ALPHA and sp["rho"] > 0 and sp["ci"][0] > 0 else "not supported"
    return {"verdict": v, "jt": jt, "spearman": sp}


def rank_verdict(p_adj: float, min_p: float, positive: bool) -> str:
    """Deviation 1 / 7: 'cannot tell' when the smallest attainable p is ≥ α, else supported iff p < α and direction."""
    if min_p >= ALPHA:
        return "cannot tell at this N"
    return "supported" if p_adj < ALPHA and positive else "not supported"


# ---------------------------------------------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------------------------------------------
def evaluate(r: int) -> dict:
    runs = load_runs(r)
    T = "T0"
    out = {}
    # H1 — HS trend E1<E2<E3<E4 and KL trend (decreasing), E0 clause by direction only
    hs = [vals(runs, T, lv) for lv in ("E1", "E2", "E3", "E4")]
    kl = [-vals(runs, T, lv, "kl_out") for lv in ("E1", "E2", "E3", "E4")]
    e0 = vals(runs, T, "E0")
    out["H1"] = {"HS": trend_verdict(hs), "KL": trend_verdict(kl),
                 "E0_below_E1": bool(e0.mean() < hs[0].mean()), "means_hs": [float(g.mean()) for g in hs]}
    # H2 — four one-sided Mann–Whitney, Holm, effect = exact bootstrap difference
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"),
             "E4>W-scramble": ("E4", "W-scramble")}
    t2 = {k: mw_exact_greater(vals(runs, T, a), vals(runs, T, b)) for k, (a, b) in pairs.items()}
    adj = holm({k: v["p"] for k, v in t2.items()})
    eff = {k: boot_mean_diff_exact(vals(runs, T, a), vals(runs, T, b)) for k, (a, b) in pairs.items()}
    out["H2"] = {k: {"p": t2[k]["p"], "holm_p": adj[k], "min_p": t2[k]["min_p"], "r_rb": t2[k]["r_rb"],
                     "ties": t2[k]["ties"], "effect": eff[k],
                     "verdict": rank_verdict(adj[k], t2[k]["min_p"], eff[k]["diff"] > 0)} for k in pairs}
    # H3 — W-freq: agreement ≥ 0.99, KL ≤ max of E2–E4 means (falsifier), HS below each of E2–E4 means
    wf = {k: vals(runs, T, "W-freq", k) for k in ("hs", "kl_out", "argmax_agreement")}
    true = {lv: {k: vals(runs, T, lv, k) for k in ("hs", "kl_out")} for lv in ("E2", "E3", "E4")}
    agree, klok = wf["argmax_agreement"].mean() >= 0.99, wf["kl_out"].mean() <= max(v["kl_out"].mean() for v in true.values())
    hsok = all(wf["hs"].mean() < v["hs"].mean() for v in true.values())
    out["H3"] = {"verdict": "supported" if agree and klok and hsok else "not supported", "agreement_ok": bool(agree),
                 "kl_not_above_max": bool(klok), "hs_below_all": bool(hsok),
                 "kl_within_range": bool(min(v["kl_out"].mean() for v in true.values()) <= wf["kl_out"].mean()
                                         <= max(v["kl_out"].mean() for v in true.values())),
                 "hs_diff": {lv: boot_mean_diff_exact(wf["hs"], v["hs"]) for lv, v in true.items()}}
    # H4 — E3, E4 vs 9 retrains (one-sided, Holm over 2); two-sided vs 3 students (Holm over 2)
    base = json.loads((RES / "baselines" / "T0.json").read_text())
    retr, stud = [x["hs"] for x in base["retrains"]], [x["hs"] for x in base["students"]]
    t4 = {lv: mw_exact_greater(vals(runs, T, lv), retr) for lv in ("E3", "E4")}
    a4 = holm({k: v["p"] for k, v in t4.items()})
    s4 = {lv: mw_exact_greater(vals(runs, T, lv), stud) for lv in ("E3", "E4")}
    a4s = holm({k: v["p_two_sided"] for k, v in s4.items()})
    out["H4"] = {"retrains": [x["run"] for x in base["retrains"]],
                 **{lv: {"p": t4[lv]["p"], "holm_p": a4[lv], "min_p": t4[lv]["min_p"],
                         "verdict": rank_verdict(a4[lv], t4[lv]["min_p"], True)} for lv in t4},
                 "students_two_sided": {lv: {"p": s4[lv]["p_two_sided"], "holm_p": a4s[lv], "min_p": 2 * s4[lv]["min_p"],
                                             "result": "cannot tell at this N" if 2 * s4[lv]["min_p"] >= ALPHA else
                                             ("differ" if a4s[lv] < ALPHA else "no significant difference")}
                                        for lv in s4}}
    # H4b — deterministic monotonicity
    rows = json.loads((RES / "controls" / "planted_calibration_T0.json").read_text())["rows"]
    h, k = [x["hs"] for x in rows], [x["kl_out"] for x in rows]
    ok = abs(h[0] - 1) < 1e-6 and k[0] < 1e-9 and all(a > b for a, b in zip(h, h[1:])) and all(a < b for a, b in zip(k, k[1:]))
    out["H4b"] = {"verdict": "supported" if ok else "not supported", "hs": h, "kl": k}
    # H5 (T0 part only; the capability trend is not evaluable, deviation 5): agent CI lower bound above the compiler
    h5 = {}
    for lv in ("E2", "E3", "E4", "W-freq"):
        v = vals(runs, T, lv)
        c = json.loads((RES / "compiler" / T / f"{lv}.json").read_text())["hs"]
        lo = wquantile(resamples(v).mean(1), 0.025)
        h5[lv] = {"gap": float(v.mean() - c), "agent_ci_low": lo, "beyond": bool(v.mean() > c and lo > c)}
    local = [d for d in runs if d["model"] != MAIN]
    out["H5"] = {"T0": h5, "local_runs": len(local), "local_above_chance": sum(d["accuracy"] > 2 / 113 for d in local)}
    # H7 — Wilcoxon on d = HS(T0) − max over the 12 preregistered wrong targets
    wt = json.loads((RES / "controls" / "wrong_targets_T0.json").read_text())
    d = [row["hs_T0"] - max(row["hs_wrong"].values()) for row in wt["rows"]]
    w7 = wilcoxon_exact_greater(d)
    out["H7"] = {"wrong_targets": list(wt["rows"][0]["hs_wrong"].keys()), "d": d,
                 "d_matches_stored": all(abs(x - row["d"]) < 1e-12 for x, row in zip(d, wt["rows"])), **w7,
                 "verdict": rank_verdict(w7["p"], w7["min_p"], w7["median"] > 0)}
    # H8 — supported iff every E3/E4 replacement's ρ > 0 (deviation 8)
    tr = json.loads((RES / "controls" / "trajectory_T0.json").read_text())
    from scipy.stats import spearmanr  # recompute ρ from the stored curves
    nl = -np.log(np.array(tr["restricted_loss"]))
    rhos = [float(spearmanr(np.array(x["hs"]), nl).statistic) for x in tr["replacements"] if "__E3__" in x["run_id"] or "__E4__" in x["run_id"]]
    out["H8"] = {"rhos": rhos, "n_checkpoints": len(nl), "verdict": "supported" if rhos and min(rhos) > 0 else "not supported"}
    # Replications (§5: H1–H3 per seed target with the same tests)
    rep = {}
    for t in ["T0", "T1", "T2", "T-new", "T-paper"]:
        lvls = [lv for lv in (("E2", "E3", "E4") if t == "T0" else ("E1", "E2", "E3", "E4")) if len(vals(runs, t, lv))]
        g = [vals(runs, t, lv) for lv in lvls]
        e3, w = vals(runs, t, "E3"), vals(runs, t, "W-freq")
        mw = mw_exact_greater(e3, w)
        tl = [lv for lv in ("E2", "E3", "E4") if lv in lvls]
        wa, wk, wh = (vals(runs, t, "W-freq", k).mean() for k in ("argmax_agreement", "kl_out", "hs"))
        rep[t] = {"levels": lvls, "H1": trend_verdict(g),
                  "H2": {"p": mw["p"], "verdict": rank_verdict(mw["p"], mw["min_p"], e3.mean() > w.mean())},
                  "H3": {"agreement_ok": bool(wa >= 0.99), "kl_ok": bool(wk <= max(vals(runs, t, lv, "kl_out").mean() for lv in tl)),
                         "hs_ok": bool(all(wh < vals(runs, t, lv).mean() for lv in tl))}}
    out["replications"] = rep
    return out


def compare(r: int, mine: dict) -> list[dict]:
    """Stored (project) value vs recomputed value for every verdict and test statistic."""
    H = json.loads((RES / f"hypotheses_round{r}.json").read_text())["hypotheses"]
    rows = []

    def add(name, stored, recomputed, tol=1e-9):
        same = (stored == recomputed) if isinstance(stored, str) or isinstance(recomputed, str) else abs(stored - recomputed) <= tol
        rows.append({"item": name, "stored": stored, "recomputed": recomputed, "agree": bool(same)})

    add("H1 HS JT p", H["H1"]["jt_hs"]["p"], mine["H1"]["HS"]["jt"]["p"])
    add("H1 KL JT p", H["H1"]["jt_neg_kl"]["p"], mine["H1"]["KL"]["jt"]["p"])
    add("H1 HS Spearman ρ", H["H1"]["spearman_hs"]["rho"], mine["H1"]["HS"]["spearman"]["rho"])
    add("H1 HS ρ CI lower > 0", str(H["H1"]["spearman_hs"]["ci"][0] > 0), str(mine["H1"]["HS"]["spearman"]["ci"][0] > 0))
    add("H1 verdict HS", H["H1"]["verdicts"]["HS trend E1→E4"], mine["H1"]["HS"]["verdict"])
    add("H1 verdict KL", H["H1"]["verdicts"]["KL_out"], mine["H1"]["KL"]["verdict"])
    add("H1 E0 below E1", str(H["H1"]["verdicts"]["E0 below E1"].startswith("observed")), str(mine["H1"]["E0_below_E1"]))
    for k, v in mine["H2"].items():
        add(f"H2 {k} p", H["H2"]["tests"][k]["p"], v["p"])
        add(f"H2 {k} Holm p", H["H2"]["holm_p"][k], v["holm_p"])
        add(f"H2 {k} verdict", H["H2"]["verdicts"][k], v["verdict"])
        add(f"H2 {k} CI includes 0", str(H["H2"]["effects"][k]["ci"][0] <= 0 <= H["H2"]["effects"][k]["ci"][1]),
            str(v["effect"]["ci"][0] <= 0 <= v["effect"]["ci"][1]))
    add("H3 verdict", H["H3"]["verdict"], mine["H3"]["verdict"])
    for lv, v in mine["H3"]["hs_diff"].items():
        s = H["H3"]["hs_diff"][lv]["ci"]
        add(f"H3 HS diff {lv} CI excludes 0", str(s[1] < 0), str(v["ci"][1] < 0))
    for lv in ("E3", "E4"):
        add(f"H4 {lv} p", H["H4"]["tests"][lv]["p"], mine["H4"][lv]["p"])
        add(f"H4 {lv} Holm p", H["H4"]["holm_p"][lv], mine["H4"][lv]["holm_p"])
        add(f"H4 {lv} verdict", H["H4"]["verdicts"][lv], mine["H4"][lv]["verdict"])
        add(f"H4 {lv} vs students", H["H4"]["students"][lv]["result"], mine["H4"]["students_two_sided"][lv]["result"])
    add("H4b verdict", H["H4b"]["verdict"], mine["H4b"]["verdict"])
    add("H7 Wilcoxon p", H["H7"]["wilcoxon"]["p"], mine["H7"]["p"])
    add("H7 verdict", H["H7"]["verdict"], mine["H7"]["verdict"])
    add("H8 min ρ", H["H8"]["spearman_min"], min(mine["H8"]["rhos"]), 1e-9)
    add("H8 median ρ", H["H8"]["spearman_hs_vs_neglog_restricted_loss"], float(np.median(mine["H8"]["rhos"])), 1e-9)
    add("H8 verdict", H["H8"]["verdict"], mine["H8"]["verdict"])
    for lv, v in mine["H5"]["T0"].items():
        add(f"H5 T0 {lv} gap", H["H5"]["agent_minus_compiler_main"][lv], v["gap"])
    return rows


def main() -> None:
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mine = evaluate(r)
    rows = compare(r, mine)
    for x in rows:
        log(f"{'OK ' if x['agree'] else 'DIFF'} {x['item']}: stored {x['stored']} | recomputed {x['recomputed']}")
    for t, v in mine["replications"].items():
        log(f"replication {t}: H1 {v['H1']['verdict']} (JT p {v['H1']['jt']['p']:.3f}, ρ {v['H1']['spearman']['rho']:.2f}, "
            f"CI [{v['H1']['spearman']['ci'][0]:.2f}, {v['H1']['spearman']['ci'][1]:.2f}]); H2 {v['H2']['verdict']} "
            f"(p {v['H2']['p']:.3f}); H3 {v['H3']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"round": r, "recomputed": mine, "comparison": rows,
                               "n_agree": sum(x["agree"] for x in rows), "n_items": len(rows)},
                              indent=2, ensure_ascii=False, default=float), encoding="utf-8", newline="\n")
    log(f"{sum(x['agree'] for x in rows)} of {len(rows)} stored items reproduced; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
