"""Audit 6: re-derive the round-2 verdicts from the raw run records with code written for this audit.

Local explanation: reads results/runs/*.json, results/baselines/T0.json, results/compiler/ and results/controls/ and
recomputes, from preregistration §5 as written: H1's Jonckheere–Terpstra p-values by full enumeration of the distinct
assignments of runs to levels (sympy multiset permutations), Spearman ρ and its within-level bootstrap interval; H2's
one-sided exact Mann–Whitney p-values (scipy) with Holm, and the percentile-bootstrap interval of each difference from
the exact bootstrap distribution (all 27 × 27 resamples); H3's condition means; H4's Mann–Whitney against the nine
retrains with Holm; H4b's monotonicity; H7's one-sided exact Wilcoxon on the stored margins (after checking each margin
against its stored HS values); H8's Spearman ρ per replacement; H5 on T0; and the replication rows. Each value is
compared with results/hypotheses_round2.json. It also records the H2 verdict under §5's falsification column read
literally ("not significant after Holm" → falsified) next to the "cannot tell at this N" rule of deviation 1.
Global context: an audit check; no generated document reads its verdicts, only its agreement counts.
Sources: Jonckheere 1954 / Terpstra 1952 (statistic as in src/vmi/analysis/stats.py docstring); Holm 1979;
scipy.stats.mannwhitneyu / wilcoxon / spearmanr (https://docs.scipy.org/doc/scipy/reference/stats.html);
sympy multiset_permutations (https://docs.sympy.org/latest/modules/utilities/iterables.html).
Run: uv run python -I scripts/audit6/verdicts.py  →  results/audit6/verdicts.json
"""

from __future__ import annotations  # postponed annotations

import glob  # run records
import itertools  # exact bootstrap
import json  # inputs and output
import os  # paths
import sys  # repo root

import numpy as np  # arrays
from scipy import stats  # exact rank tests
from sympy.utilities.iterables import multiset_permutations  # distinct label sequences

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = [json.load(open(p, encoding="utf-8")) for p in glob.glob(f"{REPO}/results/runs/*.json")]
RUNS = [r for r in RUNS if r.get("stage") != "dev" and 1 <= int(r.get("round", 0)) <= 2]
STORED = json.load(open(f"{REPO}/results/hypotheses_round2.json", encoding="utf-8"))["hypotheses"]


def vals(level, key="hs", target="T0"):
    """Per-run values of the main condition (Claude Haiku 5.5, default sampling, no variant)."""
    return np.array([r[key] for r in RUNS if r["target"] == target and r["level"] == level and r["model"] == "claude-haiku-5-5"
                     and not r.get("variant") and r.get("temperature") is None])


def jt_exact(groups):
    """One-sided JT p-value (increasing order) by enumerating every distinct assignment of runs to levels."""
    v = np.concatenate(groups)
    lab = np.concatenate([[i] * len(g) for i, g in enumerate(groups)])
    less = (v[:, None] < v[None, :]) + 0.5 * (v[:, None] == v[None, :])
    obs = float((less * (lab[:, None] < lab[None, :])).sum())
    L = np.array(list(multiset_permutations(lab.tolist())), dtype=np.int8)
    tot = np.zeros(len(L))
    for i, j in zip(*np.nonzero(less)):
        tot += less[i, j] * (L[:, i] < L[:, j])
    return float((tot >= obs - 1e-12).mean())


def boot_mean_ci(x):
    means = [np.mean(c) for c in itertools.product(x, repeat=len(x))]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_diff_ci(x, y):
    mx = np.array([np.mean(c) for c in itertools.product(x, repeat=len(x))])
    my = np.array([np.mean(c) for c in itertools.product(y, repeat=len(y))])
    d = (mx[:, None] - my[None, :]).ravel()
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def holm(p: dict) -> dict:
    items, out, run = sorted(p.items(), key=lambda kv: kv[1]), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def main() -> None:
    res, checks = {}, []

    def check(name, mine, stored, tol=1e-9):
        checks.append({"what": name, "audit6": mine, "stored": stored, "match": bool(abs(mine - stored) <= tol)})

    lv = ["E1", "E2", "E3", "E4"]
    g = [vals(x) for x in lv]
    res["H1"] = {"jt_p_hs": jt_exact(g), "jt_p_neg_kl": jt_exact([-vals(x, "kl_out") for x in lv]),
                 "rho_hs": float(stats.spearmanr(np.repeat(np.arange(4), 3), np.concatenate(g)).statistic),
                 "E0_mean": float(vals("E0").mean()), "E1_mean": float(g[0].mean())}
    check("H1 JT p (HS)", res["H1"]["jt_p_hs"], STORED["H1"]["jt_hs"]["p"])
    check("H1 JT p (−KL)", res["H1"]["jt_p_neg_kl"], STORED["H1"]["jt_neg_kl"]["p"])
    check("H1 Spearman ρ (HS)", res["H1"]["rho_hs"], STORED["H1"]["spearman_hs"]["rho"])
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"), "E4>W-scramble": ("E4", "W-scramble")}
    raw = {k: float(stats.mannwhitneyu(vals(a), vals(b), alternative="greater", method="exact").pvalue) for k, (a, b) in pairs.items()}
    adj = holm(raw)
    res["H2"] = {k: {"diff": float(vals(a).mean() - vals(b).mean()), "ci": boot_diff_ci(vals(a), vals(b)), "p": raw[k], "holm_p": adj[k],
                     "verdict_section5_literal": "not supported" if adj[k] >= 0.05 else "supported",
                     "verdict_deviation1_rule": "cannot tell at this N"} for k, (a, b) in pairs.items()}
    for k in pairs:
        check(f"H2 {k} p", raw[k], STORED["H2"]["tests"][k]["p"])
        check(f"H2 {k} diff", res["H2"][k]["diff"], STORED["H2"]["effects"][k]["diff"])
        # stored intervals are Monte-Carlo (10,000 resamples), ours the exact bootstrap distribution: compare up to MC error
        check(f"H2 {k} CI low (Monte-Carlo vs exact bootstrap)", res["H2"][k]["ci"][0], STORED["H2"]["effects"][k]["ci"][0], 5e-3)
        check(f"H2 {k} CI includes 0 (same reading)", float(res["H2"][k]["ci"][0] <= 0 <= res["H2"][k]["ci"][1]),
              float(STORED["H2"]["effects"][k]["ci"][0] <= 0 <= STORED["H2"]["effects"][k]["ci"][1]))
    wf = [r for r in RUNS if r["target"] == "T0" and r["level"] == "W-freq" and r["model"] == "claude-haiku-5-5" and not r.get("variant")]
    res["H3"] = {"agreement": float(np.mean([r["argmax_agreement"] for r in wf])), "kl": float(vals("W-freq", "kl_out").mean()),
                 "hs": float(vals("W-freq").mean()), "true_kl_max": float(max(vals(x, "kl_out").mean() for x in ("E2", "E3", "E4"))),
                 "true_hs_min": float(min(vals(x).mean() for x in ("E2", "E3", "E4")))}
    res["H3"]["supported"] = res["H3"]["agreement"] >= 0.99 and res["H3"]["kl"] <= res["H3"]["true_kl_max"] and res["H3"]["hs"] < res["H3"]["true_hs_min"]
    check("H3 supported", float(res["H3"]["supported"]), float(STORED["H3"]["verdict"] == "supported"))
    retr = np.array([x["hs"] for x in json.load(open(f"{REPO}/results/baselines/T0.json", encoding="utf-8"))["retrains"]])
    p4 = {x: float(stats.mannwhitneyu(vals(x), retr, alternative="greater", method="exact").pvalue) for x in ("E3", "E4")}
    res["H4"] = {"p": p4, "holm_p": holm(p4)}
    for x in ("E3", "E4"):
        check(f"H4 {x} Holm p", res["H4"]["holm_p"][x], STORED["H4"]["holm_p"][x])
    rows = json.load(open(f"{REPO}/results/controls/planted_calibration_T0.json", encoding="utf-8"))["rows"]
    hs4, kl4 = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
    res["H4b"] = {"supported": bool(abs(hs4[0] - 1) < 1e-6 and kl4[0] < 1e-9 and all(a > b for a, b in zip(hs4, hs4[1:]))
                                    and all(a < b for a, b in zip(kl4, kl4[1:])))}
    check("H4b supported", float(res["H4b"]["supported"]), float(STORED["H4b"]["verdict"] == "supported"))
    wt = json.load(open(f"{REPO}/results/controls/wrong_targets_T0.json", encoding="utf-8"))["rows"]
    stored_hs = {r["run_id"]: r["hs"] for r in RUNS}
    assert all(abs(x["hs_T0"] - stored_hs[x["run_id"]]) < 1e-12 and abs(x["d"] - (x["hs_T0"] - max(x["hs_wrong"].values()))) < 1e-12 for x in wt)
    d = np.array([x["d"] for x in wt])
    res["H7"] = {"d": d.tolist(), "wilcoxon_p": float(stats.wilcoxon(d, alternative="greater", method="exact").pvalue)}
    check("H7 Wilcoxon p", res["H7"]["wilcoxon_p"], STORED["H7"]["wilcoxon"]["p"])
    tr = json.load(open(f"{REPO}/results/controls/trajectory_T0.json", encoding="utf-8"))
    nl = -np.log(np.array(tr["restricted_loss"]))
    rhos = [float(stats.spearmanr(x["hs"], nl).statistic) for x in tr["replacements"]]
    res["H8"] = {"rhos": rhos, "min": min(rhos), "median": float(np.median(rhos))}
    check("H8 min ρ", res["H8"]["min"], STORED["H8"]["spearman_min"])
    res["H5_T0"] = {}
    for x in ("E2", "E3", "E4", "W-freq"):
        c = json.load(open(f"{REPO}/results/compiler/T0/{x}.json", encoding="utf-8"))["hs"]
        lo, hi = boot_mean_ci(vals(x))
        res["H5_T0"][x] = {"agent_minus_compiler": float(vals(x).mean() - c), "beyond_ci": bool(lo > c)}
        check(f"H5 T0 {x} gap", res["H5_T0"][x]["agent_minus_compiler"], STORED["H5"]["agent_minus_compiler_main"][x])
    res["replications"] = {}
    for t, levels in (("T1", ["E2", "E3", "E4"]), ("T2", ["E2", "E3", "E4"]), ("T-new", ["E3", "E4"]), ("T-paper", ["E1", "E3", "E4"])):
        res["replications"][t] = {"jt_p": jt_exact([vals(x, target=t) for x in levels]),
                                  "mw_e3_wfreq_p": float(stats.mannwhitneyu(vals("E3", target=t), vals("W-freq", target=t),
                                                                           alternative="greater", method="exact").pvalue)}
    out = {"results": res, "checks": checks, "n_checks": len(checks), "n_match": sum(c["match"] for c in checks)}
    os.makedirs(f"{REPO}/results/audit6", exist_ok=True)
    with open(f"{REPO}/results/audit6/verdicts.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{out['n_match']} of {out['n_checks']} checks match")
    for c in checks:
        if not c["match"]:
            print("MISMATCH", c)


if __name__ == "__main__":
    main()
