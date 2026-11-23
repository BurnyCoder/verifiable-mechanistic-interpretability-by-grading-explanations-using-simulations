"""Audit 8 (independent, 2026-11-23): re-derive the round-2 statistics and both verdict readings from raw records.

Local explanation: reads results/runs/*.json (per-run scores), results/baselines/T0.json (retrains),
results/compiler/<target>/<explanation>.json (agent-free builds), results/controls/wrong_targets_T0.json (H7),
results/frequencies_round2.json (memorisation), results/targets_selection.json (key frequencies of every wd1 seed) and
results/claude/ledger.jsonl (spend); recomputes every preregistered test by exact enumeration and compares each value
with results/hypotheses_round2.json. Each verdict is reported twice: as preregistration §5 defines it ("cannot tell"
only when the test is undefined, otherwise p < α decides) and under deviation 1's later rule ("cannot tell" whenever
the smallest attainable p is ≥ α). Writes results/audit8/verify.json and prints mismatches.
Global context: written for audit 8 without reading the earlier audit scripts; numpy and the standard library only,
so an error in scipy or in src/vmi cannot carry over (tests/test_audit8.py compares the tests with scipy).
Sources: Jonckheere 1954 / Terpstra 1952 (J = Σ_{i<j} #{x ∈ G_i, y ∈ G_j: x < y} + ½ ties; exact null = every distinct
assignment of runs to levels); Mann–Whitney U (exact null = every split); Wilcoxon signed-rank (exact null = every sign
pattern); Holm 1979; Spearman ρ = Pearson correlation of average ranks; percentile bootstrap.
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact enumeration
import json  # results files
from pathlib import Path  # paths

import numpy as np  # arithmetic

ROOT = Path(__file__).resolve().parents[2]  # repository root (scripts/audit8/ → ../..)
RES = ROOT / "results"
MAIN, ALPHA, TOL = "claude-haiku-5-5", 0.05, 1e-9  # main simulator (amendment 2), preregistered α, match tolerance
T0_LEVELS = {"E1": "main", "E2": "main", "E3": "main", "E4": "main", "E0": "controls", "W-freq": "controls",
             "W-alg": "controls", "W-role": "controls", "W-scramble": "controls"}  # stage that ran each T0 condition


# ------------------------------------------------------------------------------------------------- exact tests
def _assignments(sizes: list[int]):
    """Every distinct way to give n = Σsizes positions the group labels 0..k−1 with the given group sizes."""
    if not sizes:
        yield ()
        return
    n = sum(sizes)
    for first in itertools.combinations(range(n), sizes[0]):
        rest = [i for i in range(n) if i not in first]
        for tail in _assignments(sizes[1:]):
            lab = [0] * n
            for i, t in zip(rest, tail):
                lab[i] = t + 1
            yield tuple(lab)


def jt_greater(groups) -> float:
    """One-sided exact Jonckheere–Terpstra p for an increasing trend over groups in their hypothesised order."""
    x = np.concatenate([np.asarray(g, float) for g in groups])
    C = (x[:, None] < x[None, :]) + 0.5 * (x[:, None] == x[None, :])  # C[u, v]: u below v
    L = np.array(list(_assignments([len(g) for g in groups])), dtype=np.int8)
    J = np.concatenate([(C * (L[i:i + 50000, :, None] < L[i:i + 50000, None, :])).sum((1, 2)) for i in range(0, len(L), 50000)])
    obs = (C * (np.repeat(np.arange(len(groups)), [len(g) for g in groups])[:, None]
                < np.repeat(np.arange(len(groups)), [len(g) for g in groups])[None, :])).sum()
    return float((J >= obs - 1e-12).mean())


def mw_greater(x, y) -> float:
    """One-sided exact Mann–Whitney p that x tends to exceed y (U counts pairs x > y, ties ½)."""
    z = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    U = lambda a, b: float(((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).sum())  # noqa: E731
    obs = U(z[:len(x)], z[len(x):])
    us = [U(z[list(c)], np.delete(z, list(c))) for c in itertools.combinations(range(len(z)), len(x))]
    return float(np.mean(np.array(us) >= obs - 1e-12))


def wilcoxon_greater(d) -> float:
    """One-sided exact Wilcoxon signed-rank p that d > 0 (no zeros or ties expected; W+ = sum of positive ranks)."""
    d = np.asarray(d, float)
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    obs = r[d > 0].sum()
    return float(np.mean([r[np.array(s)].sum() >= obs - 1e-12 for s in itertools.product([False, True], repeat=len(d))]))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    out, run = {}, 0.0
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (len(p) - i) * v))
        out[k] = run
    return out


def verdicts(p: float | None, direction: bool, min_p: float) -> tuple[str, str]:
    """(preregistered §5 reading, deviation-1 reading) of one test at α = 0.05."""
    decided = "supported" if p is not None and p < ALPHA and direction else "not supported"
    prereg = "cannot tell at this N" if p is None else decided
    return prereg, "cannot tell at this N" if p is None or min_p >= ALPHA else decided


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        r[v == u] = r[v == u].mean()
    return r


def spearman_ci(groups, n_boot: int = 10000) -> tuple[float, float, float]:
    """Spearman ρ of level index vs value, and a 95 % percentile bootstrap CI resampling runs within each level."""
    lv = np.repeat(np.arange(len(groups)), [len(g) for g in groups]).astype(float)
    rho = lambda vals: float(np.corrcoef(_ranks(lv), _ranks(vals))[0, 1])  # noqa: E731
    rng = np.random.default_rng(8)
    boots = [rho(np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(n_boot)]
    return rho(np.concatenate(groups)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


# ------------------------------------------------------------------------------------------------- loading
def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


RUNS = [r for r in map(_json, sorted((RES / "runs").glob("*.json"))) if r["stage"] != "dev"]


def scores(target: str, level: str, stage: str, key: str = "hs") -> np.ndarray:
    """Per-run scores of one main-simulator condition without a variant, ordered by replicate index."""
    rs = sorted((r for r in RUNS if r["model"] == MAIN and r["target"] == target and r["level"] == level
                 and r["stage"] == stage and not r.get("variant")), key=lambda r: r["seed"])
    return np.array([r[key] for r in rs])


# ------------------------------------------------------------------------------------------------- re-derivation
def derive() -> dict:
    hs = {lv: scores("T0", lv, st) for lv, st in T0_LEVELS.items()}
    kl = {lv: scores("T0", lv, st, "kl_out") for lv, st in T0_LEVELS.items()}
    ladder = ["E1", "E2", "E3", "E4"]
    out: dict = {"n_claude_runs": sum(r["model"] == MAIN for r in RUNS),
                 "flagged_or_failed": sum(bool(r["audit"]["flagged"] or r["failed"]) for r in RUNS),
                 "t0_hs_means": {lv: float(v.mean()) for lv, v in hs.items()}}
    rho, lo, hi = spearman_ci([hs[lv] for lv in ladder])
    p1 = jt_greater([hs[lv] for lv in ladder])
    out["H1"] = {"jt_hs_p": p1, "jt_neg_kl_p": jt_greater([-kl[lv] for lv in ladder]), "spearman_hs": rho,
                 "spearman_ci": [lo, hi], "verdict_hs": verdicts(p1, lo > 0, 1 / 369600)}
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"),
             "E4>W-scramble": ("E4", "W-scramble")}
    p2 = {k: mw_greater(hs[a], hs[b]) for k, (a, b) in pairs.items()}
    h2 = holm(p2)
    out["H2"] = {"p": p2, "holm_p": h2, "diff": {k: float(hs[a].mean() - hs[b].mean()) for k, (a, b) in pairs.items()},
                 "verdicts": {k: verdicts(h2[k], hs[a].mean() > hs[b].mean(), 4 * 0.05) for k, (a, b) in pairs.items()}}
    wf_agree = float(scores("T0", "W-freq", "controls", "argmax_agreement").mean())
    true_kl = [kl[lv].mean() for lv in ("E2", "E3", "E4")]
    out["H3"] = {"agreement": wf_agree, "kl_in_range": bool(min(true_kl) <= kl["W-freq"].mean() <= max(true_kl)),
                 "hs_below_all": bool(all(hs["W-freq"].mean() < hs[lv].mean() for lv in ("E2", "E3", "E4"))),
                 "supported": bool(wf_agree >= 0.99 and kl["W-freq"].mean() <= max(true_kl)
                                   and all(hs["W-freq"].mean() < hs[lv].mean() for lv in ("E2", "E3", "E4")))}
    base = _json(RES / "baselines" / "T0.json")
    retr = np.array([r["hs"] for r in base["retrains"]])
    p4 = holm({lv: mw_greater(hs[lv], retr) for lv in ("E3", "E4")})
    out["H4"] = {"retrain_mean": float(retr.mean()), "retrain_max": float(retr.max()), "holm_p": p4}
    wt = _json(RES / "controls" / "wrong_targets_T0.json")
    d = [r["hs_T0"] - max(r["hs_wrong"].values()) for r in wt["rows"]]
    out["H7"] = {"d": d, "p": wilcoxon_greater(d), "best_wrong": sorted({max(r["hs_wrong"], key=r["hs_wrong"].get) for r in wt["rows"]})}
    h5 = {}
    for t in ("T1", "T2", "T-new"):
        for lv in ("E2", "E3", "E4", "W-freq"):
            a = scores(t, lv, "seed")
            if len(a):
                c = _json(RES / "compiler" / t / f"{lv}.json")["hs"]
                h5[f"{t} {lv}"] = {"margin": float(a.mean() - c), "all_runs_above": bool(a.min() > c)}
    out["H5_seed"] = {"pairs": len(h5), "above_by_sign": sum(v["margin"] > 0 for v in h5.values()),
                      "all_runs_above": sorted(k for k, v in h5.items() if v["all_runs_above"]), "margins": h5}
    # Replications: H2 per target (one test, no Holm) and H1 on T-new (E3 < E4) — the two readings differ here too.
    rep = {}
    for t in ("T1", "T2", "T-new", "T-paper"):
        e3, wf = scores(t, "E3", "seed"), scores(t, "W-freq", "seed")
        p = mw_greater(e3, wf)
        rep[f"{t} H2"] = {"p": p, "verdicts": verdicts(p, e3.mean() > wf.mean(), 0.05)}
    p = jt_greater([scores("T-new", "E3", "seed"), scores("T-new", "E4", "seed")])
    rep["T-new H1"] = {"p": p, "verdicts": verdicts(p, True, 0.05)}
    out["replications"] = rep
    # H7 scope: how many key frequencies each wrong target / retrain shares with T0 (wd0_*, rand_* have no clean mechanism).
    kf = _json(RES / "targets_selection.json")["key_freqs"]
    t0 = set(kf["wd1_s1"])
    paper = set(_json(RES / "targets_selection.json")["paper_freqs"])
    out["shared_with_T0"] = {"T1": len(t0 & set(kf["wd1_s4"])), "T2": len(t0 & set(kf["wd1_s8"])),
                             "dev": len(t0 & set(kf["wd1_s10"])), "T-paper": len(t0 & paper),
                             "retrains": {r["run"]: [len(t0 & set(kf[r["run"]])), round(r["hs"], 3)] for r in base["retrains"]}}
    fr = [r for r in _json(RES / "frequencies_round2.json")["rows"] if r["model"] == MAIN]
    mem = {}
    for lv in ("E0", "E1"):
        rows = [r for r in fr if r["level"] == lv and not r.get("variant")]
        mem[lv] = {"runs": len(rows),
                   "scripts_listing_4_of_5": sum(len(paper & set(r["declared_freqs"] or [])) >= 4 for r in rows),
                   "builds_with_4_of_5": sum(len(paper & set(r["used_freqs"])) >= 4 for r in rows),
                   "paper_share": float(np.mean([r["paper_norm_share"] for r in rows])),
                   "hs_vs_T0": float(np.mean([r["hs_vs_T0"] for r in rows])),
                   "hs_vs_paper": float(np.mean([r["hs_vs_paper"] for r in rows]))}
    out["memorisation"] = mem
    ident = {lv: all((RES / "explanations" / t / lv / "EXPLANATION.md").read_bytes()
                     == (RES / "explanations" / "T0" / lv / "EXPLANATION.md").read_bytes() for t in ("T1", "T2", "T-new", "T-paper"))
             for lv in ("E0", "E1")}
    led = [json.loads(line) for line in (RES / "claude" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    out["e0_e1_identical_across_targets"] = ident
    out["ledger"] = {"requests": len(led), "usd": round(sum(x.get("cost_usd", 0) for x in led), 4)}
    return out


def compare(out: dict) -> list[str]:
    """Mismatches between the re-derived values and results/hypotheses_round2.json."""
    h = _json(RES / "hypotheses_round2.json")["hypotheses"]
    checks = [("H1 JT HS p", out["H1"]["jt_hs_p"], h["H1"]["jt_hs"]["p"]),
              ("H1 JT −KL p", out["H1"]["jt_neg_kl_p"], h["H1"]["jt_neg_kl"]["p"]),
              ("H1 Spearman ρ", out["H1"]["spearman_hs"], h["H1"]["spearman_hs"]["rho"]),
              ("H4 retrain mean", out["H4"]["retrain_mean"], h["H4"]["retrain_hs_mean"]),
              ("H7 Wilcoxon p", out["H7"]["p"], h["H7"]["wilcoxon"]["p"])]
    checks += [(f"H2 {k} p", out["H2"]["p"][k], h["H2"]["tests"][k]["p"]) for k in out["H2"]["p"]]
    checks += [(f"H2 {k} Holm p", out["H2"]["holm_p"][k], h["H2"]["holm_p"][k]) for k in out["H2"]["p"]]
    checks += [(f"H4 {k} Holm p", out["H4"]["holm_p"][k], h["H4"]["holm_p"][k]) for k in ("E3", "E4")]
    checks += [(f"H5 T0 {k}", out["t0_hs_means"][k] - _json(RES / "compiler" / "T0" / f"{k}.json")["hs"],
                h["H5"]["agent_minus_compiler_main"][k]) for k in ("E2", "E3", "E4", "W-freq")]
    out["checks"] = [{"name": n, "derived": a, "stored": b, "agree": abs(a - b) <= TOL} for n, a, b in checks]
    bad = [f"{n}: derived {a} vs stored {b}" for n, a, b in checks if abs(a - b) > TOL]
    if h["H3"]["verdict"] != ("supported" if out["H3"]["supported"] else "not supported"):
        bad.append("H3 verdict")
    if not (out["H1"]["spearman_ci"][0] > 0) == (h["H1"]["spearman_hs"]["ci"][0] > 0):
        bad.append("H1 Spearman CI sign")
    return bad


def main() -> None:
    out = derive()
    bad = compare(out)
    out["mismatches"] = bad
    (RES / "audit8").mkdir(exist_ok=True)
    (RES / "audit8" / "verify.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"{sum(c['agree'] for c in out['checks'])} of {len(out['checks'])} stored statistics reproduced; mismatches: {bad or 'none'}")
    for k, v in out["H2"]["verdicts"].items():
        print(f"H2 {k}: preregistered '{v[0]}', deviation-1 '{v[1]}' (Holm p {out['H2']['holm_p'][k]:.2f})")
    for k, v in out["replications"].items():
        print(f"{k}: preregistered '{v['verdicts'][0]}', deviation-1 '{v['verdicts'][1]}' (p {v['p']:.3f})")


if __name__ == "__main__":
    main()
