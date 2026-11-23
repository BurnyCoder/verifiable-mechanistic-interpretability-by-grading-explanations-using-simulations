"""Audit 9 (independent, 2026-11-23): re-derive the round-2 statistics from the raw records and check the new readings.

Local explanation: reads results/runs/*.json, results/baselines/T0.json, results/controls/wrong_targets_T0.json,
results/compiler/<target>/*.json, results/frequencies_round2.json and results/claude/ledger.jsonl; recomputes every
preregistered test of rounds 2 by exact enumeration with code written for this audit (numpy and the standard library
only), compares each value with results/hypotheses_round2.json, and adds the values audit 9's corrections quote:
Holm over all nine confirmatory p-values, the separation behind the students comparison, and the counts of the H5
seed-target comparison. Writes results/audit9/verify.json; exits with status 1 on any mismatch.
Global context: written without reading the earlier audit scripts; an error in scipy or in src/vmi cannot carry over.
Sources: Jonckheere 1954 / Terpstra 1952 (J = Σ_{i<j} #{x ∈ G_i, y ∈ G_j: x < y}; exact null = every labelled
assignment of runs to levels); Mann–Whitney U (exact null = every split); Wilcoxon signed-rank (exact null = every
sign pattern); Holm 1979 (step-down, running maximum).
"""

from __future__ import annotations  # postponed annotations

import itertools  # exact enumeration
import json  # results files
import sys  # exit status
from pathlib import Path  # paths

import numpy as np  # arithmetic

ROOT = Path(__file__).resolve().parents[2]  # scripts/audit9/ → repository root
RES = ROOT / "results"
MAIN, TOL = "claude-haiku-5-5", 1e-9  # main simulator (amendment 2), match tolerance for stored values


def _load(rel: str):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


RUNS = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]


def scores(target: str, level: str, key: str = "hs", stage: str | None = None) -> list[float]:
    """Per-run values of one main-simulator condition (no variant)."""
    return [r[key] for r in RUNS if r["model"] == MAIN and r["target"] == target and r["level"] == level
            and not r.get("variant") and r["stage"] != "dev" and (stage is None or r["stage"] == stage)]


def labelled_assignments(sizes: list[int]):
    """Every way to place group labels 0..k−1 on n = Σ sizes positions with the given group sizes."""
    def rec(free: tuple, k: int):
        if k == len(sizes):
            yield {}
            return
        for chosen in itertools.combinations(free, sizes[k]):
            rest = tuple(i for i in free if i not in chosen)
            for tail in rec(rest, k + 1):
                yield {**{i: k for i in chosen}, **tail}
    yield from rec(tuple(range(sum(sizes))), 0)


def jt_p(groups: list[list[float]]) -> tuple[float, float]:
    """(J, one-sided exact p) for an increasing trend over the ordered groups."""
    vals = [v for g in groups for v in g]
    less = np.array([[a < b for b in vals] for a in vals], float)
    def J(lab):
        return sum(less[i, j] for i in range(len(vals)) for j in range(len(vals)) if lab[i] < lab[j])
    obs = J({i: k for k, g in enumerate(groups) for i in range(sum(map(len, groups[:k])), sum(map(len, groups[:k + 1])))})
    null = [J(a) for a in labelled_assignments([len(g) for g in groups])]
    return obs, float(np.mean([x >= obs - TOL for x in null]))


def mw_p(x: list[float], y: list[float], two_sided: bool = False) -> tuple[float, float]:
    """(U = #{x > y}, exact p) one-sided for x > y, or two-sided by doubling the smaller tail (capped at 1)."""
    allv = list(x) + list(y)
    U = lambda xs, ys: sum(a > b for a in xs for b in ys)  # noqa: E731
    obs = U(x, y)
    null = [U([allv[i] for i in c], [allv[i] for i in range(len(allv)) if i not in c])
            for c in itertools.combinations(range(len(allv)), len(x))]
    up = float(np.mean([u >= obs for u in null]))
    if not two_sided:
        return obs, up
    return obs, min(1.0, 2 * min(up, float(np.mean([u <= obs for u in null]))))


def wilcoxon_p(d: list[float]) -> float:
    """One-sided exact p for median d > 0 (ranks of |d|; no zeros or ties in this data)."""
    r = np.argsort(np.argsort(np.abs(d))) + 1
    obs = r[np.array(d) > 0].sum()
    null = [sum(ri for ri, s in zip(r, signs) if s) for signs in itertools.product([0, 1], repeat=len(d))]
    return float(np.mean([w >= obs for w in null]))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    out, run, m = {}, 0.0, len(p)
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def derive() -> dict:
    """Every statistic, recomputed."""
    t0 = {lv: scores("T0", lv) for lv in ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble")}
    kl = {lv: scores("T0", lv, "kl_out") for lv in ("E1", "E2", "E3", "E4")}
    base = _load("baselines/T0.json")
    ret, stu = [x["hs"] for x in base["retrains"]], [x["hs"] for x in base["students"]]
    out: dict = {}
    out["H1 JT HS"] = jt_p([t0[lv] for lv in ("E1", "E2", "E3", "E4")])[1]
    out["H1 JT -KL"] = jt_p([[-v for v in kl[lv]] for lv in ("E1", "E2", "E3", "E4")])[1]
    h2 = {k: mw_p(t0[a], t0[b])[1] for k, (a, b) in {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"),
                                                     "E3>W-role": ("E3", "W-role"), "E4>W-scramble": ("E4", "W-scramble")}.items()}
    out.update({f"H2 p {k}": v for k, v in h2.items()})
    out.update({f"H2 Holm {k}": v for k, v in holm(h2).items()})
    h4 = {lv: mw_p(t0[lv], ret)[1] for lv in ("E3", "E4")}
    out.update({f"H4 Holm {k}": v for k, v in holm(h4).items()})
    h4s = {lv: mw_p(t0[lv], stu, two_sided=True)[1] for lv in ("E3", "E4")}
    out.update({f"H4 students Holm {k}": v for k, v in holm(h4s).items()})
    out["H4 students separated (every agent run above every student)"] = float(all(min(t0[lv]) > max(stu) for lv in ("E3", "E4")))
    d = [row["d"] for row in _load("controls/wrong_targets_T0.json")["rows"]]
    out["H7 Wilcoxon"] = wilcoxon_p(d)
    for t, lvs in (("T1", ("E2", "E3", "E4")), ("T2", ("E2", "E3", "E4")), ("T-new", ("E3", "E4")), ("T-paper", ("E1", "E3", "E4"))):
        out[f"replication JT {t}"] = jt_p([scores(t, lv) for lv in lvs])[1]
        out[f"replication MW E3>W-freq {t}"] = mw_p(scores(t, "E3"), scores(t, "W-freq"))[1]
    # Audit 9 additions: Holm across all nine confirmatory tests (each test's own p)
    allp = {"H1 HS": out["H1 JT HS"], "H1 KL": out["H1 JT -KL"], **{f"H2 {k}": v for k, v in h2.items()},
            **{f"H4 {k}": v for k, v in h4.items()}, "H7": out["H7 Wilcoxon"]}
    out.update({f"Holm-all {k}": v for k, v in holm(allp).items()})
    sign = beyond = 0  # H5 on the seed targets: agent mean vs compiler, and all runs above it (the 3-run percentile CI)
    for t in ("T1", "T2", "T-new"):
        for lv in ("E2", "E3", "E4", "W-freq"):
            a, cf = scores(t, lv), RES / "compiler" / t / f"{lv}.json"
            if a and cf.exists():
                c = json.loads(cf.read_text(encoding="utf-8"))["hs"]
                sign += np.mean(a) > c
                beyond += min(a) > c
    out["H5 seed pairs above compiler by sign"], out["H5 seed pairs beyond the range"] = float(sign), float(beyond)
    true_kl = scores("T0", "E2", "kl_out", "main") + scores("T0", "E3", "kl_out", "main") + scores("T0", "E4", "kl_out", "main")
    out["mean KL of the true E2-E4 main runs"] = float(np.mean(true_kl))
    led = [json.loads(line) for line in (RES / "claude" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rep = [e for e in led if not e["run_id"].startswith("dev__") and e["run_id"] != "claude_check"]
    out["spend of reported runs (USD)"] = round(sum(e["cost_usd"] for e in rep), 6)
    out["requests of reported runs"] = float(len(rep))
    return out


def stored() -> dict:
    """The values the project's analysis stored, keyed like derive()."""
    H = _load("hypotheses_round2.json")["hypotheses"]
    s = {"H1 JT HS": H["H1"]["jt_hs"]["p"], "H1 JT -KL": H["H1"]["jt_neg_kl"]["p"], "H7 Wilcoxon": H["H7"]["wilcoxon"]["p"]}
    s.update({f"H2 p {k}": t["p"] for k, t in H["H2"]["tests"].items()})
    s.update({f"H2 Holm {k}": v for k, v in H["H2"]["holm_p"].items()})
    s.update({f"H4 Holm {k}": v for k, v in H["H4"]["holm_p"].items()})
    s.update({f"H4 students Holm {k}": t["holm_p"] for k, t in H["H4"]["students"].items()})
    return s


def main() -> int:
    got, ref = derive(), stored()
    checks = [{"name": k, "derived": got[k], "stored": ref[k], "agree": abs(got[k] - ref[k]) < 1e-9} for k in ref]
    out = {"checks": checks, "derived_only": {k: v for k, v in got.items() if k not in ref}}
    (RES / "audit9").mkdir(exist_ok=True)
    (RES / "audit9" / "verify.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    bad = [c for c in checks if not c["agree"]]
    print(f"{len(checks) - len(bad)} of {len(checks)} stored statistics reproduce")
    for c in bad:
        print("MISMATCH", c)
    for k, v in out["derived_only"].items():
        print(f"{k}: {v:.6g}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
