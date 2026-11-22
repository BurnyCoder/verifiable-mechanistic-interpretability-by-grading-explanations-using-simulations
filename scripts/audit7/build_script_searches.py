"""Audit 7: searches inside the agents' final build scripts, and the results without the runs that contain one.

Local explanation: TASK.md forbids "any loop that updates weights to reduce a loss"; the static audit
(src/vmi/simulator/audit.py) flags only gradient/optimiser patterns, so a loop that tries a few values of a constant and
keeps the best by some criterion passes it. This script (1) scans every reported run's final build_weights.py for such
loops — a `best` variable updated inside a loop, or a fixed-count `for _ in range(n)` loop that updates a weight — (2)
attaches the hand-read label of each detected loop (what it varies, how many candidates, and the criterion: task
labels, a quantity the explanation states, or an analytic formula; read in full for this audit), and (3) recomputes the
seed-target replication rows, the agent–compiler comparison on the seed targets and the affected ablation means without
those runs.
Global context: seventh audit (2026-11-22). Build scripts live in the run folders outside the repository
(VMI_RUNS_DIR, .env.example); the detected lines are copied into results/audit7/build_script_searches.json so the
finding can be checked without them. No preregistered verdict uses a run listed here (the T0 main and controls runs,
which every hypothesis test uses, contain no such loop).
Sources: rule in sandbox_template/TASK.md ("No training"); manual-review standard in results/audit_reviews.json (a
search scored without data or labels was judged valid on the dev target); tests as in scripts/audit7/verdicts.py.
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import re  # loop detection
import sys  # import path for the sibling audit script
from pathlib import Path  # paths

import numpy as np  # means

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/audit7 on the path
from verdicts import RES, ROOT, cond, jt, mw_greater, runs  # noqa: E402  exact tests and loaders (this audit's own code)

BEST = re.compile(r"^\s*if\s+best\b.*(<|>)", re.M)  # "if best is None or err < best[0]:" — keep-the-best update
FIXED_ITER = re.compile(r"^\s*for\s+_\s+in\s+range\((\w+|\d+)\):\s*$", re.M)  # bare fixed-count loop (calibration/search)

# Hand-read labels (audit 7): every detected loop was read in its script. "criterion" says what decides the choice.
LABELS = {
    "ablations__T0__E3-fmt-math__claudehaiku55-claudecode__tdef__s2": (
        "MLP bias offset, 5 values", "least-squares fit of the MLP output to cos/sin(w_k(a+b)) at the explanation's frequencies",
        "explanation"),
    "ablations__T0__E3-noselftest__claudehaiku55-claudecode__tdef__s1": (
        "MLP bias scale, 3 values", "mean R² of the least-squares read-out of cos/sin(w_k(a+b))", "explanation"),
    "seed__T-new__E3__claudehaiku55-claudecode__tdef__s2": (
        "8-step multiplicative update of each head's '=' attention score; then MLP bias scale, 4 values",
        "attention to '=' matched to the explanation's stated value; R² of the read-out of cos/sin(w_k(a+b))", "explanation"),
    "seed__T-new__W-freq__claudehaiku55-claudecode__tdef__s2": (
        "attention gain per head, 231-point scan", "mean squared error to the explanation's fitted attention formula",
        "explanation"),
    "seed__T-paper__E1__claudehaiku55-claudecode__tdef__s2": (
        "set of 5 frequencies, 20,000 random draws", "largest off-peak value of Σ_k cos(w_k d) (no data, no labels)",
        "analytic"),
    "seed__T-paper__E3__claudehaiku55-claudecode__tdef__s0": (
        "spread of neuron phases, 3 values", "least-squares fit error of the MLP output to cos/sin(w_k(a+b))", "explanation"),
    "seed__T-paper__W-freq__claudehaiku55-claudecode__tdef__s2": (
        "MLP bias ratio, 4 values", "task accuracy, then mean probability of the correct answer (task labels)", "labels"),
    "hyper__T0__W-freq-effort-low__claudehaiku55-claudecode__tdef__s0": (
        "one candidate only (G = 3, B = [1, 2, 3, 4])", "task accuracy, but with a single candidate nothing is selected",
        "none (degenerate loop)"),
}


def runs_dir() -> Path:
    """VMI_RUNS_DIR from .env (the run folders with each agent's final build_weights.py)."""
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line and not line.startswith("#"))
    return Path(env.get("VMI_RUNS_DIR", "D:/AI/vmi-sim/runs").split("#")[0].strip())


def detect(rs: list[dict]) -> dict[str, list[str]]:
    """Run id → the detected loop lines of its final build_weights.py (main simulator's reported runs only)."""
    out = {}
    for r in rs:
        if r["model"] != "claude-haiku-5-5":
            continue
        src = (runs_dir() / r["run_id"] / "work_out" / "build_weights.py").read_text(encoding="utf-8", errors="replace")
        hits = [m.group(0).strip() for m in BEST.finditer(src)] + [m.group(0).strip() for m in FIXED_ITER.finditer(src)]
        if hits:
            out[r["run_id"]] = hits
    return out


def without(rs: list[dict], drop: set[str]) -> list[dict]:
    """The reported runs minus the given run ids."""
    return [r for r in rs if r["run_id"] not in drop]


def replication(rs: list[dict]) -> dict:
    """Seed-target replication rows (H1 trend over the ladder levels run, H2 E3 > W-freq, H3) and agent − compiler."""
    out = {}
    for t, lv in (("T1", ["E2", "E3", "E4"]), ("T2", ["E2", "E3", "E4"]), ("T-new", ["E3", "E4"]), ("T-paper", ["E1", "E3", "E4"])):
        gs = [cond(rs, t, x) for x in lv]
        wf = cond(rs, t, "W-freq")
        trues = [x for x in ("E2", "E3", "E4") if len(cond(rs, t, x))]
        kl_max = max(float(cond(rs, t, x, "kl_out").mean()) for x in trues)
        agree = float(cond(rs, t, "W-freq", "argmax_agreement").mean())
        comp = {x: json.loads((RES / "compiler" / t / f"{x}.json").read_text())["hs"] for x in trues + ["W-freq"]}
        out[t] = {"n": {x: len(g) for x, g in zip(lv, gs)} | {"W-freq": len(wf)},
                  "hs_means": {x: float(g.mean()) for x, g in zip(lv, gs)},
                  "jt_p": jt(gs)[1], "e3_minus_wfreq": float(cond(rs, t, "E3").mean() - wf.mean()),
                  "mw_p": mw_greater(cond(rs, t, "E3"), wf),
                  "h3_holds": bool(agree >= 0.99 and cond(rs, t, "W-freq", "kl_out").mean() <= kl_max
                                   and all(wf.mean() < cond(rs, t, x).mean() for x in trues)),
                  "agent_minus_compiler": {x: float(cond(rs, t, x).mean() - comp[x]) for x in comp},
                  "compiler_below_every_run": {x: bool(cond(rs, t, x).min() > comp[x]) for x in comp}}
    return out


def ablation_means(rs: list[dict]) -> dict:
    """Mean HS of the T0 ablation and effort conditions that contain a listed run."""
    def m(variant):
        return float(np.mean([r["hs"] for r in rs if r["target"] == "T0" and r.get("variant") == variant
                              and r["model"] == "claude-haiku-5-5"]))
    return {v: m(v) for v in ("fmt-math", "noselftest")}


def main() -> dict:
    rs = runs()
    found = detect(rs)
    assert set(found) == set(LABELS), f"detected {sorted(found)} != labelled {sorted(LABELS)}"  # every hit was read
    selecting = {k for k, v in LABELS.items() if not v[2].startswith("none")}
    hyp_runs = [r["run_id"] for r in rs if r["target"] == "T0" and r["stage"] in ("main", "controls")]
    return {
        "rule": "TASK.md: no gradient descent, optimisers or any loop that updates weights to reduce a loss",
        "n_reported_main_simulator": sum(r["model"] == "claude-haiku-5-5" for r in rs),
        "detected": {k: {"lines": found[k], "varies": LABELS[k][0], "criterion": LABELS[k][1], "criterion_kind": LABELS[k][2]}
                     for k in sorted(found)},
        "n_selecting": len(selecting),
        "by_kind": {k: sum(LABELS[r][2] == k for r in selecting) for k in ("explanation", "labels", "analytic")},
        "in_hypothesis_tests": sorted(set(selecting) & set(hyp_runs)),
        "replication_all": (rep_all := replication(rs)), "replication_without": (rep_wo := replication(without(rs, selecting))),
        "summary": {k: summarise(r) for k, r in (("all", rep_all), ("without", rep_wo))},
        "ablation_means_all": ablation_means(rs), "ablation_means_without": ablation_means(without(rs, selecting)),
    }


def summarise(rep: dict) -> dict:
    """Counts the paper quotes for the seed targets T1, T2, T-new (H5 extension) and the readings of every target."""
    seed = [(t, x, d) for t in ("T1", "T2", "T-new") for x, d in rep[t]["agent_minus_compiler"].items()]
    return {"pairs": len(seed), "above_by_sign": int(sum(d > 0 for _, _, d in seed)),
            "beyond_ci": sorted(f"{t} {x}" for t in ("T1", "T2", "T-new") for x, b in rep[t]["compiler_below_every_run"].items() if b),
            "tpaper_beyond_ci": sorted(x for x, b in rep["T-paper"]["compiler_below_every_run"].items() if b),
            "h3_holds": {t: r["h3_holds"] for t, r in rep.items()},
            "e3_above_wfreq": {t: bool(r["e3_minus_wfreq"] > 0) for t, r in rep.items()},
            "trend_supported": {t: bool(r["jt_p"] < 0.05) for t, r in rep.items()}}


if __name__ == "__main__":
    res = main()
    (RES / "audit7").mkdir(exist_ok=True)
    (RES / "audit7" / "build_script_searches.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(f"{res['n_selecting']} of {res['n_reported_main_simulator']} final build scripts select among candidates;",
          "by criterion", res["by_kind"], "; in hypothesis tests:", res["in_hypothesis_tests"])
