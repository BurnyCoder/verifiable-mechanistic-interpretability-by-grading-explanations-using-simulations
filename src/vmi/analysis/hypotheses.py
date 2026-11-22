"""Evaluate every preregistered hypothesis from results/ (preregistration §5–6) and label each verdict with its N.

Local explanation: loads all graded agent runs (results/runs/*.json), compiler results (results/compiler/) and
no-agent baselines/controls (results/baselines/, results/controls/), selects the runs of the current round or earlier,
and returns one verdict per hypothesis: "supported", "not supported" or "cannot tell at this N", with the test
statistics and the N each verdict is based on. Main-condition runs use the main simulator at the preregistered
temperature, without variants, on T0.
Global context: `vmi analyse` writes the output to results/hypotheses_round{r}.json; tables, figures, README and paper
read that file (no hand-typed numbers).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
from collections import defaultdict  # grouping
from pathlib import Path  # paths

import numpy as np  # arrays

from vmi.analysis import stats as st  # tests
from vmi.config import REPO_ROOT  # results folder

RES = REPO_ROOT / "results"
MAIN_MODEL, MAIN_TEMP = "claude-haiku-5-5", None  # amendment 2: Claude Haiku 5.5 at its default sampling
LOCAL_MODELS = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.6-35b-a3b"]  # capability ablation, smallest first
ALPHA = 0.05


def load_runs(max_round: int) -> list[dict]:
    """All graded agent runs up to and including round max_round (stage 0 dev runs excluded)."""
    out = []
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("stage") != "dev" and 1 <= int(r.get("round", 0)) <= max_round:
            out.append(r)
    return out


def _same_temp(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9)


def _select(runs, *, target="T0", level=None, model=MAIN_MODEL, temperature=MAIN_TEMP, variant="", stage=None):
    """Runs of one condition; local models are matched at their own (preregistered 0.6) temperature."""
    temp = temperature if model == MAIN_MODEL else 0.6
    sel = [r for r in runs if r["target"] == target and r["model"] == model and _same_temp(r.get("temperature"), temp)
           and r.get("variant", "") == variant and (level is None or r["level"] == level) and (stage is None or r["stage"] == stage)]
    return sel


def _vals(runs, key="hs"):
    return np.array([r[key] for r in runs], float)


def _n_label(n: int) -> str:
    return {1: "at N=1 — not yet N=3 or N=10", 3: "at N=3 — not yet N=10", 10: "at N=10"}.get(n, f"at N={n}")


def compiler(target: str, level: str) -> dict | None:
    p = RES / "compiler" / target / f"{level}.json"
    return json.loads(p.read_text()) if p.exists() else None


def evaluate(max_round: int) -> dict:
    """All hypothesis verdicts using runs up to round max_round."""
    runs = load_runs(max_round)
    out: dict = {"round": max_round, "n_runs": len(runs), "hypotheses": {}}
    H = out["hypotheses"]
    levels = ["E1", "E2", "E3", "E4"]
    groups = [_vals(_select(runs, level=lv)) for lv in levels]
    n = min((len(g) for g in groups), default=0)
    # H1 — trend over E1 < E2 < E3 < E4 on HS (increasing) and on −KL (decreasing KL)
    if n >= 1:
        jt_hs = st.jonckheere(groups)
        kl_groups = [-_vals(_select(runs, level=lv), "kl_out") for lv in levels]
        jt_kl = st.jonckheere(kl_groups)
        lv_idx = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        sp = st.spearman_ci(lv_idx, np.concatenate(groups), groups)
        sp_kl = st.spearman_ci(lv_idx, np.concatenate(kl_groups), kl_groups)

        def trend_ok(jt, s):  # §5: one-sided trend p < α and Spearman ρ > 0 with its CI above 0
            return jt["p"] < ALPHA and s["rho"] > 0 and (s["ci"] is None or s["ci"][0] > 0)
        # §5 tests HS and KL_out "separately, same rule" (deviation 7: both parts are reported, as preregistered)
        H["H1"] = {"N": n, "label": _n_label(n),
                   "verdicts": {"HS": "supported" if trend_ok(jt_hs, sp) else "not supported",
                                "KL_out": "supported" if trend_ok(jt_kl, sp_kl) else "not supported"},
                   "jt_hs": jt_hs, "jt_neg_kl": jt_kl, "spearman_hs": sp, "spearman_neg_kl": sp_kl,
                   "means_hs": {lv: float(g.mean()) for lv, g in zip(levels, groups)},
                   "means_kl": {lv: float(-g.mean()) for lv, g in zip(levels, kl_groups)}}
    # H2 — true vs length-matched false explanations on HS
    pairs = {"E3>W-freq": ("E3", "W-freq"), "E3>W-alg": ("E3", "W-alg"), "E3>W-role": ("E3", "W-role"), "E4>W-scramble": ("E4", "W-scramble")}
    tests, effects, nmin = {}, {}, 10**9
    for name, (a, b) in pairs.items():
        xa, xb = _vals(_select(runs, level=a)), _vals(_select(runs, level=b))
        nmin = min(nmin, len(xa), len(xb))
        tests[name] = st.mann_whitney_greater(xa, xb)
        effects[name] = st.bootstrap_mean_diff(xa, xb) if len(xa) and len(xb) else None
    if nmin < 10**9 and nmin >= 1:
        adj = st.holm({k: v["p"] for k, v in tests.items()})
        if nmin < 2:
            verdicts = {k: ("cannot tell at this N" if effects[k] is None or effects[k]["diff"] > 0 else "not supported") for k in pairs}
        else:  # §6 / deviation 1: a test whose smallest attainable p is ≥ α cannot decide (3 vs 3: 1/20); deviation 7
            verdicts = {k: st.verdict(adj[k], effects[k]["diff"] > 0,
                                      st.min_p_mann_whitney(len(_select(runs, level=a)), len(_select(runs, level=b))))
                        for k, (a, b) in pairs.items()}
        H["H2"] = {"N": nmin, "label": _n_label(nmin), "verdicts": verdicts, "tests": tests, "holm_p": adj, "effects": effects}
    # H3 — W-freq fools output KL but not HS
    wf = _select(runs, level="W-freq")
    trues = {lv: _select(runs, level=lv) for lv in ("E2", "E3", "E4")}
    if wf and all(trues.values()):
        n3 = min(len(wf), *(len(v) for v in trues.values()))
        kl_means = {lv: float(_vals(v, "kl_out").mean()) for lv, v in trues.items()}
        hs_means = {lv: float(_vals(v).mean()) for lv, v in trues.items()}
        wf_kl, wf_hs, wf_agree = float(_vals(wf, "kl_out").mean()), float(_vals(wf).mean()), float(_vals(wf, "argmax_agreement").mean())
        kl_fooled = wf_kl <= max(kl_means.values())
        hs_catches = all(wf_hs < v for v in hs_means.values())
        H["H3"] = {"N": n3, "label": _n_label(n3),
                   "verdict": "supported" if (wf_agree >= 0.99 and kl_fooled and hs_catches) else "not supported",
                   "wfreq": {"kl": wf_kl, "hs": wf_hs, "argmax_agreement": wf_agree}, "true_kl_means": kl_means,
                   "true_hs_means": hs_means, "kl_fooled": kl_fooled, "hs_catches": hs_catches}
    # H4 — E3/E4 vs independent retrains (and two-sided vs distilled students)
    rb = RES / "baselines" / "T0.json"
    if rb.exists():
        base = json.loads(rb.read_text())
        retr = np.array([x["hs"] for x in base.get("retrains", [])], float)
        stud = np.array([x["hs"] for x in base.get("students", [])], float)
        tests = {lv: st.mann_whitney_greater(_vals(_select(runs, level=lv)), retr) for lv in ("E3", "E4")}
        adj = st.holm({k: v["p"] for k, v in tests.items()})
        n4 = min(len(_select(runs, level="E3")), len(_select(runs, level="E4")))
        verd = {lv: st.verdict(adj[lv], True, st.min_p_mann_whitney(len(_select(runs, level=lv)), len(retr))) for lv in tests}
        H["H4"] = {"N": n4, "label": _n_label(n4), "verdicts": verd, "tests": tests, "holm_p": adj,
                   "retrain_hs_mean": float(retr.mean()) if len(retr) else None,
                   "student_hs_mean": float(stud.mean()) if len(stud) else None,
                   "agent_hs_means": {lv: float(_vals(_select(runs, level=lv)).mean()) for lv in ("E3", "E4") if _select(runs, level=lv)}}
    # H4b — planted calibration (deterministic)
    pc = RES / "controls" / "planted_calibration_T0.json"
    if pc.exists():
        rows = json.loads(pc.read_text())["rows"]
        hs = [r["hs"] for r in rows]
        kl = [r["kl_out"] for r in rows]
        ok = abs(hs[0] - 1) < 1e-6 and kl[0] < 1e-9 and all(a > b for a, b in zip(hs, hs[1:])) and all(a < b for a, b in zip(kl, kl[1:]))
        H["H4b"] = {"N": "deterministic", "verdict": "supported" if ok else "not supported", "rows": rows}
    # H5/H6 — simulator gap and invariance across simulators of increasing capability (amendment 2: local Qwens + Haiku)
    lv4 = ("E2", "E3", "E4", "W-freq")
    models = LOCAL_MODELS + [MAIN_MODEL]
    gaps, means = {}, {}
    for m in models:
        for lv in lv4:
            rs = _select(runs, level=lv, model=m)
            c = compiler("T0", lv)
            if rs and c:
                gaps.setdefault(m, []).extend((c["hs"] - _vals(rs)).tolist())
                means.setdefault(m, {})[lv] = float(_vals(rs).mean())
    if len(gaps) == len(models):
        n5 = min(len(_select(runs, level=lv, model=m)) for m in models for lv in lv4)
        jt = st.jonckheere([-np.array(gaps[m]) for m in models])  # gap should shrink: −gap increases with capability
        exceed = {lv: means[MAIN_MODEL][lv] - compiler("T0", lv)["hs"] for lv in lv4}
        H["H5"] = {"N": n5, "label": _n_label(n5), "verdict": "supported" if jt["p"] < ALPHA and max(exceed.values()) <= 0 else "not supported",
                   "jt_gap": jt, "agent_minus_compiler_main": exceed, "mean_gap": {m: float(np.mean(g)) for m, g in gaps.items()}}
        if all(len(means[m]) == 4 for m in models):
            kw = st.kendalls_w(np.array([[means[m][lv] for lv in lv4] for m in models]))
            H["H6"] = {"N": n5, "label": _n_label(n5), "verdict": "supported" if kw["p"] < ALPHA else "not supported", **kw,
                       "means": means}
    else:  # deviation 5: the local simulators were stopped after every run was at chance, so the tests cannot run
        local = [r for r in runs if r["model"] in LOCAL_MODELS]
        desc = {m: {"runs": len(rs), "above_chance": sum(r["accuracy"] > 2 / 113 for r in rs),
                    "mean_hs": float(np.mean([r["hs"] for r in rs])) if rs else None}
                for m in LOCAL_MODELS if (rs := [r for r in local if r["model"] == m])}
        main_gap = {lv: means[MAIN_MODEL][lv] - compiler("T0", lv)["hs"] for lv in lv4 if lv in means.get(MAIN_MODEL, {})}
        label = f"{len(local)} local runs, " + ", ".join(f"{m}: {d['above_chance']}/{d['runs']} above chance" for m, d in desc.items())
        for k in ("H5", "H6"):
            H[k] = {"N": len(local), "label": label, "verdict": "not evaluable (local simulators stopped at chance, deviation 5)",
                    "local": desc, "agent_minus_compiler_main": main_gap}
    # H7 — specificity (wrong targets), H8 — trajectory
    for key, name in (("H7", "wrong_targets_T0.json"), ("H8", "trajectory_T0.json")):
        p = RES / "controls" / name
        if p.exists():
            H[key] = json.loads(p.read_text())
    if "H7" in H:  # verdict from the stored rows, with the power check (Wilcoxon over n runs: smallest p = 1/2^n)
        d = np.array([row["d"] for row in H["H7"]["rows"]], float)
        w = st.wilcoxon_greater(d)
        n_cond = min(len(_select(runs, level="E3")), len(_select(runs, level="E4")))  # runs per condition, not replacements
        H["H7"].update({"N": len(d), "label": f"{len(d)} E3/E4 replacements ({_n_label(n_cond)})", "wilcoxon": w,
                        "positive_d": int((d > 0).sum()),
                        "verdict": st.verdict(w["p"], bool(len(d)) and float(np.median(d)) > 0, w["min_p"])})
    return out


def write(max_round: int) -> Path:
    path = RES / f"hypotheses_round{max_round}.json"
    path.write_text(json.dumps(evaluate(max_round), indent=2, default=float), encoding="utf-8", newline="\n")
    return path
