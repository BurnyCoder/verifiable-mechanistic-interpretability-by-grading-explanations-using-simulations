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


def round_trajectory(data: dict, run_ids: set[str]) -> dict:
    """Reconstruct the requested cohort from archived HS vectors on one training trajectory."""
    steps = data["steps"]
    loss = np.asarray(data["restricted_loss"], dtype=float)
    if len(steps) != len(loss) or not np.all(np.isfinite(loss) & (loss > 0)):
        raise ValueError("Invalid checkpoint or restricted-loss series")
    series = -np.log(loss)
    def selected(name):
        rows = []
        for row in data[name]:
            if row["run_id"] in run_ids:
                if len(row["hs"]) != len(steps):
                    raise ValueError(f"Incomplete HS trajectory for {row['run_id']}")
                rows.append({**row, "rho": float(st.stats.spearmanr(row["hs"], series).statistic)})
        return rows
    reps, controls = selected("replacements"), selected("negative_controls")
    if not reps:
        return {}
    by_level = defaultdict(list)
    for row in controls:
        by_level[row["level"]].append(row["rho"])
    rhos = [r["rho"] for r in reps]
    means = np.mean([r["hs"] for r in reps], axis=0)
    return {"N": f"{len(reps)} E3/E4 replacements × {len(steps)} checkpoints", "steps": steps,
            "restricted_loss": data["restricted_loss"], "replacements": reps, "negative_controls": controls,
            "negative_controls_by_level": dict(by_level), "spearman_min": min(rhos),
            "spearman_hs_vs_neglog_restricted_loss": float(np.median(rhos)),
            "rows": [{"step": s, "hs": float(m), "restricted_loss": l} for s, m, l in zip(steps, means, loss)],
            "verdict": "supported" if all(r > 0 for r in rhos) else "not supported"}


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
        # §5 tests HS and KL_out "separately, same rule" (deviation 7: both parts are reported, as preregistered).
        # H1's hypothesis also states "E0 … lies below E1"; §5 preregisters no test for that clause, but "supported
        # requires the stated direction", so the clause gets its own direction-only verdict (deviation 10, audit 3).
        e0 = _vals(_select(runs, level="E0"))
        e0_clause = ("not run" if not len(e0) else "observed (direction only, no test preregistered)"
                     if e0.mean() < groups[0].mean() else "not observed (E0 above E1; direction only, no test preregistered)")
        H["H1"] = {"N": n, "label": _n_label(n),
                   "verdicts": {"HS trend E1→E4": "supported" if trend_ok(jt_hs, sp) else "not supported",
                                "E0 below E1": e0_clause,
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
            verdicts = dict.fromkeys(pairs, "cannot tell at this N")
            prereg = dict(verdicts)  # §6 fixes this label irrespective of the single observed difference.
        else:  # deviation 1 (post hoc): a test whose smallest attainable p is ≥ α cannot decide (3 vs 3: 1/20); deviation 7
            verdicts = {k: st.verdict(adj[k], effects[k]["diff"] > 0,
                                      st.min_p_mann_whitney(len(_select(runs, level=a)), len(_select(runs, level=b))))
                        for k, (a, b) in pairs.items()}
            prereg = {k: st.verdict_prereg(adj[k], effects[k]["diff"] > 0) for k in pairs}  # §5 as written (audit 8)
        H["H2"] = {"N": nmin, "label": _n_label(nmin), "verdicts": verdicts, "verdicts_prereg": prereg, "tests": tests,
                   "holm_p": adj, "effects": effects}
    # H3 — W-freq fools output KL but not HS
    wf = _select(runs, level="W-freq")
    trues = {lv: _select(runs, level=lv) for lv in ("E2", "E3", "E4")}
    if wf and all(trues.values()):
        n3 = min(len(wf), *(len(v) for v in trues.values()))
        kl_means = {lv: float(_vals(v, "kl_out").mean()) for lv, v in trues.items()}
        hs_means = {lv: float(_vals(v).mean()) for lv, v in trues.items()}
        wf_kl, wf_hs, wf_agree = float(_vals(wf, "kl_out").mean()), float(_vals(wf).mean()), float(_vals(wf, "argmax_agreement").mean())
        kl_fooled = wf_kl <= max(kl_means.values())  # §5's falsifier: KL above the largest E2–E4 mean
        hs_catches = all(wf_hs < v for v in hs_means.values())
        within = min(kl_means.values()) <= wf_kl <= max(kl_means.values())
        H["H3"] = {"N": n3, "label": _n_label(n3),
                   "verdict": "supported" if (wf_agree >= 0.99 and within and hs_catches) else "not supported",
                   "upper_bound_reading": "supported" if (wf_agree >= 0.99 and kl_fooled and hs_catches) else "not supported",
                   "wfreq": {"kl": wf_kl, "hs": wf_hs, "argmax_agreement": wf_agree}, "true_kl_means": kl_means,
                   "true_hs_means": hs_means, "kl_fooled": kl_fooled, "hs_catches": hs_catches,
                   # §5's hypothesis wording also asks for KL within [min, max] of the E2–E4 means (reported, not a falsifier)
                   "kl_within_range": within,
                   # §5: "HS difference to each of E2–E4 with bootstrap CI" (deviation 8: was not computed before)
                   "hs_diff": {lv: st.bootstrap_mean_diff(_vals(wf), _vals(v)) for lv, v in trues.items()}}
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
        # §5 secondary, two-sided, no predicted direction: E3/E4 vs the distilled students (deviation 8; they match output probabilities, audit 9), Holm over both
        two = {lv: st.mann_whitney_two_sided(_vals(_select(runs, level=lv)), stud) for lv in ("E3", "E4")} if len(stud) else {}
        adj2 = st.holm({k: v["p"] for k, v in two.items()})
        for lv, t in two.items():
            t["holm_p"] = adj2[lv]
            # Audit 9: "not significant", never "no significant difference": a test that cannot reach α (two-sided
            # 3 vs 3: smallest p 0.10) says nothing about equality (Altman & Bland 1995, BMJ 311:485), so the smallest
            # attainable p and the observed separation are stated; deviation 1's "cannot tell" follows.
            sep = ("every agent run above every student" if t["r_rb"] == 1 else "every student above every agent run"
                   if t["r_rb"] == -1 else f"rank-biserial r {t['r_rb']:+.2f}")
            t["result"] = ("cannot tell at this N (protocol §6)" if n4 == 1 or t["p"] is None
                           else ("differ: " + ("agent higher" if t["r_rb"] > 0 else "students higher")) if adj2[lv] < ALPHA
                           else f"not significant (Holm p {adj2[lv]:.2f}; smallest attainable p {t['min_p']:.2f}; {sep}"
                           + ("; post-hoc rule: cannot tell at this N)" if t["min_p"] >= ALPHA else ")"))
        H["H4"] = {"N": n4, "label": _n_label(n4), "verdicts": verd, "tests": tests, "holm_p": adj, "students": two,
                   "retrain_hs_mean": float(retr.mean()) if len(retr) else None,
                   "student_hs_mean": float(stud.mean()) if len(stud) else None,
                   "agent_hs_means": {lv: float(_vals(_select(runs, level=lv)).mean()) for lv in ("E3", "E4") if _select(runs, level=lv)}}
    # H4b — planted calibration (deterministic)
    pc = RES / "controls" / "planted_calibration_T0.json"
    if pc.exists():
        rows = json.loads(pc.read_text())["rows"]
        hs = [r["hs"] for r in rows]
        kl = [r["kl_out"] for r in rows]
        ok = hs[0] == 1 and kl[0] == 0 and all(a > b for a, b in zip(hs, hs[1:])) and all(a < b for a, b in zip(kl, kl[1:]))
        H["H4b"] = {"N": "deterministic", "verdict": "supported" if ok else "not supported", "rows": rows,
                    "identity_exact": hs[0] == 1 and kl[0] == 0,
                    "identity_with_numerical_tolerance": abs(hs[0] - 1) < 1e-6 and abs(kl[0]) < 1e-9}
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
    if all(len(means.get(m, {})) == len(lv4) for m in LOCAL_MODELS) and len(means.get(MAIN_MODEL, {})) == len(lv4):
        n5 = min(len(_select(runs, level=lv, model=m)) for m in LOCAL_MODELS for lv in lv4)
        jt = st.jonckheere([-np.array(gaps[m]) for m in LOCAL_MODELS])  # the registered size trend compares the three local simulators
        exceed = {lv: means[MAIN_MODEL][lv] - compiler("T0", lv)["hs"] for lv in lv4}
        # §5 states a mean inequality but its falsifier asks whether the compiler is outside the agent interval.
        # The two readings remain separate even if an agent mean barely exceeds the comparator.
        intervals = {}
        for lv in lv4:
            scores = _vals(_select(runs, level=lv))
            reference = np.full(len(scores), compiler("T0", lv)["hs"])
            intervals[lv] = st.bootstrap_mean_diff(scores, reference)["ci"]
        ci_verdict = None if any(ci is None for ci in intervals.values()) else jt["p"] < ALPHA and all(ci[0] <= 0 for ci in intervals.values())
        H["H5"] = {"N": n5, "label": _n_label(n5), "gap_intervals": intervals, "passes_ci_reading": ci_verdict, "verdict": "supported" if jt["p"] < ALPHA and max(exceed.values()) <= 0 else "not supported",
                   "jt_gap": jt, "agent_minus_compiler_main": exceed, "mean_gap": {m: float(np.mean(g)) for m, g in gaps.items()}}
        if all(len(means[m]) == 4 for m in models):
            kw = st.kendalls_w(np.array([[means[m][lv] for lv in lv4] for m in LOCAL_MODELS]))
            H["H6"] = {"N": n5, "label": _n_label(n5), "verdict": st.verdict_prereg(kw["p"], True), **kw,
                       "means": means}
    else:  # deviation 5: the local simulators were stopped after every run was at chance, so the tests cannot run
        local = [r for r in runs if r["model"] in LOCAL_MODELS]
        desc = {m: {"runs": len(rs), "above_chance": sum(r["accuracy"] > 1 / 113 for r in rs),
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
            data = json.loads(p.read_text())
            ids = {r["run_id"] for r in runs}
            if key == "H7":
                rows = [r for r in data["rows"] if r["run_id"] in ids]
                H[key] = {**data, "N_runs": len(rows), "rows": rows}
            elif "replacements" in data:
                cohort = round_trajectory(data, ids)
                if cohort:
                    H[key] = cohort
                    archive = RES / ("controls/trajectory_T0_round1_historical.json" if max_round == 1 else "controls/trajectory_T0_superseded.json")
                    if archive.exists():
                        hist = json.loads(archive.read_text())
                        H[key]["historical"] = {"source": hist.get("source", archive.relative_to(RES).as_posix()),
                                               "checkpoints": len(hist["rows"]), "replacement": hist["replacement"],
                                               "rho": hist["spearman_hs_vs_neglog_restricted_loss"]}
            elif data.get("replacement") in ids:
                H[key] = {**data, "N": f"one replacement × {len(data['rows'])} sampled checkpoints"}
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
