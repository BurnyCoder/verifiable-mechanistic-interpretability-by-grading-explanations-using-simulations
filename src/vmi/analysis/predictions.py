"""Preregistered predictions next to what the main simulator produced (one row per hypothesis), plus replications.

Local explanation: `prediction_rows(r)` reads results/hypotheses_round{r}.json and the no-agent result files and, for
every hypothesis of preregistration §5, states the prediction in words, the observed numbers, and whether the observed
direction matches the prediction ("yes" / "partly" / "no", with the reason). The "matches" column is descriptive: it
compares directions and preregistered thresholds, not significance; the formal verdict (which does) is shown beside it.
`replication_rows(r)` re-checks H1–H3 on the seed-sensitivity targets, as §5 preregisters ("re-test H1–H3 on those
targets as replications, reported per target with the same tests").
Global context: rendered as markdown by `tables.predictions_table` / `tables.replications_table` into README, summary,
paper and round report, and merged into docs/data.json for the results explorer, so no number here is typed by hand.
"""

from __future__ import annotations  # postponed annotations

import json  # result files
import math  # rounding a lower bound down

import numpy as np  # means

from vmi.analysis import logit_fit  # audit 9: logit-level output comparison (post hoc)
from vmi.analysis.hypotheses import LOCAL_MODELS, MAIN_MODEL, _select, compiler, load_runs  # same selection as the tests
from vmi.config import REPO_ROOT  # results folder

RES = REPO_ROOT / "results"
SEED_TARGETS = ["T1", "T2", "T-new", "T-paper"]  # T-paper is contamination-flagged (memorisation, see frequencies)
PREDICTED = {  # wording follows preregistration §5; thresholds are the preregistered constants
    "H1": "HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1",
    "H2": "true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble",
    "H3": "W-freq: argmax agreement ≥ 0.99, KL within the E2–E4 mean range, HS below each true mean; the falsifier tests only the upper KL bound",
    "H4": "E3 and E4 beat independent retrains on HS (vs. distilled students: open question, no direction)",
    "H4b": "planted model: HS = 1 and KL = 0 at step 0, then HS strictly falls and KL strictly rises",
    "H5": "agent ≤ reference compiler on every explanation; the gap shrinks with simulator capability",
    "H6": "the local simulators rank E2, E3, E4, W-freq the same way",
    "H7": "a T0 replacement fits T0 better than any other trained or no-mechanism model (HS)",
    "H8": "HS against T0's checkpoints rises as restricted loss falls (circuit formation)",
}


PLAIN = {  # the same predictions in plain words, for readers outside the field (README top)
    "H1": "The more true detail an explanation gives, the more faithfully the rebuilt network should match the original.",
    "H2": "A true explanation should beat a wrong explanation of the same length.",
    "H3": "A wrong explanation can still give all the right answers; checking only the answers is fooled, checking the "
          "insides (HS) should not be.",
    "H4": "Rebuilding from an explanation should match the original's insides better than training a fresh network the "
          "same way does.",
    "H4b": "On a network whose mechanism we planted ourselves, the score should start perfect and drop as further "
           "training moves the network away from it.",
    "H5": "The AI should not beat our automatic, AI-free compiler, and stronger AIs should come closer to it.",
    "H6": "Different AIs should put the explanations in the same order.",
    "H7": "An explanation of one network should not fit a different network.",
    "H8": "The score should rise while the original network learns its mechanism during training.",
}


def _f(x, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, int):
        return str(x)
    return f"{x:.2e}" if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4) else f"{x:.{nd}f}"


def _mono(xs, increasing: bool = True) -> bool:
    return all((a < b) if increasing else (a > b) for a, b in zip(xs, xs[1:]))


def _match(checks: dict[str, bool]) -> str:
    """'yes' if every check holds, 'no' if none does, else 'partly', followed by the checks that failed."""
    failed = [k for k, ok in checks.items() if not ok]
    if not failed:
        return "yes"
    return ("no" if len(failed) == len(checks) else "partly") + " (" + "; ".join(failed) + ")"


def _verdict(h: dict) -> str:
    """Formal verdict text, preregistered reading first.

    Audit 8: preregistration §5 reserves "cannot tell at this N" for a test that is undefined at the current N and
    falsifies a comparison that is "not significant after Holm"; at N = 3 a 3-vs-3 exact test is defined but cannot be
    significant (§6 notes this and grants no exception), so §5's verdict is "not supported". The "cannot tell" rule is
    deviation 1's (post hoc, applied at N = 3 by deviation 7 after the N = 3 verdicts were first read) and is shown
    second. Audits 6–7 had shown the literal reading only in a parenthesis after "cannot tell"."""
    if h.get("verdict"):
        return h["verdict"]
    holm_p, verd = h.get("holm_p") or {}, h.get("verdicts") or {}
    powerless = [k for k, v in verd.items() if v == "cannot tell at this N" and holm_p.get(k) is not None and holm_p[k] >= 0.05]
    if not (powerless and isinstance(h.get("N"), int) and h["N"] > 1):
        return "; ".join(f"{a}: {b}" for a, b in verd.items())
    from math import comb
    best = len(holm_p) / comb(2 * h["N"], h["N"])  # Holm p of a complete separation (m × smallest one-sided p)
    prereg = h.get("verdicts_prereg") or {k: ("not supported" if k in powerless else v) for k, v in verd.items()}
    return ("; ".join(f"{a}: {b}" + (f" (Holm p {_f(holm_p[a], 2)})" if a in powerless else "") for a, b in prereg.items())
            + f" — preregistered §5 criterion; at N = {h['N']} even complete separation gives Holm p {_f(best, 2)}, so "
            + ("these tests" if len(powerless) > 1 else "this test") + " could not reach α. Post-hoc rule of deviation 1 "
            + "(applied at this N by deviation 7): " + ("all read" if len(powerless) == len(verd) else ", ".join(powerless) + " read")
            + " \"cannot tell at this N\"")


def _mean_hs(runs, **kw):
    rs = _select(runs, **kw)
    return float(np.mean([r["hs"] for r in rs])) if rs else None


def h5_gaps(runs) -> dict[str, list[dict]]:
    """Agent minus reference-compiler HS per (target, explanation) on H5's explanations {E2, E3, E4, W-freq}, grouped by
    what the comparison is: "T0" (the preregistered H5 test — §5's tests use the main simulator on T0), "seed" (T1, T2,
    T-new: an extension, post hoc) and "T-paper" (contamination-flagged: memorisation can lift the agent, so it is
    reported apart and not counted). `beyond` applies §5's falsifier: the agent mean exceeds the compiler by more than
    the agent mean's nominal 95 % percentile interval. With three observations its exact endpoints are their extrema.
    Audit 2026-11-22 (second pass): earlier versions pooled T0 with the seed targets and silently dropped T-paper."""
    from vmi.analysis import stats as st
    out: dict[str, list[dict]] = {"T0": [], "seed": [], "T-paper": []}
    for t in ["T0"] + SEED_TARGETS:
        group = "T0" if t == "T0" else "T-paper" if t == "T-paper" else "seed"
        for lv in ("E2", "E3", "E4", "W-freq"):
            rs, c = _select(runs, target=t, level=lv), compiler(t, lv)
            if rs and c:
                v = np.array([r["hs"] for r in rs])
                ci = [v.min(), v.max()] if len(v) == 3 else st.bootstrap_mean_diff(v, np.zeros(2))["ci"]
                gap = float(v.mean() - c["hs"])
                out[group].append({"target": t, "level": lv, "gap": gap,
                                   "beyond": None if ci is None else bool(gap > 0 and ci[0] > c["hs"])})
    return out


def h5_summary(g: dict[str, list[dict]]) -> dict[str, str]:
    """Words for the agent-versus-compiler result, used in the predictions table and in prose (`v:h5-*` values)."""
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    w = lambda n: words.get(n, str(n))  # noqa: E731
    above = lambda rows: [r for r in rows if r["gap"] > 0]  # noqa: E731
    beyond = lambda rows: [r for r in rows if r["beyond"]]  # noqa: E731
    t0, seed, tp = g["T0"], g["seed"], g["T-paper"]
    top = max(seed, key=lambda r: r["gap"]) if seed else None
    n_t = len({r["target"] for r in beyond(seed)})
    return {
        "t0": (f"{w(len(beyond(t0)))} of the {w(len(t0))} explanations" if beyond(t0) else
               f"none of the {w(len(t0))} explanations") + " beyond the bootstrap CI",
        "seed": (f"{w(len(beyond(seed)))} of {w(len(seed))} target–explanation pairs on {w(n_t)} target{'s' if n_t != 1 else ''}"
                 if seed else "no pairs"),
        "seed-sign": f"{w(len(above(seed)))} of {w(len(seed))}",
        "max": f"{top['gap']:+.3f} ({top['target']} {top['level']})" if top else "–",
        "list": ", ".join(f"{r['target']} {r['level']}" for r in beyond(seed)) or "none",
        # audit 8: margins under 0.01 HS, which "beyond the range of three runs" still counts (about 1 in 8 by chance)
        "small": ", ".join(f"{r['target']} {r['level']} {r['gap']:+.3f}" for r in beyond(seed) if r["gap"] < 0.01) or "none",
        "tpaper": ", ".join(f"{r['level']} {r['gap']:+.3f}" + (" beyond the CI" if r["beyond"] else "") for r in tp) or "–",
    }


def prediction_rows(max_round: int) -> list[dict]:
    p = RES / f"hypotheses_round{max_round}.json"
    if not p.exists():
        return []
    H = json.loads(p.read_text())["hypotheses"]
    runs = load_runs(max_round)
    out = []

    def add(k, observed, matches):
        h = H.get(k, {})
        out.append({"id": k, "plain": PLAIN[k], "predicted": PREDICTED[k], "observed": observed, "matches": matches,
                    "verdict": _verdict(h) or "not evaluated yet", "evidence": str(h.get("label", h.get("N", "–")))})

    if "H1" in H:
        mh, mk, e0 = H["H1"]["means_hs"], H["H1"]["means_kl"], _mean_hs(runs, level="E0")
        obs = "HS " + ", ".join(f"{lv} {_f(v)}" for lv, v in ([("E0", e0)] if e0 is not None else []) + list(mh.items()))
        obs += "; KL " + ", ".join(f"{lv} {_f(v)}" for lv, v in mk.items())
        add("H1", obs, _match({"HS not monotone E1→E4": _mono(list(mh.values())),
                               "KL does not fall E1→E4": _mono(list(mk.values()), increasing=False),
                               "E0 not below E1": e0 is None or e0 < mh["E1"]}))
    if "H2" in H:
        eff = {k: v for k, v in H["H2"]["effects"].items() if v}

        def status(v):  # with ≥ 2 runs per side the bootstrap CI decides; with 1 run only the sign is known
            ci = v.get("ci")
            if ci and ci[0] <= 0 <= ci[1]:
                return "direction uncertain (CI includes 0)"
            return "ahead" if v["diff"] > 0 else "reversed"
        add("H2", "HS difference " + "; ".join(f"{k} {v['diff']:+.3f}" + (f" [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" if v.get("ci") else "")
                                               for k, v in eff.items()),
            _match({f"{k} {status(v)}": status(v) == "ahead" for k, v in eff.items()}))
    if "H3" in H:
        h = H["H3"]
        w, tk, th = h["wfreq"], list(h["true_kl_means"].values()), list(h["true_hs_means"].values())
        dd = h.get("hs_diff", {})  # deviation 8: §5's HS difference to each of E2–E4 with bootstrap CI
        gap = ("; HS − " + ", ".join(f"{lv} {v['diff']:+.3f}" + (f" [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" if v.get("ci") else "")
                                     for lv, v in dd.items())) if dd else ""
        add("H3", f"W-freq: agreement {_f(w['argmax_agreement'])}, KL {_f(w['kl'])} (true {_f(min(tk))}–{_f(max(tk))}), "
                  f"HS {_f(w['hs'])} (true {_f(min(th))}–{_f(max(th))})" + gap,
            _match({"agreement < 0.99": w["argmax_agreement"] >= 0.99, "KL outside the true range": h["kl_within_range"],
                    "HS not below every true explanation": h["hs_catches"]}))
    b = RES / "baselines" / "T0.json"
    if "H4" in H and b.exists():
        d = json.loads(b.read_text())
        rt, st = [x["hs"] for x in d["retrains"]], [x["hs"] for x in d["students"]]
        ag = H["H4"]["agent_hs_means"]
        two = H["H4"].get("students", {})  # deviation 8: §5's secondary two-sided test against the students
        add("H4", ", ".join(f"{lv} {_f(v)}" for lv, v in ag.items()) + f"; {len(rt)} retrains mean {_f(np.mean(rt))} "
                  f"(max {_f(max(rt))}); {len(st)} students mean {_f(np.mean(st))} (max {_f(max(st))})"
                  + ("; vs students, two-sided: " + ", ".join(f"{lv} {t['result']}" for lv, t in two.items()) if two else ""),
            _match({"not above every retrain": min(ag.values()) > max(rt)}))
    if "H4b" in H:
        rows = H["H4b"]["rows"]
        hs, kl = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
        add("H4b", "HS " + " → ".join(_f(x) for x in hs) + f" (step 0 unrounded: {hs[0]:.12f}); KL " + " → ".join(_f(x) for x in kl),
            _match({"step-zero HS differs from exact identity": hs[0] == 1, "HS not strictly falling": _mono(hs, increasing=False),
                    "KL not strictly rising": kl[0] < 1e-9 and _mono(kl)}))
    g5 = h5_gaps(runs)
    if g5["T0"]:
        s5 = h5_summary(g5)
        obs = ("T0 (preregistered): agent − compiler " + ", ".join(f"{r['level']} {r['gap']:+.3f}" for r in g5["T0"])
               + f", {s5['t0']}; seed targets T1, T2, T-new (extension, post hoc): agent above compiler in {s5['seed-sign']} "
               f"pairs by sign, {s5['seed']} beyond the bootstrap CI ({s5['list']}; largest {s5['max']}; margins under "
               f"0.01: {s5['small']}); "
               f"T-paper (memorisation-flagged, not counted): {s5['tpaper']}")
        local = [r for r in runs if r["model"] in LOCAL_MODELS]
        if all(r["beyond"] is None for r in g5["T0"]):
            obs = "T0 agent minus compiler HS: " + ", ".join(f"{r['level']} {r['gap']:+.3f}" for r in g5["T0"]) + ". One run per condition; interval unavailable"
        add("H5", obs + f"; {len(local)} local runs, {sum(r['accuracy'] > 1 / 113 for r in local)} above chance",
            _match({"T0: agent above compiler beyond its CI": not any(r["beyond"] for r in g5["T0"]),
                    "seed targets: agent above compiler beyond its CI": not any(r["beyond"] for r in g5["seed"]),
                    "capability trend not testable": False}))
        add("H6", f"{len(local)} local runs, {sum(r['accuracy'] > 1 / 113 for r in local)} above uniform-guess accuracy. The three-model ranking matrix is incomplete",
            "not testable")
    if "H7" in H:
        rows = H["H7"]["rows"]
        obs = "; ".join(f"{r['level']}: HS {_f(r['hs_T0'])} vs best other {_f(max(r['hs_wrong'].values()))} "
                        f"({max(r['hs_wrong'], key=r['hs_wrong'].get)})" for r in rows)
        add("H7", obs, _match({"a wrong target scores as high": all(r["d"] > 0 for r in rows)}))
    if "H8" in H:
        h = H["H8"]
        rho, lo = h["spearman_hs_vs_neglog_restricted_loss"], h.get("spearman_min", h["spearman_hs_vs_neglog_restricted_loss"])
        ctrl = h.get("negative_controls_by_level", {})  # post hoc (audit 2026-11-22, second pass): specificity of ρ
        # Audit 4 positive control (post hoc): final T0 itself and agent-free builds against the same checkpoints
        pcf = RES / "audit4" / "h8_positive_control.json"
        pc = json.loads(pcf.read_text(encoding="utf-8")) if pcf.exists() else {}
        names = {"final T0 (weights dump)": "final T0 itself", "Fourier-restricted T0": "Fourier-restricted T0",
                 "reference compiler E3 (no agent)": "compiler E3", "independent retrain wd1_s0": "a retrain"}
        hist = h.get("historical")
        history = (f"Historical round {max_round}: one selected replacement, {hist['checkpoints']} checkpoints, "
                   f"ρ {_f(hist['rho'])}. Retrospective full-checkpoint analysis: " if hist else "")
        add("H8", history + f"Spearman ρ(HS, −log restricted loss) per replacement: median {_f(rho)}, min {_f(lo)} ({h['N']})"
                  + ("; negative controls (post hoc), ρ per replacement: " + "; ".join(
                      f"{lv} " + ", ".join(f"{x:+.2f}" for x in v) for lv, v in ctrl.items()) if ctrl else "")
                  + ("; reference networks (post hoc, audit 4): " + ", ".join(
                      f"{names[k]} {x:+.2f}" for k, x in pc["rho_vs_neglog_restricted_loss"].items())
                     + f"; during circuit formation (steps ≤ {pc['phases']['plateau_start_step']:,}) every network with "
                       # audit 5: a lower bound is rounded down (0.99668 → 0.996, not 0.997)
                       f"T0's frequencies has ρ ≥ {math.floor(1000 * min(x for k, x in pc['phases']['rho_formation'].items() if 'retrain' not in k)) / 1000:.3f}"
                     if pc and max_round >= 2 else ""),
            _match({"ρ ≤ 0 for some replacement": lo > 0}))
    return out


def highlight_rows(max_round: int, target: str = "T0") -> list[dict]:
    """Concrete examples of what works (✅) and what does not (❌), every number read from results/ (README top)."""
    runs = load_runs(max_round)

    def m(lv, key="hs", t=target, **kw):
        rs = [r for r in _select(runs, target=t, level=lv, **kw) if r.get(key) is not None]
        return float(np.mean([r[key] for r in rs])) if rs else None

    out = []

    def add(ok, title, evidence, means):
        if None not in evidence[1]:
            out.append({"ok": ok, "title": title, "evidence": evidence[0].format(*[_f(x) for x in evidence[1]]), "means": means})

    c = {lv: compiler(target, lv) for lv in ("E2", "E3", "E4")}
    lv9 = logit_fit.inline_values()  # audit 9: logit-level output comparison (post hoc)
    add(True, "Naming the key frequencies is what raises HS",  # audit 8: not "faithful" — E1–E3 rebuilds are Pizza-like
        ("HS {} with the algorithm only (E1) → {} once the frequencies are named (E2); the AI-free compiler gets {} from E2",
         [m("E1"), m("E2"), c["E2"] and c["E2"]["hs"]]),
        "The score rewards the part of the explanation that pins down which frequencies this network uses.")
    add(True, "Right answers from the wrong key frequencies: caught by HS, missed by output KL",
        ("wrong-frequency explanation: accuracy {}, output KL {} (true E3: {}), but HS {} (true E3: {})",
         [m("W-freq", "accuracy"), m("W-freq", "kl_out"), m("E3", "kl_out"), m("W-freq"), m("E3")]),
        "Checking only the answers would rate a wrong explanation as good as a true one."
        + (" The output logits do show it (post hoc): after one fitted scale, their centred-logit FVU is "
           f"{lv9['a9-cal-w-freq']} for W-freq against {lv9['a9-cal-e3']} for E3." if "a9-cal-w-freq" in lv9 else ""))
    b = RES / "baselines" / f"{target}.json"
    ri = RES / "controls" / f"random_init_{target}.json"
    if b.exists() and ri.exists():
        bd, rnd = json.loads(b.read_text()), json.loads(ri.read_text())
        st, rt = bd["students"], bd["retrains"]
        # Audit 9: compare like with like — the students' mean against the mean of the true explanations' main runs
        # (earlier versions took the minimum over every T0 run, which was a wrong-explanation W-freq variant)
        true_kl = [r["kl_out"] for lv in ("E2", "E3", "E4") for r in _select(runs, target=target, level=lv, stage="main")
                   if not r.get("failed")]
        ls = bd.get("students_logit")
        add(True, "Probability-distilled students achieved small output KL with low HS",
            ("networks trained to copy the target's output probabilities: KL {} (true explanations E2–E4: mean {}), HS {}; "
             "random networks: HS {}; networks trained on the task from other seeds: HS {}"
             + ("; networks trained to copy its centred logits (post hoc): HS {}" if ls else ""),
             [np.mean([x["kl_out"] for x in st]), np.mean(true_kl) if true_kl else None, np.mean([x["hs"] for x in st]),
              np.mean([x["hs"] for x in rnd]), np.mean([x["hs"] for x in rt])] + ([np.mean([x["hs"] for x in ls])] if ls else [])),
            "Mean HS for these students was close to the random-initialisation mean. HS measures cross-validated "
            "linear predictability. Random networks had positive scores."
            + (f" Matching the logits (pre-specified reading): {lv9['a9-ls-reading']}." if "a9-ls-reading" in lv9 else ""))
    p = RES / f"hypotheses_round{max_round}.json"
    H = json.loads(p.read_text())["hypotheses"] if p.exists() else {}
    if "H7" in H and H["H7"]["rows"]:
        rows = H["H7"]["rows"]
        add(True, "T0 replacements scored higher against T0 than against every tested wrong target",
            ("T0's E3/E4 rebuilds: HS {} against T0, at most {} against {} other trained networks (other seeds, the "
             "published model, no-mechanism controls)",
             [np.mean([r["hs_T0"] for r in rows]), max(max(r["hs_wrong"].values()) for r in rows), len(rows[0]["hs_wrong"])]),
            "The score was specific to the network that was explained, against networks with other key frequencies "
            "(each trained wrong target shares at most one with T0); same-frequency networks were not among them.")
    # Audit 9: descriptive only — T-new's preregistered replication tests cannot reach α at N = 3 (H1, H2 "not supported")
    add(True, "The gap to the wrong-frequency explanation also appears on a network trained after the protocol was frozen",
        ("T-new: HS {} (E3) and {} (E4) vs {} (wrong frequencies)",
         [m("E3", t="T-new"), m("E4", t="T-new"), m("W-freq", t="T-new")]),
        "Prompts and templates were frozen before this network was trained, so they cannot have been tuned to it. "
        "Descriptive: with three runs per condition its preregistered replication tests cannot reach significance.")
    # Audit 5: "cannot tell" read as a demonstrated inability; at N = 3 the data show only that HS did not separate them.
    add(False, "Did not separate the Clock algorithm from the Pizza algorithm",
        ("Pizza explanation (W-alg): HS {}; true explanation (E3): HS {}", [m("W-alg"), m("E3")]),
        "Both builds use the same frequency features; HS, a linear fit over whole layers, did not separate what is "
        "computed from them (difference within noise at N = 3). Post hoc, Zhong et al.'s output diagnostic finds T0 "
        "Clock-like but most true-explanation rebuilds Pizza-like too, and the rebuilds that are on the Clock side "
        "score no higher HS than the Pizza-side ones (paper §5).")
    a5 = (RES / "audit5" / "checks.json")
    gap = json.loads(a5.read_text(encoding="utf-8"))["compiler_gaps"].get(target) if a5.exists() else None
    add(False, "The HS comparison with a redrawn neuron table was inconclusive",
        ("redrawn table (W-scramble): HS {}; true table (E4): HS {}; matched-neuron correlation {} vs {} does separate them",
         [m("W-scramble"), m("E4"), m("W-scramble", "matched_neuron_corr"), m("E4", "matched_neuron_corr")]),
        "Neuron-level detail needs a neuron-level metric. The agents' builds did not follow the shuffled table into a "
        "lower HS"
        + (f"; the AI-free compiler's literal build of it does score {gap['E4_minus_Wscramble']:.2f} lower on this "
           f"target (not on the other targets; audit 5)." if gap else "."))
    kls = [m(lv, "kl_out") for lv in ("E2", "E3", "E4")]
    backwards = None not in kls and _mono(kls)  # the title states only what the numbers show
    add(False, "Output KL rises from E2 to E4 (post hoc)" if backwards else "Output KL does not rank the explanations",
        ("E2 → E3 → E4: output KL {} → {} → {}, while HS {} → {} → {}",
         [m("E2", "kl_out"), m("E3", "kl_out"), m("E4", "kl_out"), m("E2"), m("E3"), m("E4")]),
         "Grading by output KL alone would have ranked the most detailed true explanation lowest of these three (a post "
         "hoc reading, untested; the preregistered KL trend over E1–E4 is not supported, and E1's KL is the highest).")
    fq = RES / f"frequencies_round{max_round}.json"
    if fq.exists():
        F = json.loads(fq.read_text())
        # E1 is identical on every target, so its runs on T0 and T-paper are replicates of one condition (pooled)
        e1 = [r for r in F["rows"] if r["level"] == "E1" and r["model"] == MAIN_MODEL and r["variant"] == ""
              and "hs_vs_paper" in r and "hs_vs_T0" in r]
        if e1:
            paper = set(F["paper_freqs"])
            listed = sum(bool(r.get("declared_freqs")) and len(set(r["declared_freqs"]) & paper) >= 4 for r in e1)
            # audit 5/8: recall of the published mechanism, cued by ARCHITECTURE.md (the paper's exact architecture) and
            # E1's text (the paper's algorithm); only E0's text outlines the training setup; not necessarily the paper's text
            add(False, "Build scripts listed published frequencies that the prompt omitted",
                ("algorithm-only rebuilds ({} runs on T0 and the published checkpoint; same input text): {} build scripts "
                 "list at least four of the paper's five frequencies; HS {} against T0 but {} against the published "
                 "checkpoint (same weights)",
                 [len(e1), listed, np.mean([r["hs_vs_T0"] for r in e1]), np.mean([r["hs_vs_paper"] for r in e1])]),
                "On a published network, a high score can come from the AI's memory instead of the explanation "
                "(one simulator, one published model).")
    local = [r for r in runs if r["model"] in LOCAL_MODELS]
    if local:
        add(False, "Small local AI models cannot do the task at all",
            ("{} local runs (Qwen3.5-4B, 9B): task accuracy {} (chance = 1/113), HS {}",
             [len(local), np.mean([r["accuracy"] for r in local]), np.mean([r["hs"] for r in local])]),
            "In this harness, with 4-bit weights and a 60-step, 60-minute budget, the small local models never built a "
            "working network; model size, quantisation and harness are not separated.")
    return out


def _trend_numbers(groups):
    """Preserve numeric evidence alongside the human-readable trend summary."""
    from vmi.analysis import stats as st
    jt = st.jonckheere(groups)
    lv = np.repeat(np.arange(len(groups)), list(map(len, groups)))
    sp = st.spearman_ci(lv, np.concatenate(groups), groups)
    return {**jt, **sp, "supported": jt["p"] < 0.05 and sp["rho"] > 0 and (sp["ci"] is None or sp["ci"][0] > 0)}


def _trend_test(groups: list[np.ndarray]) -> str:
    return _trend_text(_trend_numbers(groups))


def _trend_text(result) -> str:
    """§5's H1 rule on one target: one-sided Jonckheere–Terpstra p < α and Spearman ρ's bootstrap CI above 0, else "not
    supported". Audit 8: when even a perfect order could not reach p < α (smallest exact p = 1 / number of distinct
    orderings), deviation 1's post-hoc rule reads "cannot tell at this N"; it is shown after the preregistered verdict."""
    v = "supported" if result["supported"] else "not supported"
    post_hoc = "; post-hoc rule: cannot tell at this N" if 1 / result["orderings"] >= 0.05 else ""
    return f"{v} (JT p {_f(result['p'])}, ρ {_f(result['rho'], 2)}{post_hoc})"


def replication_rows(max_round: int) -> list[dict]:
    """H1–H3 re-tested per seed-sensitivity target with the preregistered tests (main simulator): H1 by the trend test
    over the true levels each target has, H2 by the one-sided Mann–Whitney E3 > W-freq (the only wrong explanation run
    on these targets), H3 by its rule on condition means.

    The first row is T0 restricted to E2–E4 (post hoc, audit 2026-11-22 second pass): T1, T2 and T-new ran no E1, so
    their trend test covers only the fine E2 → E4 part of the ladder; the T0 row shows what the same test gives there.
    Audit 3 (deviation 10): the trend test uses every ladder level E1–E4 a target ran, as §5's "same tests" require;
    T-paper ran E1, so its test is E1 < E3 < E4 (earlier versions dropped E1 and tested E3 < E4 only). H3's "true
    range" stays E2–E4, as §5 defines it."""
    from vmi.analysis import stats as st
    runs = load_runs(max_round)
    out = []
    for t in ["T0"] + SEED_TARGETS:
        hs = {lv: _mean_hs(runs, target=t, level=lv) for lv in (("E2", "E3", "E4") if t == "T0" else ("E1", "E2", "E3", "E4"))}
        hs = {lv: v for lv, v in hs.items() if v is not None}
        wf = _select(runs, target=t, level="W-freq")
        if not hs or not wf:
            continue
        true_lv = [lv for lv in ("E2", "E3", "E4") if lv in hs]  # §5 compares with condition means, not single runs
        trend_lv = [lv for lv in ("E1", "E2", "E3", "E4") if lv in hs]  # H1's ladder levels this target ran
        kl_true = [float(np.mean([r["kl_out"] for r in _select(runs, target=t, level=lv)])) for lv in true_lv]
        w_agree = float(np.mean([r["argmax_agreement"] for r in wf]))
        w_kl, w_hs = float(np.mean([r["kl_out"] for r in wf])), float(np.mean([r["hs"] for r in wf]))
        e3, wfv = np.array([r["hs"] for r in _select(runs, target=t, level="E3")]), np.array([r["hs"] for r in wf])
        mw = st.mann_whitney_greater(e3, wfv) if len(e3) else {"p": None}
        up = len(e3) > 0 and e3.mean() > wfv.mean()
        h2v = st.verdict_prereg(mw["p"], up)  # §5 as written first; deviation 1's reading after it (audit 8)
        post_hoc = st.verdict(mw["p"], up, st.min_p_mann_whitney(len(e3), len(wfv)))
        within = min(kl_true) <= w_kl <= max(kl_true)
        upper = w_kl <= max(kl_true)
        other = w_agree >= 0.99 and all(w_hs < hs[lv] for lv in true_lv)
        missing = [lv for lv in ("E2", "E3", "E4") if lv not in true_lv]
        coverage = f" ({', '.join(missing)} missing; full comparison unavailable)" if missing else ""
        h3 = (f"interval {'passes' if within and other else 'fails'}, "
              f"upper-bound reading {'passes' if upper and other else 'fails'}" + coverage)
        hs_test = _trend_numbers([np.array([r["hs"] for r in _select(runs, target=t, level=lv)]) for lv in trend_lv])
        kl_test = _trend_numbers([-np.array([r["kl_out"] for r in _select(runs, target=t, level=lv)]) for lv in trend_lv])
        out.append({
            "target_id": t, "hs_test": hs_test, "kl_test": kl_test,
            "h2_difference": hs["E3"]-w_hs if "E3" in hs else None, "h2_p": mw["p"], "h2_registered": h2v,
            "target": t + (" (flagged: memorisation)" if t == "T-paper" else
                           " (E2–E4 only; post hoc, for comparison)" if t == "T0" else ""),
            "hs_by_level": ", ".join(f"{lv} {_f(v)}" for lv, v in hs.items()),
            "h1": (("means rise" if _mono([hs[lv] for lv in trend_lv]) else "means do not rise") + f" {'→'.join(trend_lv)}; "
                   + _trend_text(hs_test)),
            "h2": (f"{hs['E3'] - w_hs:+.3f}; {h2v}" + (f" (p {_f(mw['p'])}" if mw["p"] is not None else "")
                   + (f"; post-hoc rule: {post_hoc})" if post_hoc != h2v else ")" if mw["p"] is not None else "")) if "E3" in hs else "–",
            "h1_kl": _trend_text(kl_test),
            "kl_within_observed_range": within, "kl_at_or_below_observed_max": upper,
            "h3_interval": within and other, "h3_upper": upper and other,
            "h3_observed": f"agreement {_f(w_agree)}, KL {_f(w_kl)} (true {_f(min(kl_true))}–{_f(max(kl_true))}), HS {_f(w_hs)}",
            "h3": h3,
        })
    return out
