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

import numpy as np  # means

from vmi.analysis.hypotheses import LOCAL_MODELS, MAIN_MODEL, _select, compiler, load_runs  # same selection as the tests
from vmi.config import REPO_ROOT  # results folder

RES = REPO_ROOT / "results"
SEED_TARGETS = ["T1", "T2", "T-new", "T-paper"]  # T-paper is contamination-flagged (memorisation, see frequencies)
PREDICTED = {  # wording follows preregistration §5; thresholds are the preregistered constants
    "H1": "HS rises and output KL falls from E1 to E4 on T0; E0 lies below E1",
    "H2": "true explanations beat length-matched false ones on HS: E3 > W-freq, W-alg, W-role; E4 > W-scramble",
    "H3": "W-freq: argmax agreement ≥ 0.99, KL not above the true E2–E4 range, HS below each of E2–E4",
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
    return h.get("verdict") or "; ".join(f"{a}: {b}" for a, b in h.get("verdicts", {}).items())


def _mean_hs(runs, **kw):
    rs = _select(runs, **kw)
    return float(np.mean([r["hs"] for r in rs])) if rs else None


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
        eff = {k: v["diff"] for k, v in H["H2"]["effects"].items() if v}
        add("H2", "HS difference " + "; ".join(f"{k} {d:+.3f}" for k, d in eff.items()),
            _match({f"{k} reversed": d > 0 for k, d in eff.items()}))
    if "H3" in H:
        h = H["H3"]
        w, tk, th = h["wfreq"], list(h["true_kl_means"].values()), list(h["true_hs_means"].values())
        add("H3", f"W-freq: agreement {_f(w['argmax_agreement'])}, KL {_f(w['kl'])} (true {_f(min(tk))}–{_f(max(tk))}), "
                  f"HS {_f(w['hs'])} (true {_f(min(th))}–{_f(max(th))})",
            _match({"agreement < 0.99": w["argmax_agreement"] >= 0.99, "KL above the true range": h["kl_fooled"],
                    "HS not below every true explanation": h["hs_catches"]}))
    b = RES / "baselines" / "T0.json"
    if "H4" in H and b.exists():
        d = json.loads(b.read_text())
        rt, st = [x["hs"] for x in d["retrains"]], [x["hs"] for x in d["students"]]
        ag = H["H4"]["agent_hs_means"]
        add("H4", ", ".join(f"{lv} {_f(v)}" for lv, v in ag.items()) + f"; {len(rt)} retrains mean {_f(np.mean(rt))} "
                  f"(max {_f(max(rt))}); {len(st)} students mean {_f(np.mean(st))} (max {_f(max(st))})",
            _match({"not above every retrain": min(ag.values()) > max(rt)}))
    if "H4b" in H:
        rows = H["H4b"]["rows"]
        hs, kl = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
        add("H4b", "HS " + " → ".join(_f(x) for x in hs) + "; KL " + " → ".join(_f(x) for x in kl),
            _match({"HS not strictly falling": abs(hs[0] - 1) < 1e-6 and _mono(hs, increasing=False),
                    "KL not strictly rising": kl[0] < 1e-9 and _mono(kl)}))
    gaps = []  # agent minus compiler HS, main simulator, every target and explanation H5 covers
    for t in ["T0"] + SEED_TARGETS[:-1]:
        for lv in ("E2", "E3", "E4", "W-freq"):
            a, c = _mean_hs(runs, target=t, level=lv), compiler(t, lv)
            if a is not None and c:
                gaps.append((t, lv, a - c["hs"]))
    if gaps:
        top = max(gaps, key=lambda g: g[2])
        obs = "agent − compiler on T0: " + ", ".join(f"{lv} {g:+.3f}" for t, lv, g in gaps if t == "T0")
        obs += f"; agent above compiler in {sum(g > 0 for *_, g in gaps)} of {len(gaps)} target–explanation pairs (largest {top[2]:+.3f}, {top[0]} {top[1]})"
        local = [r for r in runs if r["model"] in LOCAL_MODELS]
        add("H5", obs + f"; {len(local)} local runs, {sum(r['accuracy'] > 2 / 113 for r in local)} above chance",
            _match({"agent above compiler somewhere": all(g <= 0 for *_, g in gaps), "capability trend not testable": False}))
        add("H6", f"{len(local)} local runs, {sum(r['accuracy'] > 2 / 113 for r in local)} above chance: nothing to rank",
            "not testable")
    if "H7" in H:
        rows = H["H7"]["rows"]
        obs = "; ".join(f"{r['level']}: HS {_f(r['hs_T0'])} vs best other {_f(max(r['hs_wrong'].values()))} "
                        f"({max(r['hs_wrong'], key=r['hs_wrong'].get)})" for r in rows)
        add("H7", obs, _match({"a wrong target scores as high": all(r["d"] > 0 for r in rows)}))
    if "H8" in H:
        rho = H["H8"]["spearman_hs_vs_neglog_restricted_loss"]
        add("H8", f"Spearman ρ(HS, −log restricted loss) = {_f(rho)}", _match({"ρ ≤ 0": rho > 0}))
    return out


def replication_rows(max_round: int) -> list[dict]:
    """H1–H3 re-checked per seed-sensitivity target (main simulator, same rules as on T0)."""
    runs = load_runs(max_round)
    out = []
    for t in SEED_TARGETS:
        hs = {lv: _mean_hs(runs, target=t, level=lv) for lv in ("E1", "E2", "E3", "E4")}
        hs = {lv: v for lv, v in hs.items() if v is not None}
        wf = _select(runs, target=t, level="W-freq")
        if not hs or not wf:
            continue
        true_lv = [lv for lv in ("E2", "E3", "E4") if lv in hs]  # §5 compares with condition means, not single runs
        kl_true = [float(np.mean([r["kl_out"] for r in _select(runs, target=t, level=lv)])) for lv in true_lv]
        w_agree = float(np.mean([r["argmax_agreement"] for r in wf]))
        w_kl, w_hs = float(np.mean([r["kl_out"] for r in wf])), float(np.mean([r["hs"] for r in wf]))
        out.append({
            "target": t + (" (flagged: memorisation)" if t == "T-paper" else ""),
            "hs_by_level": ", ".join(f"{lv} {_f(v)}" for lv, v in hs.items()),
            "h1": "yes" if _mono([hs[lv] for lv in ("E2", "E3", "E4") if lv in hs]) else "no",
            "h2": f"{hs['E3'] - w_hs:+.3f}" if "E3" in hs else "–",
            "h3_observed": f"agreement {_f(w_agree)}, KL {_f(w_kl)} (true max {_f(max(kl_true))}), HS {_f(w_hs)}",
            "h3": _match({"agreement < 0.99": w_agree >= 0.99, "KL above the true range": w_kl <= max(kl_true),
                          "HS not below every true explanation": all(w_hs < hs[lv] for lv in true_lv)}),
        })
    return out
