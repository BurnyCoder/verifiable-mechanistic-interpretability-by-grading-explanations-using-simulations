"""Logit-level output comparison with the target T0 (post hoc, audit 9, preregistration deviation 16).

Local explanation:
- `centred_logits(L)`: each input's logits minus their mean over the 113 number classes, the part a softmax sees.
- `calibrated_logit_fvu(Lm, Lr)`: fraction of the target's centred-logit variance left unexplained after rescaling the
  replacement's centred logits by the one positive scale s that minimises it,
  min_{s>0} Σ‖L̃m − s·L̃r‖² / Σ‖L̃m‖² = 1 − ⟨L̃m, L̃r⟩² / (‖L̃m‖²‖L̃r‖²) (1 if ⟨L̃m, L̃r⟩ ≤ 0), i.e. one minus the squared
  cosine of the two centred-logit tables. The raw value without the scale is the preregistered secondary metric
  `centred_logit_fvu` (preregistration §4, `vmi.grader.metrics`); the one-scale version mirrors the preregistered
  temperature-calibrated KL, which also fits a single scale.
- `analyse(settings)`: both values for every reported T0 run of the main and controls stages, the reference
  compiler's T0 builds, the independent retrains, both kinds of distilled students and the random initialisations,
  written to results/controls/logit_fvu_T0.json.
Global context: a grokked model's softmax is nearly one-hot, so output KL sees little beyond the answer and the logit
scale, while its centred logits carry the key frequencies (logit(c) ≈ Σ_k α_k cos(w_k(a + b − c)), Nanda et al. 2023
§4.2). This file measures whether an output-only comparison at the logit level separates the explanations the way
HS does. Post hoc; no preregistered verdict uses it.
Sources: Nanda et al. 2023 §4.2 logit form (https://arxiv.org/abs/2301.05217); matching zero-mean logits as the
high-temperature limit of distillation — Hinton et al. 2015 §2.1 (https://arxiv.org/abs/1503.02531); least-squares
scale in closed form (projection of one vector on another).
"""

from __future__ import annotations  # postponed annotations

import json  # results file
from pathlib import Path  # paths

import numpy as np  # weight files
import torch  # logits

from vmi import targets, weights  # target checkpoints, weights.npz → model
from vmi.config import REPO_ROOT, Settings  # paths
from vmi.grader.activations import collect  # logits on all 113² inputs
from vmi.grader.metrics import centred_logit_fvu  # preregistered raw value

RES = REPO_ROOT / "results"
TARGET = "T0"


def centred_logits(L: torch.Tensor) -> torch.Tensor:
    """Logits minus each input's mean over the classes (float64)."""
    L = L.double()
    return L - L.mean(-1, keepdim=True)


def calibrated_logit_fvu(Lm: torch.Tensor, Lr: torch.Tensor) -> tuple[float, float]:
    """(FVU after the best positive scale, that scale) of the replacement's centred logits against the target's."""
    a, b = centred_logits(Lm), centred_logits(Lr)
    ab, aa, bb = float((a * b).sum()), float((a * a).sum()), float((b * b).sum())
    if aa == 0 or bb == 0 or ab <= 0:
        return 1.0, 0.0  # no positive scale explains anything
    return 1.0 - ab * ab / (aa * bb), ab / bb


def _row(kind: str, name: str, Lm: torch.Tensor, model, **extra) -> dict:
    """Both values for one network graded against the target's logits."""
    Lr = collect(model.double(), Lm.device).logits
    cal, scale = calibrated_logit_fvu(Lm, Lr)
    return {"kind": kind, "name": name, **extra, "centred_logit_fvu": centred_logit_fvu(Lm, Lr),
            "calibrated_logit_fvu": cal, "scale": scale}


def analyse(settings: Settings, max_round: int = 2) -> Path:
    """Grade every T0 network the paper compares at the logit level; write results/controls/logit_fvu_T0.json."""
    from grokviz.model import GrokTransformer
    from vmi.analysis.hypotheses import MAIN_MODEL, load_runs
    from vmi.nonagent import retrain_ids
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Lm = collect(targets.load_model(settings, TARGET).double(), dev).logits
    rows = []
    for r in load_runs(max_round):  # reported agent runs of the main simulator, main and controls stages, no variants
        if r["target"] == TARGET and r["model"] == MAIN_MODEL and not r.get("variant") and r["stage"] in ("main", "controls"):
            npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
            rows.append(_row("agent", r["run_id"], Lm, weights.to_model(weights.load_npz(npz)), level=r["level"], hs=r["hs"]))
    for npz in sorted((settings.runs_dir / "_compiler" / TARGET).glob("*.npz")):
        rows.append(_row("compiler", npz.stem, Lm, weights.to_model(weights.load_npz(npz)), level=npz.stem))
    for run in retrain_ids(TARGET):
        rows.append(_row("retrain", run, Lm, targets.load_model(settings, run)))
    for kind, folder in (("student_kl", "student"), ("student_logit", "student_logit")):
        for d in sorted((settings.data_dir / "trained").glob(f"{folder}_{TARGET}_s*")):
            if not (d / "step_020000.npz").exists():  # still training: skip rather than fail
                continue
            arr = dict(np.load(d / "step_020000.npz"))
            rows.append(_row(kind, d.name, Lm, weights.to_model(arr)))
    for s in range(5):  # the same five random initialisations as nonagent.random_init
        rows.append(_row("random_init", f"seed {1000 + s}", Lm, GrokTransformer(seed=1000 + s)))
    out = {"definition": __doc__.split("Global context:")[0].strip(), "target": TARGET, "rows": rows}
    path = RES / "controls" / f"logit_fvu_{TARGET}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return path


# ------------------------------------------------------------------------------------------------
# Readers for the report generators (tables.py): every number below is read from results/
# ------------------------------------------------------------------------------------------------
TRUE_LEVELS = ("E2", "E3", "E4")
ORDER = ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble")


def load() -> dict | None:
    """results/controls/logit_fvu_T0.json, or None before `vmi logitfit` has run."""
    f = RES / "controls" / f"logit_fvu_{TARGET}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def groups(d: dict) -> dict[str, list[dict]]:
    """Rows grouped as 'agent E3', 'compiler E3', 'retrain', 'student_kl', 'student_logit', 'random_init'."""
    out: dict[str, list[dict]] = {}
    for r in d["rows"]:
        key = f"{r['kind']} {r['level']}" if r["kind"] in ("agent", "compiler") else r["kind"]
        out.setdefault(key, []).append(r)
    return out


def _rng(vals, nd=2) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}–{hi:.{nd}f}"


def student_reading(rows: list[dict]) -> str:
    """The reading pre-specified in deviation 16(a), applied per student: only students that meet the training criterion
    (centred-logit FVU < 0.05) are interpreted."""
    conv = [r for r in rows if r["converged"]]
    fvu = _rng([r["centred_logit_fvu"] for r in rows], 3)
    if not conv:
        return (f"none of the {({2: 'two', 3: 'three'}).get(len(rows), len(rows))} reached the pre-specified training criterion (centred-logit FVU < 0.05; reached "
                f"{fvu}), so by the pre-specified rule their HS (mean {np.mean([r['hs'] for r in rows]):.2f}) is not "
                f"interpreted, and whether matched logits carry T0's hidden states stays open")
    hs = float(np.mean([r["hs"] for r in conv]))
    note = "" if len(conv) == len(rows) else f" ({len(conv)} of {len(rows)} met the training criterion; the others are not interpreted)"
    if hs >= 0.85:
        return f"mean HS {hs:.2f} is in the E2–E4 range, so by the pre-specified reading HS adds little beyond the logits on this task{note}"
    if hs <= 0.35:
        return (f"mean HS {hs:.2f} stays in the range of retrains, probability-matched students and random networks, so by "
                f"the pre-specified reading T0's hidden states carry mechanism information that matched logits do not{note}")
    return f"mean HS {hs:.2f} lies between those ranges, which the pre-specified reading calls partial{note}"


def table(baselines: dict | None = None) -> str:
    """Markdown: HS next to the logit-level comparison, per explanation (agent and compiler) and per reference network."""
    d = load()
    if d is None:
        return ""
    g = groups(d)
    ms = lambda v, nd=3: f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}" if len(v) > 1 else f"{v[0]:.{nd}f}"  # noqa: E731
    rows = ["| Network | n | HS ↑ | centred-logit FVU ↓ (preregistered secondary) | centred-logit FVU, one fitted scale ↓ "
            "(post hoc; range) | compiler build, one fitted scale |", "|---|---|---|---|---|---|"]
    for lv in ORDER:
        rs, comp = g.get(f"agent {lv}", []), g.get(f"compiler {lv}", [])
        if rs:
            rows.append(f"| {lv} (agent) | {len(rs)} | {ms([r['hs'] for r in rs])} | {ms([r['centred_logit_fvu'] for r in rs], 2)} | "
                        f"{_rng([r['calibrated_logit_fvu'] for r in rs])} | "
                        + (f"{comp[0]['calibrated_logit_fvu']:.2f} |" if comp else "– |"))
    b = baselines or {}
    hs_of = {"retrain": [x["hs"] for x in b.get("retrains", [])], "student_kl": [x["hs"] for x in b.get("students", [])],
             "student_logit": [x["hs"] for x in b.get("students_logit", [])]}
    ri = RES / "controls" / f"random_init_{TARGET}.json"
    hs_of["random_init"] = [x["hs"] for x in json.loads(ri.read_text())] if ri.exists() else []
    names = {"retrain": "Independent retrains", "student_kl": "Students matched to M's output probabilities (KL, temperature 1)",
             "student_logit": "Students matched to M's centred logits (post hoc)", "random_init": "Random initialisation"}
    for k, name in names.items():
        rs = g.get(k, [])
        if rs:
            rows.append(f"| {name} | {len(rs)} | {ms(hs_of[k]) if hs_of[k] else '–'} | {ms([r['centred_logit_fvu'] for r in rs], 2)} | "
                        f"{_rng([r['calibrated_logit_fvu'] for r in rs])} | – |")
    return "\n".join(rows)


def inline_values() -> dict[str, str]:
    """`v:a9-*` values for prose sentences (post hoc, audit 9)."""
    v: dict[str, str] = {}
    d = load()
    if d is not None:
        g = groups(d)
        cal = lambda k: [r["calibrated_logit_fvu"] for r in g.get(k, [])]  # noqa: E731
        raw = lambda k: [r["centred_logit_fvu"] for r in g.get(k, [])]  # noqa: E731
        for lv in ORDER:
            if cal(f"agent {lv}"):
                v[f"a9-cal-{lv.lower()}"] = _rng(cal(f"agent {lv}"))
                v[f"a9-raw-{lv.lower()}"] = _rng(raw(f"agent {lv}"))
        wrong = cal("agent W-freq") + cal("agent W-role") + cal("agent E0") + cal("agent E1")
        refs = cal("retrain") + cal("student_kl") + cal("random_init")
        if wrong and refs:
            v["a9-cal-wrong"] = _rng(wrong)
            v["a9-cal-refs"] = _rng(refs)
            v["a9-raw-ok"] = f"≤ {max(raw('agent E3') + raw('agent E4') + raw('agent W-alg') + raw('agent W-scramble')):.2f}"
            v["a9-raw-bad"] = f"≥ {min(raw('agent W-freq') + raw('agent W-role') + raw('agent E0') + raw('agent E1')):.2f}"
            v["a9-stu-raw"] = _rng(raw("student_kl"))
            v["a9-ret-raw"] = _rng(raw("retrain"))
    b = RES / "baselines" / f"{TARGET}.json"
    if b.exists():
        bd = json.loads(b.read_text(encoding="utf-8"))
        st = bd.get("students_logit")
        if st:
            v["a9-ls-hs"] = _rng([r["hs"] for r in st])
            v["a9-ls-fvu"] = _rng([r["centred_logit_fvu"] for r in st], 3)
            v["a9-ls-kl"] = f"{np.mean([r['kl_out'] for r in st]):.2e}"
            v["a9-ls-reading"] = student_reading(st)
    fr = RES / "frequencies_round2.json"  # E0 above E1 (H1's untested clause): what the runs built (post hoc)
    if fr.exists():
        rows = [r for r in json.loads(fr.read_text(encoding="utf-8"))["rows"] if r["target"] == TARGET and not r.get("variant")]
        e0 = sorted((r for r in rows if r["level"] == "E0"), key=lambda r: r["hs"])
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 8: "eight"}
        consecutive = lambda f: len(f) >= 6 and all(b - a == 1 for a, b in zip(f, f[1:]))  # noqa: E731
        broad = [r for r in e0 if consecutive(r["used_freqs"])]
        if broad and len(broad) < len(e0) and len({len(r["used_freqs"]) for r in broad}) == 1:
            n_b, n_f = len(broad), len(broad[0]["used_freqs"])
            v["a9-e0-broad"] = (f"{words.get(n_b, n_b)} of its {words.get(len(e0), len(e0))} runs built {words.get(n_f, n_f)} consecutive "
                                f"frequencies (HS {' and '.join(f'{r['hs']:.3f}' for r in broad)}; share of W_L on T0's "
                                f"frequencies {' and '.join(f'{r['target_norm_share']:.2f}' for r in broad)}), while the "
                                f"other, built on {words.get(len(e0[0]['used_freqs']), len(e0[0]['used_freqs']))} "
                                + ("of the paper's frequencies" if set(e0[0]["used_freqs"]) <= set(json.loads(fr.read_text(encoding="utf-8"))["paper_freqs"]) else "frequencies")
                                + f", scores {e0[0]['hs']:.3f}")
            e1 = [r["hs"] for r in rows if r["level"] == "E1"]
            v["a9-e1-hs"] = _rng(e1, 3)
    vf = RES / "audit9" / "verify.json"  # scripts/audit9/verify.py: independent re-derivation of the stored statistics
    if vf.exists():
        ck = json.loads(vf.read_text(encoding="utf-8"))["checks"]
        v["a9-verify"] = f"{sum(c['agree'] for c in ck)} of {len(ck)}"
    ri = RES / "controls" / f"random_init_{TARGET}.json"
    if ri.exists():
        v["a9-random-hs"] = f"{np.mean([x['hs'] for x in json.loads(ri.read_text())]):.2f}"
    return v
