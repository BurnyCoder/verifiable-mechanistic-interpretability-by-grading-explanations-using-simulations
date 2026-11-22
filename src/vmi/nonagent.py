"""No-agent controls and baselines (preregistration §7): everything graded without the simulator.

Local explanation:
- `random_init`        5 randomly initialised models graded against T0 (the floor).
- `wrong_targets`      every graded T0 agent replacement (E3/E4) also graded against T1, T2, dev, T-paper and the
                       no-mechanism models wd0_s*, rand_s*; H7 statistic d = HS(T0) − max HS(wrong) per run.
- `trajectory`         every T0 E3/E4 replacement graded against all 401 of T0's checkpoints, next to restricted loss (H8).
- `planted`            reference-compiled E3 model for T0 trained further; the step-0 model graded against each snapshot (H4b).
- `baselines`          independent retrains (9 other wd1 seeds), logit-distilled students (trained), Fourier-restricted M,
                       weights dump; written to results/baselines/<target>.json.
Global context: these runs need no LLM; they are deterministic given the fixed seeds and finish in minutes.
Sources: preregistration §5, §7; restricted loss — grokviz.metrics.restricted_loss (paper Sec. 5.1).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
from pathlib import Path  # paths

import numpy as np  # arrays
import torch  # models

from vmi import targets, weights  # target loading, weight files
from vmi.analysis import stats as st  # Wilcoxon, Spearman
from vmi.config import REPO_ROOT, Settings  # paths
from vmi.grader.score import grade_models, grade_npz  # grading

RES = REPO_ROOT / "results"
WRONG = ["T1", "T2", "dev", "T-paper", "wd0_s0", "wd0_s1", "wd0_s2", "wd0_s3", "wd0_s4", "rand_s0", "rand_s1", "rand_s2"]


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8", newline="\n")


def random_init(settings: Settings, target: str = "T0", seeds=range(5)) -> list[dict]:
    """Grade randomly initialised paper-layout models (seeds 0–4) against the target."""
    from grokviz.model import GrokTransformer
    m = targets.load_model(settings, target)
    rows = [{"seed": s, **grade_models(m, GrokTransformer(seed=1000 + s), secondary=False)} for s in seeds]
    _dump(RES / "controls" / f"random_init_{target}.json", rows)
    return rows


def _agent_runs(levels=("E3", "E4"), target="T0", max_round=99, stages=("main",)) -> list[dict]:
    out = []
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["target"] == target and r["level"] in levels and r.get("variant", "") == "" and r["stage"] in stages \
                and int(r.get("round", 0)) <= max_round and not r.get("failed"):
            out.append(r)
    return out


def wrong_targets(settings: Settings, max_round: int) -> dict:
    """H7: grade every T0 E3/E4 agent replacement against wrong targets; Wilcoxon on d = HS(T0) − max HS(wrong)."""
    runs = _agent_runs(max_round=max_round)
    models = {t: targets.load_model(settings, t) for t in WRONG}
    rows = []
    for r in runs:
        npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        hs_wrong = {t: grade_npz(m, npz, secondary=False)["hs"] for t, m in models.items()}
        rows.append({"run_id": r["run_id"], "level": r["level"], "hs_T0": r["hs"], "hs_wrong": hs_wrong,
                     "d": r["hs"] - max(hs_wrong.values())})
    d = np.array([x["d"] for x in rows])
    w = st.wilcoxon_greater(d) if len(d) else {"p": None, "median": None}
    n = len(rows)
    verdict = st.verdict(w["p"], bool(w["median"] and w["median"] > 0), w["min_p"])
    out = {"N_runs": n, "verdict": verdict, "wilcoxon": w, "rows": rows}
    _dump(RES / "controls" / "wrong_targets_T0.json", out)
    return out


def trajectory(settings: Settings, max_round: int, every: int = 1) -> dict:
    """H8 (preregistration §5): grade every T0 E3/E4 replacement against each of T0's 401 checkpoints; per replacement,
    Spearman ρ of HS(t) with −log restricted loss(t); supported only if every ρ > 0.

    Deviation 8 (audit 2026-11-22): rounds 1–2 had used only the highest-HS replacement and every 10th checkpoint (41),
    labelled "all checkpoints"; this follows §5 as written. Descriptive: checkpoints of one run are not independent.

    Negative controls (post hoc, audit 2026-11-22 second pass): ρ between two quantities that both trend over training
    is positive for many pairs, so the same ρ is computed for replacements built without T0's mechanism — W-freq (wrong
    frequencies), E1 (algorithm only) and E0 (no explanation). If HS tracks circuit formation specifically, their ρ
    should not be positive like the E3/E4 replacements'. They do not enter the preregistered verdict."""
    from grokviz import metrics as gm
    from grokviz.runstore import RunStore
    runs = _agent_runs(max_round=max_round)
    if not runs:
        return {}
    controls = _agent_runs(levels=("W-freq", "E1", "E0"), max_round=max_round, stages=("main", "controls"))
    store = RunStore(targets.run_dir(settings, targets.resolve("T0").run_id))
    grid = targets.grid("T0", "cuda" if torch.cuda.is_available() else "cpu")
    key = targets.published_key_freqs(settings, "T0")
    npz = {r["run_id"]: settings.runs_dir / r["run_id"] / "work_out" / "weights.npz" for r in runs + controls}
    steps, rl, hs = [], [], {rid: [] for rid in npz}
    for t, step, model in store.iter_models("cpu", every=every):
        acts = gm.evaluate(model.to(grid.F.device), grid)
        steps.append(step)
        rl.append(gm.restricted_loss(acts.logits, grid, key))
        for rid, p in npz.items():
            hs[rid].append(grade_npz(model, p, secondary=False)["hs"])
    neg_log_rl = -np.log(np.array(rl))
    rho = lambda v: float(st.stats.spearmanr(v, neg_log_rl).statistic)  # noqa: E731
    reps = [{"run_id": r["run_id"], "rho": rho(hs[r["run_id"]]), "hs": hs[r["run_id"]]} for r in runs]
    ctrl = [{"run_id": r["run_id"], "level": r["level"], "rho": rho(hs[r["run_id"]]), "hs": hs[r["run_id"]]} for r in controls]
    by_level = {}
    for c in ctrl:
        by_level.setdefault(c["level"], []).append(c["rho"])
    rhos = [r["rho"] for r in reps]
    rows = [{"step": s, "hs": float(np.mean([hs[r["run_id"]][i] for r in runs])), "restricted_loss": rl[i]} for i, s in enumerate(steps)]
    out = {"N": f"{len(reps)} E3/E4 replacements × {len(steps)} checkpoints", "steps": steps, "restricted_loss": rl,
           "replacements": reps, "spearman_min": min(rhos), "spearman_hs_vs_neglog_restricted_loss": float(np.median(rhos)),
           "verdict": "supported" if min(rhos) > 0 else "not supported", "rows": rows,
           "negative_controls": ctrl, "negative_controls_by_level": by_level}
    _dump(RES / "controls" / "trajectory_T0.json", out)
    return out


def planted(settings: Settings, target: str = "T0", steps=(0, 100, 1000, 10000)) -> dict:
    """H4b: train the reference-compiled E3 model further on the task; grade the step-0 model against each snapshot."""
    from vmi.pipeline import load_explanation
    from vmi.reference.compile import compile_spec
    from vmi.training import train_from
    spec, _, _ = load_explanation(target, "E3")
    init = weights.validate(compile_spec(spec))
    snaps = train_from(init, mode="labels", steps=max(steps), snapshots=list(steps),
                       out_dir=settings.data_dir / "trained" / f"planted_{target}", split_seed=targets.resolve(target).split_seed)
    rows = []
    for s in steps:
        g = grade_models(weights.to_model(snaps[s]), weights.to_model(init), secondary=False)
        rows.append({"step": s, "hs": g["hs"], "kl_out": g["kl_out"], "snapshot_accuracy": g["target_accuracy"]})
    out = {"target": target, "rows": rows}
    _dump(RES / "controls" / f"planted_calibration_{target}.json", out)
    return out


def fourier_restricted(model) -> dict[str, np.ndarray]:
    """M with its embedding and unembedding projected onto the key-frequency (and constant) Fourier components."""
    from grokviz import metrics as gm
    from grokviz.fourier import fourier_basis
    F, _ = fourier_basis(113)
    key = gm.key_frequencies(model, F)
    keep = [0] + [i for k in key for i in (2 * k - 1, 2 * k)]
    Fk = F[keep].double().numpy()
    arr = weights.from_model(model)
    arr["W_E"][:, :113] = arr["W_E"][:, :113] @ Fk.T @ Fk  # keep only the key-frequency content along the token axis
    arr["W_U"][:, :113] = arr["W_U"][:, :113] @ Fk.T @ Fk
    return arr


def baselines(settings: Settings, target: str = "T0") -> dict:
    """Independent retrains, distilled students, Fourier-restricted M and the weights dump, graded against the target."""
    from grokviz.model import GrokTransformer
    from vmi.grader.activations import collect
    from vmi.training import train_from
    m = targets.load_model(settings, target)
    own = targets.resolve(target).run_id
    others = [f"wd1_s{s}" for s in range(11) if f"wd1_s{s}" != own and f"wd1_s{s}" != "wd1_s10"][:9]
    if target == "T-paper":
        others = ["wd1_s0", "split0_init100", "split0_init101", "split0_init102"] + others
    retrains = [{"run": r, **grade_models(m, targets.load_model(settings, r), secondary=False)} for r in others]
    teacher = collect(m.double(), "cuda" if torch.cuda.is_available() else "cpu").logits
    students = []
    for s in (1000, 1001, 1002):
        init = weights.from_model(GrokTransformer(seed=s))
        snap = train_from(init, mode="distill", steps=20000, snapshots=[20000], teacher_logits=teacher,
                          out_dir=settings.data_dir / "trained" / f"student_{target}_s{s}")
        students.append({"seed": s, **grade_models(m, weights.to_model(snap[20000]), secondary=False)})
    fr = grade_models(m, weights.to_model(fourier_restricted(targets.load_model(settings, target))), secondary=False)
    dump = grade_models(m, weights.to_model(weights.from_model(targets.load_model(settings, target))), secondary=False)
    out = {"target": target, "retrains": retrains, "students": students, "fourier_restricted": fr, "weights_dump": dump,
           "weights_dump_length": {"numbers": int(sum(np.prod(s) for _, s in weights.SCHEMA.values()))}}
    _dump(RES / "baselines" / f"{target}.json", out)
    return out
