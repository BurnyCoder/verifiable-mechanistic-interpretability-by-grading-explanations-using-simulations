"""Audit 4 (2026-11-22, post hoc): positive and agent-free controls for H8 (HS against T0's training checkpoints).

Local explanation: H8 correlates, over T0's 401 checkpoints, a replacement's HS(t) against checkpoint t with
−log restricted loss(t). Checkpoints converge to the final T0, so any network that resembles final T0 should get a
positive ρ whether or not it "tracks circuit formation" in any further sense. This script grades, against every
checkpoint, (i) final T0 itself (the weights dump), (ii) T0 with only its key-frequency Fourier components in W_E/W_U,
(iii) the agent-free reference-compiler E3 build, and (iv) an independent retrain (wd1_s0) as a negative reference, and
reports each ρ next to the stored E3/E4 replacement ρ. It also reports how closely each E3/E4 replacement's HS(t) curve
follows final T0's own curve (Spearman over checkpoints), which is the check that tells "tracks circuit formation" apart
from "resembles the end point".
Global context: post hoc; no preregistered verdict uses it. Output `results/audit4/h8_positive_control.json`, cited by
the audit report and the paper's §4.3. Grading uses the project's grader (`vmi.grader.score.grade_models`), whose
primary metrics this audit reproduced independently (`scripts/audit4/regrade.py`); restricted loss is read from
`results/controls/trajectory_T0.json` (same 401 checkpoints, grokviz.metrics.restricted_loss).
Sources: preregistration §5 H8; restricted loss — Nanda et al. 2023 §5.1 (https://arxiv.org/abs/2301.05217);
Spearman ρ — scipy.stats.spearmanr (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
Run: `.venv/Scripts/python.exe scripts/audit4/h8_positive_control.py` (GPU, ≈ 10 minutes).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # repo import path
import time  # timestamped log
from pathlib import Path  # paths

import numpy as np  # arrays
from scipy.stats import spearmanr  # rank correlation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grokviz.runstore import RunStore  # noqa: E402  checkpoint iterator of the target repository
from vmi import nonagent, targets, weights  # noqa: E402  target loading, Fourier restriction, weight files
from vmi.config import load  # noqa: E402  paths
from vmi.grader.score import grade_models  # noqa: E402  the (independently reproduced) grader

OUT = ROOT / "results" / "audit4" / "h8_positive_control.json"


PLATEAU_FACTOR = 1.2  # plateau = checkpoints after which restricted loss stays within 1.2 × its minimum (post hoc rule)


def phases(out: dict, traj: dict) -> dict:
    """Split the 401 checkpoints into circuit formation and the post-grokking plateau and recompute every ρ in each.

    Why: the whole-trajectory ρ turned out lower for final T0 itself than for the replacements, so its size cannot mean
    "tracks circuit formation better". Rule (post hoc, stated before splitting): the plateau starts at the first
    checkpoint after which the restricted loss never again exceeds PLATEAU_FACTOR × its minimum; formation = all
    checkpoints before (and including) that step."""
    steps, rl = np.array(traj["steps"]), np.array(traj["restricted_loss"])
    nl = -np.log(rl)
    above = np.nonzero(rl > PLATEAU_FACTOR * rl.min())[0]
    start = int(above.max()) + 1  # first index of the plateau
    form, plat = np.arange(len(steps)) <= start, np.arange(len(steps)) > start
    curves = {k: np.array(v) for k, v in out["hs"].items()}
    curves.update({r["run_id"]: np.array(r["hs"]) for r in traj["replacements"]})
    rho = lambda v, m: float(spearmanr(v[m], nl[m]).statistic)  # noqa: E731
    return {"plateau_rule": f"restricted loss ≤ {PLATEAU_FACTOR} × its minimum from this step on",
            "plateau_start_step": int(steps[start]), "n_formation": int(form.sum()), "n_plateau": int(plat.sum()),
            "restricted_loss_min": float(rl.min()), "restricted_loss_last": float(rl[-1]),
            "plateau_rl_vs_step_spearman": float(spearmanr(steps[plat], rl[plat]).statistic),
            "rho_formation": {k: rho(v, form) for k, v in curves.items()},
            "rho_plateau": {k: rho(v, plat) for k, v in curves.items()}}


def main() -> None:
    st = load()
    traj = json.loads((ROOT / "results" / "controls" / "trajectory_T0.json").read_text(encoding="utf-8"))
    if OUT.exists() and "--regrade" not in sys.argv:  # grading takes ~15 min; the phase analysis reuses saved curves
        out = json.loads(OUT.read_text(encoding="utf-8"))
        out["phases"] = phases(out, traj)
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in out["phases"].items()}, indent=1))
        return
    nl = -np.log(np.array(traj["restricted_loss"]))
    final = targets.load_model(st, "T0")
    refs = {
        "final T0 (weights dump)": weights.from_model(final),
        "Fourier-restricted T0": nonagent.fourier_restricted(targets.load_model(st, "T0")),
        "reference compiler E3 (no agent)": weights.load_npz(st.runs_dir / "_compiler" / "T0" / "E3.npz"),
        "independent retrain wd1_s0": weights.from_model(targets.load_model(st, "wd1_s0")),
    }
    models = {k: weights.to_model(v) for k, v in refs.items()}
    store = RunStore(targets.run_dir(st, targets.resolve("T0").run_id))
    steps, hs = [], {k: [] for k in refs}
    for _, step, ckpt in store.iter_models("cpu", every=1):
        steps.append(step)
        for k, m in models.items():
            hs[k].append(grade_models(ckpt, m, secondary=False)["hs"])
        if step % 4000 == 0:
            print(time.strftime("%H:%M:%S"), "step", step, {k: round(v[-1], 3) for k, v in hs.items()}, flush=True)
    assert steps == traj["steps"], "checkpoint steps differ from the stored trajectory"
    rho = {k: float(spearmanr(v, nl).statistic) for k, v in hs.items()}
    follow = {r["run_id"]: float(spearmanr(r["hs"], hs["final T0 (weights dump)"]).statistic) for r in traj["replacements"]}
    out = {"n_checkpoints": len(steps), "rho_vs_neglog_restricted_loss": rho,
           "stored_e3e4_rho": {r["run_id"]: r["rho"] for r in traj["replacements"]},
           "spearman_replacement_curve_vs_final_T0_curve": follow, "hs": hs, "steps": steps,
           "note": "post hoc (audit 4); H8's verdict is unchanged"}
    out["phases"] = phases(out, traj)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"rho": rho, "follow_final_T0": follow}, indent=1))


if __name__ == "__main__":
    main()
