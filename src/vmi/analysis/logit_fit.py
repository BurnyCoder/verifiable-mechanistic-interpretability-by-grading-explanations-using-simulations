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
            arr = dict(np.load(d / "step_020000.npz"))
            rows.append(_row(kind, d.name, Lm, weights.to_model(arr)))
    for s in range(5):  # the same five random initialisations as nonagent.random_init
        rows.append(_row("random_init", f"seed {1000 + s}", Lm, GrokTransformer(seed=1000 + s)))
    out = {"definition": __doc__.split("Global context:")[0].strip(), "target": TARGET, "rows": rows}
    path = RES / "controls" / f"logit_fvu_{TARGET}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return path
