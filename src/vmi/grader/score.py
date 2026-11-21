"""Grade one replacement against one target: every preregistered metric in one JSON-ready dict.

Local explanation: `grade_models` runs both models on all 113² inputs once and computes the two primary metrics
(KL_out, HS) and all secondary metrics; `grade_npz` adds the failure rule: an unreadable or invalid weight file is
graded as the all-zero model and marked `failed` (preregistration §4).
Global context: called by the simulator queue after each agent run, by the baselines/controls stages for no-agent
models, and by the analysis to re-grade with alternative grader settings (hyperparameter sensitivity).
Sources: metric definitions in vmi.grader.metrics; failure rule in reports/preregistration.md §4.
"""

from __future__ import annotations  # postponed annotations

from pathlib import Path  # weight file path

import torch  # device handling
from grokviz.fourier import fourier_basis  # orthonormal Fourier basis for the embedding spectrum
from grokviz.model import GrokTransformer  # target / replacement network
from grokviz.task import make_labels  # (a+b) mod p labels

from vmi import weights  # schema, validation, zero model
from vmi.config import P  # modulus
from vmi.grader import metrics as mt  # metric functions
from vmi.grader.activations import SITES, Acts, collect  # activation extraction


def device() -> torch.device:
    """CUDA when available (all grading fits easily in 8 GB), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def grade_acts(acts_m: Acts, acts_r: Acts, target: GrokTransformer, replacement: GrokTransformer, *,
               ridge_rel: float = mt.RIDGE_REL, k: int = mt.N_FOLDS, fold_seed: int = mt.FOLD_SEED,
               secondary: bool = True) -> dict:
    """All metrics from precomputed activations (lets the sensitivity analysis vary grader settings cheaply)."""
    labels = make_labels(P).to(acts_m.logits.device)
    res: dict = {
        "kl_out": mt.kl_out(acts_m.logits, acts_r.logits),
        "argmax_agreement": mt.argmax_agreement(acts_m.logits, acts_r.logits),
        "accuracy": mt.accuracy(acts_r.logits, labels),
        "target_accuracy": mt.accuracy(acts_m.logits, labels),
        "sites": {},
    }
    for s in SITES:
        res["sites"][s] = mt.site_score(acts_m.site(s), acts_r.site(s), ridge_rel=ridge_rel, k=k, seed=fold_seed)
    res["hs"] = sum(v["harmonic"] for v in res["sites"].values()) / len(SITES)  # primary 2
    if secondary:
        cal_kl, cal_scale = mt.calibrated_kl(acts_m.logits, acts_r.logits)
        F, _ = fourier_basis(P)
        res.update({
            "centred_logit_fvu": mt.centred_logit_fvu(acts_m.logits, acts_r.logits),
            "calibrated_kl": cal_kl,
            "calibration_scale": cal_scale,
            "cka": {s: mt.linear_cka(acts_m.site(s), acts_r.site(s)) for s in SITES},
            "procrustes": {s: mt.procrustes_distance(acts_m.site(s), acts_r.site(s)) for s in SITES},
            "matched_neuron_corr": mt.matched_neuron_corr(acts_m.mlp_post, acts_r.mlp_post),
            "matched_attention_distance": mt.matched_attention_distance(acts_m.pattern, acts_r.pattern),
            "stitching_kl": mt.stitching_kl(target, acts_r.resid_mid, acts_m.resid_mid, acts_m.logits, k=k),
            "embedding_spectrum_cosine": mt.embedding_spectrum_cosine(
                target.embed.W_E.detach().double(), replacement.embed.W_E.detach().double(), F.to(acts_m.logits.device)),
        })
    return res


def grade_models(target: GrokTransformer, replacement: GrokTransformer, **kw) -> dict:
    """Grade replacement against target (both GrokTransformer) on the grading device, in float64.

    float64 matches the agent's numpy self-test: a replacement whose weights rely on large cancelling terms must not
    pass the self-test and then fail grading because of float32 rounding (targets are float32 checkpoints, so casting
    them up is exact).
    """
    dev = device()
    target = target.to(dev, torch.float64).eval()
    replacement = replacement.to(dev, torch.float64).eval()
    return grade_acts(collect(target, dev), collect(replacement, dev), target, replacement, **kw)


def grade_npz(target: GrokTransformer, path: Path | None, **kw) -> dict:
    """Grade an agent's weights.npz; missing/invalid files are graded as the all-zero model with failed=True."""
    error = None
    try:
        if path is None or not Path(path).exists():
            raise weights.WeightsError("no weights.npz produced")
        arrays = weights.load_npz(Path(path))
    except (weights.WeightsError, ValueError, OSError, KeyError) as exc:  # any unreadable file counts as failure
        error, arrays = str(exc), weights.zeros()
    res = grade_models(target, weights.to_model(arrays), **kw)
    res["failed"] = error is not None
    res["error"] = error
    return res
