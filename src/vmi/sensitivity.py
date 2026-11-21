"""Preregistered sensitivity analyses that need no new agent runs (preregistration §8): `vmi sensitivity ROUND`.

Local explanation:
- `grader_sensitivity`  re-scores every T0 replacement (main-simulator runs of the main and controls stages, and the
                        reference compiler's build for each explanation) under other grader settings — ridge λ × 10^±2,
                        3 and 10 folds, other fold seeds, each site alone, orthogonal instead of general linear
                        alignment — and checks whether the ordering of explanations by mean HS survives (Kendall τ
                        against the preregistered settings).
- `explainer_sensitivity` re-runs the explainer on T0 with other key-frequency (τ ∈ {0.15, 0.35}) and neuron-FVE
                        (0.75, 0.9) thresholds, re-compiles E2–E4 and grades the builds (no agent).
Results go to results/sensitivity/; tables are generated from them.
Global context: part of the hyperparameter-sensitivity stage; the agent-side hyperparameter (effort) needs new runs and
is in the manifests. Orthogonal alignment: orthogonal Procrustes with an isotropic scale (Schönemann 1966; scipy
`orthogonal_procrustes`), fitted on training folds exactly like the ridge map.
"""

from __future__ import annotations  # postponed annotations

import json  # results
from pathlib import Path  # paths

import numpy as np  # arrays
import torch  # activations
from scipy import stats  # Kendall tau

from vmi.config import REPO_ROOT, Settings  # paths
from vmi.grader import metrics as mt  # folds, ridge R²
from vmi.grader.activations import SITES, collect  # activations

RES = REPO_ROOT / "results"
LEVELS = ["E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble"]
VARIANTS: dict[str, dict] = {
    "preregistered (λ 1e-6, 5 folds, seed 0, 3 sites, general linear)": {},
    "ridge λ × 100": {"ridge_rel": mt.RIDGE_REL * 100},
    "ridge λ / 100": {"ridge_rel": mt.RIDGE_REL / 100},
    "3 folds": {"k": 3},
    "10 folds": {"k": 10},
    "fold seed 1": {"seed": 1},
    "fold seed 2": {"seed": 2},
    "resid_mid only": {"sites": ("resid_mid",)},
    "mlp_post only": {"sites": ("mlp_post",)},
    "resid_post only": {"sites": ("resid_post",)},
    "orthogonal alignment (with scale)": {"orth": True},
}


def cv_r2_orth(X: torch.Tensor, Y: torch.Tensor, k: int = mt.N_FOLDS, seed: int = mt.FOLD_SEED) -> float:
    """Held-out R² of Y from X under a scaled orthogonal map (same dims), fitted per training fold.

    Per fold: centre with training means, SVD of XcᵀYc = U S Vᵀ, R = U Vᵀ, scale s = tr(S)/‖Xc‖²_F (the least-squares
    optimum for a similarity transform), predict s·(X − m_x)R + m_y. Pooled SSE/SST as in `metrics.cv_r2`."""
    X, Y = X.double(), Y.double()
    sse = sst = 0.0
    for test_idx in mt.folds(X.shape[0], k, seed):
        test = torch.as_tensor(test_idx, device=X.device)
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        mask[test] = False
        mx, my = X[mask].mean(0), Y[mask].mean(0)
        Xc, Yc = X[mask] - mx, Y[mask] - my
        U, S, Vh = torch.linalg.svd(Xc.T @ Yc)
        R = U @ Vh
        s = float(S.sum()) / max(float((Xc ** 2).sum()), 1e-300)
        pred = s * (X[test] - mx) @ R + my
        sse += float(((Y[test] - pred) ** 2).sum())
        sst += float(((Y[test] - my) ** 2).sum())
    if sst <= 1e-12 * X.shape[0]:
        return 0.0
    return float(min(max(1.0 - sse / sst, 0.0), 1.0))


def hs_variant(acts_m, acts_r, *, ridge_rel=mt.RIDGE_REL, k=mt.N_FOLDS, seed=mt.FOLD_SEED, sites=SITES, orth=False) -> float:
    """HS (mean over `sites` of the harmonic mean of both directions) under one grader setting."""
    vals = []
    for s in sites:
        xm, xr = acts_m.site(s), acts_r.site(s)
        if orth:
            comp, sound = cv_r2_orth(xr, xm, k=k, seed=seed), cv_r2_orth(xm, xr, k=k, seed=seed)
        else:
            comp, sound = mt.cv_r2(xr, xm, k=k, ridge_rel=ridge_rel, seed=seed), mt.cv_r2(xm, xr, k=k, ridge_rel=ridge_rel, seed=seed)
        vals.append(0.0 if comp + sound == 0 else 2 * comp * sound / (comp + sound))
    return float(np.mean(vals))


def _units(settings: Settings, max_round: int, target: str) -> list[dict]:
    """Replacements to re-score: main-simulator agent runs of the main/controls stages and the compiler builds."""
    from vmi.analysis.hypotheses import _select, load_runs
    units = []
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    for lv in LEVELS:
        for r in _select(runs, target=target, level=lv):
            units.append({"kind": "agent", "level": lv, "id": r["run_id"],
                          "npz": settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"})
        c = settings.runs_dir / "_compiler" / target / f"{lv}.npz"
        if c.exists():
            units.append({"kind": "compiler", "level": lv, "id": f"compiler/{target}/{lv}", "npz": c})
    return units


def _order_tau(means_a: dict, means_b: dict) -> float | None:
    common = [lv for lv in LEVELS if lv in means_a and lv in means_b]
    if len(common) < 3:
        return None
    return float(stats.kendalltau([means_a[lv] for lv in common], [means_b[lv] for lv in common]).statistic)


def grader_sensitivity(settings: Settings, max_round: int, target: str = "T0") -> dict:
    """HS of every replacement under every grader variant, and the stability of the explanation ordering."""
    from vmi import targets, weights
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_m = collect(targets.load_model(settings, target).to(dev, torch.float64).eval(), dev)
    rows = []
    for u in _units(settings, max_round, target):
        rep = weights.to_model(weights.load_npz(Path(u["npz"]))).to(dev, torch.float64).eval()
        acts_r = collect(rep, dev)
        rows.append({**{k: str(v) if k == "npz" else v for k, v in u.items()},
                     "hs": {name: hs_variant(acts_m, acts_r, **kw) for name, kw in VARIANTS.items()}})
    default = next(iter(VARIANTS))
    summary = {}
    for kind in ("agent", "compiler"):
        means = {name: {} for name in VARIANTS}
        for name in VARIANTS:
            for lv in LEVELS:
                v = [r["hs"][name] for r in rows if r["kind"] == kind and r["level"] == lv]
                if v:
                    means[name][lv] = float(np.mean(v))
        summary[kind] = {name: {"means": means[name], "kendall_tau_vs_preregistered": _order_tau(means[default], means[name])}
                         for name in VARIANTS}
    out = {"target": target, "round": max_round, "variants": list(VARIANTS), "rows": rows, "summary": summary}
    p = RES / "sensitivity" / f"grader_{target}_round{max_round}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out


def explainer_sensitivity(settings: Settings, target: str = "T0") -> dict:
    """Explainer thresholds varied; E2–E4 re-rendered and re-compiled; compiler builds graded (no agent)."""
    from vmi import targets, weights
    from vmi.explainer.analysis import analyse
    from vmi.explainer.ladder import make_spec
    from vmi.grader.score import device, grade_models
    from vmi.reference.compile import compile_spec
    dev = device()
    model = targets.load_model(settings, target, dev)
    grid = targets.grid(target, dev)
    rows = []
    for tau, fve in ((None, None), (0.15, None), (0.35, None), (None, 0.75), (None, 0.9)):
        kw = {k: v for k, v in (("tau", tau), ("neuron_fve", fve)) if v is not None}
        an = analyse(model, grid, **kw)
        for lv in ("E2", "E3", "E4"):
            spec = make_spec(an, lv, target)
            g = grade_models(targets.load_model(settings, target), weights.to_model(weights.validate(compile_spec(spec))), secondary=False)
            rows.append({"tau": an["tau"], "neuron_fve": an["neuron_fve_threshold"], "key_freqs": an["key_freqs"], "level": lv,
                         "hs": g["hs"], "kl_out": g["kl_out"], "accuracy": g["accuracy"]})
    out = {"target": target, "rows": rows}
    p = RES / "sensitivity" / f"explainer_{target}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out
