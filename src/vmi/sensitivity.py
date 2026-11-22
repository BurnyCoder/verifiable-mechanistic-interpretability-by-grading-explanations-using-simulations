"""Preregistered sensitivity analyses that need no new agent runs (preregistration §8): `vmi sensitivity ROUND`.

Local explanation:
- `grader_sensitivity`  re-scores every T0 replacement (main-simulator runs of the main and controls stages, and the
                        reference compiler's build for each explanation) under other grader settings — ridge λ × 10^±2,
                        3 and 10 folds, other fold seeds, each site alone, orthogonal instead of general linear
                        alignment — and checks whether the ordering of explanations by mean HS survives (Kendall τ
                        against the preregistered settings). Post hoc (audit 2026-11-22): whitened soundness (fully
                        invariant to invertible linear maps), the retrains and random-initialisation networks under
                        every variant, and H7's margin over the wrong targets under the whitened variants.
- `explainer_sensitivity` re-runs the explainer on T0 with other key-frequency (τ ∈ {0.15, 0.35}) and neuron-FVE
                        (0.75, 0.9) thresholds, re-compiles E2–E4 and grades the builds (no agent).
Results go to results/sensitivity/; tables are generated from them.
Global context: part of the hyperparameter-sensitivity stage; the agent-side hyperparameter (effort) needs new runs and
is in the manifests. Orthogonal alignment: orthogonal Procrustes rotation R = UVᵀ (Schönemann 1966) with the
least-squares isotropic scale s = tr(S)/‖X‖²_F (Schönemann & Carroll 1970, Psychometrika 35:245–255,
https://doi.org/10.1007/BF02291266), fitted on training folds exactly like the ridge map.
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
    # Post hoc (audit 2026-11-22): soundness pooled over M̂'s dimensions is not invariant to non-orthogonal maps of M̂;
    # whitening M̂'s states first makes both directions invariant to any invertible linear map (two eigenvalue cut-offs).
    "whitened soundness, eig > 1e-6·max (post hoc)": {"whiten": 1e-6},
    "whitened soundness, eig > 1e-3·max (post hoc)": {"whiten": 1e-3},
}
WHITENED = [name for name, kw in VARIANTS.items() if "whiten" in kw]


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


def cv_r2_whitened(X: torch.Tensor, Y: torch.Tensor, k: int = mt.N_FOLDS, seed: int = mt.FOLD_SEED,
                   rel_eig: float = 1e-6, ridge_rel: float = mt.RIDGE_REL) -> float:
    """Held-out R² of Y from X after whitening Y with its training-fold covariance (post hoc, audit 2026-11-22).

    `metrics.cv_r2` pools SSE/SST over Y's dimensions, so Y's high-variance directions dominate and the score changes
    under a non-orthogonal map of Y. Here, per fold, Y is centred with the training mean and projected on the
    eigenvectors of its training covariance whose eigenvalue exceeds rel_eig·max, each scaled to unit variance; then the
    same ridge map as `metrics.cv_r2` predicts the whitened held-out rows, and SSE/SST is pooled. Every direction of Y
    counts equally, so the score is invariant to any invertible linear map of Y (and, through the regression, of X): it
    is a held-out analogue of Σρᵢ²/dim(Y kept), which equals R²_CCA, the mean squared canonical correlation, when Y's
    kept dimension ≤ X's (Kornblith et al. 2019, §4, eq. 7,
    https://arxiv.org/abs/1905.00414; Raghu et al. 2017, https://arxiv.org/abs/1706.05806). The cut-off drops
    directions with numerically zero variance, which have no meaning to whiten."""
    X, Y = X.double(), Y.double()
    sse = sst = 0.0
    for test_idx in mt.folds(X.shape[0], k, seed):
        test = torch.as_tensor(test_idx, device=X.device)
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        mask[test] = False
        mx, my = X[mask].mean(0), Y[mask].mean(0)
        Yc = Y[mask] - my
        evals, evecs = torch.linalg.eigh(Yc.T @ Yc / Yc.shape[0])  # training-fold covariance of Y
        keep = evals > rel_eig * float(evals.max().clamp_min(1e-300))
        if not bool(keep.any()):
            continue  # Y has no variance in this fold: contributes nothing (score 0 if never any)
        Wh = evecs[:, keep] / evals[keep].sqrt()  # whitening map onto the kept directions
        W = mt.ridge_map(X[mask] - mx, Yc @ Wh, ridge_rel)
        yt = (Y[test] - my) @ Wh  # whitened held-out rows (centred with the training mean)
        sse += float(((yt - (X[test] - mx) @ W) ** 2).sum())
        sst += float((yt ** 2).sum())
    if sst <= 1e-12 * X.shape[0]:
        return 0.0
    return float(min(max(1.0 - sse / sst, 0.0), 1.0))


def hs_variant(acts_m, acts_r, *, ridge_rel=mt.RIDGE_REL, k=mt.N_FOLDS, seed=mt.FOLD_SEED, sites=SITES, orth=False,
               whiten: float | None = None) -> float:
    """HS (mean over `sites` of the harmonic mean of both directions) under one grader setting."""
    vals = []
    for s in sites:
        xm, xr = acts_m.site(s), acts_r.site(s)
        if orth:
            comp, sound = cv_r2_orth(xr, xm, k=k, seed=seed), cv_r2_orth(xm, xr, k=k, seed=seed)
        elif whiten is not None:  # completeness unchanged (already invariant); soundness on M̂'s whitened states
            comp = mt.cv_r2(xr, xm, k=k, ridge_rel=ridge_rel, seed=seed)
            sound = cv_r2_whitened(xm, xr, k=k, seed=seed, rel_eig=whiten, ridge_rel=ridge_rel)
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
    # No-agent comparison networks (H4's retrains and the random-initialisation floor) under every variant
    from grokviz.model import GrokTransformer
    from vmi.nonagent import WRONG
    own = targets.resolve(target).run_id
    others = [f"wd1_s{s}" for s in range(10) if f"wd1_s{s}" != own][:9]  # same 9 retrains as nonagent.baselines
    nets = [("retrain", r, targets.load_model(settings, r)) for r in others]
    nets += [("random init", f"seed {1000 + s}", GrokTransformer(seed=1000 + s)) for s in range(5)]
    for level, name, net in nets:
        acts_r = collect(net.to(dev, torch.float64).eval(), dev)
        rows.append({"kind": "baseline", "level": level, "id": name, "hs": {v: hs_variant(acts_m, acts_r, **kw) for v, kw in VARIANTS.items()}})
    default = next(iter(VARIANTS))
    summary = {}
    for kind in ("agent", "compiler", "baseline"):
        means = {name: {} for name in VARIANTS}
        for name in VARIANTS:
            for lv in LEVELS + ["retrain", "random init"]:
                v = [r["hs"][name] for r in rows if r["kind"] == kind and r["level"] == lv]
                if v:
                    means[name][lv] = float(np.mean(v))
        summary[kind] = {name: {"means": means[name], "kendall_tau_vs_preregistered": _order_tau(means[default], means[name])}
                         for name in VARIANTS}
    # H7 under the whitened variants: each E3/E4 agent replacement against every wrong target, d = HS(T0) − max HS(wrong)
    e34 = [r for r in rows if r["kind"] == "agent" and r["level"] in ("E3", "E4")]
    wrong = {name: {r["id"]: [] for r in e34} for name in [default] + WHITENED}
    for t in WRONG:
        acts_t = collect(targets.load_model(settings, t).to(dev, torch.float64).eval(), dev)
        for r in e34:
            acts_r = collect(weights.to_model(weights.load_npz(Path(r["npz"]))).to(dev, torch.float64).eval(), dev)
            for name in wrong:
                wrong[name][r["id"]].append(hs_variant(acts_t, acts_r, **VARIANTS[name]))
    h7 = {name: {rid: r_hs - max(wrong[name][rid]) for rid, r_hs in ((r["id"], r["hs"][name]) for r in e34)} for name in wrong}
    out = {"target": target, "round": max_round, "variants": list(VARIANTS), "rows": rows, "summary": summary,
           "h7_d": h7, "wrong_targets": WRONG}
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
