"""Independent HS from the preregistration §4 text (numpy), compared with stored run records.

Only the model loading and the activation hooks come from the project; the metric is re-implemented here:
R²(X→Y) = 1 − ΣSSE/ΣSST pooled over output dims, held-out folds (5, seed 0), ridge λ = 1e-6·tr(XᵀX)/d, clipped to
[0, 1]; site = harmonic mean of both directions; HS = mean over resid_mid, mlp_post, resid_post at the '=' position.
Audit 3 (2026-11-22). Needs the run folders (VMI_RUNS_DIR). Usage: uv run python scripts/audit3/independent_hs.py .
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from vmi import targets, weights  # noqa: E402
from vmi.config import load  # noqa: E402

S = load()
P = 113


def acts(model):
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    toks = torch.stack([a.flatten(), b.flatten(), torch.full((P * P,), P)], 1)
    cache = {}
    with torch.no_grad():
        model.double()(toks, cache=cache)
    return {k: cache[n][:, -1].double().numpy() for k, n in
            (("resid_mid", "blocks.0.resid_mid"), ("mlp_post", "blocks.0.mlp.post"), ("resid_post", "blocks.0.resid_post"))}


def r2(X, Y, k=5, seed=0, rel=1e-6):
    perm = np.random.default_rng(seed).permutation(len(X))
    sse = sst = 0.0
    for te in np.array_split(perm, k):
        tr = np.setdiff1d(np.arange(len(X)), te)
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - mx, Y[tr] - my
        G = Xc.T @ Xc
        lam = rel * np.trace(G) / G.shape[0] + 1e-12
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
        sse += ((Y[te] - ((X[te] - mx) @ W + my)) ** 2).sum()
        sst += ((Y[te] - my) ** 2).sum()
    return 0.0 if sst <= 1e-12 * len(X) else float(np.clip(1 - sse / sst, 0, 1))


def hs(am, ar):
    out = []
    for site in ("resid_mid", "mlp_post", "resid_post"):
        c, s = r2(ar[site], am[site]), r2(am[site], ar[site])
        out.append(0.0 if c + s == 0 else 2 * c * s / (c + s))
    return float(np.mean(out))


am = acts(targets.load_model(S, "T0"))
for rid in ("main__T0__E1__claudehaiku55-claudecode__tdef__s0", "main__T0__E3__claudehaiku55-claudecode__tdef__s0",
            "controls__T0__W-freq__claudehaiku55-claudecode__tdef__s0", "controls__T0__W-scramble__claudehaiku55-claudecode__tdef__s0",
            "controls__T0__W-alg__claudehaiku55-claudecode__tdef__s0"):
    stored = json.loads((ROOT / "results/runs" / f"{rid}.json").read_text())["hs"]
    mine = hs(am, acts(weights.to_model(weights.load_npz(S.runs_dir / rid / "work_out/weights.npz"))))
    print(f"{rid[:48]:48s} stored {stored:.6f} independent {mine:.6f} diff {mine - stored:+.1e}")
