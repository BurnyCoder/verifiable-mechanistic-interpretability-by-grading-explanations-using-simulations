"""Audit 7: re-grade the T0 runs that the preregistered tests use, with a forward pass and metrics written from the specs.

Local explanation: for every T0 main and controls run of the main simulator (E0–E4, W-freq, W-alg, W-role, W-scramble;
27 runs) and every T0 reference-compiler build (9), compute output KL and HS from the weight files with numpy only:
the forward pass follows sandbox_template/ARCHITECTURE.md line by line, KL and HS follow §4 of
reports/preregistration.md (ridge λ = 1e-6·tr(XᵀX)/d on training-fold-centred data, 5 folds from seed 0, pooled
held-out R² clipped to [0, 1], harmonic mean per site, mean over resid_mid, mlp_post, resid_post). Then compare with
the stored scores in results/runs/ and results/compiler/T0/.
Global context: seventh audit (2026-11-22). Only the loading of the target checkpoint uses project code
(vmi.targets/vmi.weights read the published state dict into the 11 named arrays); everything that computes a score
is written here. Fold assignment: §4 says "folds fixed by seed 0"; numpy default_rng(0).permutation split into 5 is
the reading used (a different split would change scores only at the level of fold noise).
Sources: ARCHITECTURE.md (forward pass given to every agent); preregistration §4 (metrics); numpy linear algebra
(https://numpy.org/doc/stable/reference/routines.linalg.html).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # project import path
from pathlib import Path  # paths

import numpy as np  # everything numeric

ROOT = Path(__file__).resolve().parents[2]  # repository root
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
P = 113  # modulus; token 113 is "="


def forward(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ARCHITECTURE.md steps 1–6 for every input [a, b, =], returning the '=' position's sites and number logits."""
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    toks = np.stack([a.ravel(), b.ravel(), np.full(P * P, P)], 1)  # row a·113 + b
    x = w["W_E"][:, toks].transpose(1, 2, 0) + w["W_pos"][None]  # step 1: [N, 3, 128]
    q = np.einsum("hdm,npm->nhpd", w["W_Q"], x, optimize=True)  # step 2: queries, keys, values per head
    k = np.einsum("hdm,npm->nhpd", w["W_K"], x, optimize=True)
    v = np.einsum("hdm,npm->nhpd", w["W_V"], x, optimize=True)
    s = np.einsum("nhid,nhjd->nhij", q, k, optimize=True) / np.sqrt(32)
    s = np.where(np.tril(np.ones((3, 3), bool))[None, None], s, -np.inf)  # causal: j ≤ i
    A = np.exp(s - s.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    z = np.einsum("nhij,nhjd->nihd", A, v, optimize=True).reshape(len(toks), 3, 128)  # concat heads, head-major
    mid = x + z @ w["W_O"].T  # step 3
    post = np.maximum(mid @ w["W_in"].T + w["b_in"], 0)  # step 4
    res = mid + post @ w["W_out"].T + w["b_out"]  # step 5
    logits = res @ w["W_U"]  # step 6
    return {"resid_mid": mid[:, 2], "mlp_post": post[:, 2], "resid_post": res[:, 2], "logits": logits[:, 2, :P]}


def kl(lm: np.ndarray, lr: np.ndarray) -> float:
    """Mean KL(softmax(lm) ‖ softmax(lr)) in nats."""
    def logsm(l):
        l = l - l.max(1, keepdims=True)
        return l - np.log(np.exp(l).sum(1, keepdims=True))
    p, q = logsm(lm), logsm(lr)
    return float((np.exp(p) * (p - q)).sum(1).mean())


FOLDS = np.array_split(np.random.default_rng(0).permutation(P * P), 5)  # "folds fixed by seed 0"


def r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Pooled held-out R² of the ridge-regularised affine map X → Y, clipped to [0, 1] (SST about the training mean)."""
    sse = sst = 0.0
    for test in FOLDS:
        tr = np.ones(len(X), bool)
        tr[test] = False
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc = X[tr] - mx
        G = Xc.T @ Xc
        W = np.linalg.solve(G + (1e-6 * np.trace(G) / len(G) + 1e-12) * np.eye(len(G)), Xc.T @ (Y[tr] - my))
        sse += ((Y[test] - (X[test] - mx) @ W - my) ** 2).sum()
        sst += ((Y[test] - my) ** 2).sum()
    return 0.0 if sst <= 1e-12 * len(X) else float(min(max(1 - sse / sst, 0.0), 1.0))


def hs(m: dict, r: dict) -> float:
    """Mean over the three sites of the harmonic mean of completeness R²(M̂→M) and soundness R²(M→M̂)."""
    out = []
    for site in ("resid_mid", "mlp_post", "resid_post"):
        c, s = r2(r[site], m[site]), r2(m[site], r[site])
        out.append(0.0 if c + s == 0 else 2 * c * s / (c + s))
    return float(np.mean(out))


def main() -> dict:
    from vmi import targets, weights  # loading only: published checkpoint → the 11 named arrays
    from vmi.config import load as load_settings  # .env → paths
    settings = load_settings()
    m = forward(weights.from_model(targets.load_model(settings, "T0").double()))
    load = lambda p: {k: v.astype(np.float64) for k, v in np.load(p, allow_pickle=False).items()}  # noqa: E731
    rows = []
    for p in sorted((RES / "runs").glob("*__T0__*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["stage"] in ("main", "controls") and rec["model"] == "claude-haiku-5-5":
            r = forward(load(settings.runs_dir / rec["run_id"] / "work_out" / "weights.npz"))
            rows.append({"name": rec["run_id"], "hs": hs(m, r), "kl": kl(m["logits"], r["logits"]),
                         "stored_hs": rec["hs"], "stored_kl": rec["kl_out"]})
    for lv in ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble"):
        r = forward(load(settings.runs_dir / "_compiler" / "T0" / f"{lv}.npz"))
        st = json.loads((RES / "compiler" / "T0" / f"{lv}.json").read_text())
        rows.append({"name": f"compiler T0 {lv}", "hs": hs(m, r), "kl": kl(m["logits"], r["logits"]),
                     "stored_hs": st["hs"], "stored_kl": st["kl_out"]})
    return {"n": len(rows), "max_abs_diff_hs": max(abs(x["hs"] - x["stored_hs"]) for x in rows),
            "max_abs_diff_kl": max(abs(x["kl"] - x["stored_kl"]) for x in rows), "rows": rows}


if __name__ == "__main__":
    res = main()
    (RES / "audit7").mkdir(exist_ok=True)
    (RES / "audit7" / "regrade.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(f"{res['n']} gradings; largest difference HS {res['max_abs_diff_hs']:.2e}, KL {res['max_abs_diff_kl']:.2e}")
