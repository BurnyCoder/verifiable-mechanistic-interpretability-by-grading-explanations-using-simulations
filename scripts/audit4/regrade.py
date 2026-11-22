"""Audit 4 (2026-11-22): independent re-grading of every stored score from the raw weight files.

Local explanation: the grader under test (`vmi.grader`) runs the models with grokviz/torch and solves the ridge maps
with torch. This script re-implements both from the preregistration (§2 architecture, §4 metrics) with numpy and
scikit-learn only, re-grades every agent run, every reference-compiler build, the no-agent baselines/controls and the
H7 wrong-target gradings from the weight files on disk, and compares with the numbers stored in `results/`.
It also records what the stored numbers hide: per-site R² before clipping to [0, 1], and R² with the total sum of
squares taken about the held-out fold's own mean (the stored grader uses the training-fold mean; §4 does not say).
Global context: `results/audit4/regrade.json` is read by the audit report (`reports/reviews/audit-4-2026-11-22.md`);
nothing in the preregistered analysis reads it. Only the weight *loading* reuses project code (`vmi.targets` reads the
pinned checkpoints, `vmi.weights` validates the 11-array schema); forward pass and metrics are independent.
Sources: architecture and parameter layout — Nanda et al. 2023, App. A, and the paper's code
(https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py:
embed W_E[:, x]; attention scores k·q masked causally, divided by √d_head, softmax; z = Σ attn·v; out = W_O·concat(z);
MLP ReLU(W_in x + b_in), W_out·· + b_out; logits = x @ W_U; no LayerNorm). KL(P_M ‖ P_M̂) in nats and the CV ridge
R² from preregistration §4. Ridge objective ‖y − Xw‖² + α‖w‖² with intercept — scikit-learn `Ridge`
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). log-softmax —
scipy.special.log_softmax (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html).
Run: `.venv/Scripts/python.exe scripts/audit4/regrade.py` (≈ 15 minutes on CPU).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # repo import path
import time  # timestamps for the log
from pathlib import Path  # paths

import numpy as np  # forward pass and metrics
from scipy.special import log_softmax  # numerically stable log-probabilities
from sklearn.linear_model import Ridge  # independent ridge solver

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))  # import vmi for weight loading only
from vmi import targets, weights  # noqa: E402  pinned checkpoint loading and schema validation
from vmi.config import load  # noqa: E402  paths from .env / .env.example

P, D_HEAD = 113, 32  # modulus and head width (preregistration §2)
OUT = ROOT / "results" / "audit4" / "regrade.json"
SITES = ("resid_mid", "mlp_post", "resid_post")  # preregistered sites, '=' position


def log(msg: str) -> None:
    """Timestamped progress line (CLAUDE.md: timestamped logging)."""
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------
# Forward pass (numpy, float64), written from the paper's code
# ------------------------------------------------------------------------------------------------
def forward(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Activations at the '=' position for all 113² inputs [a, b, =], row a·113 + b."""
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    toks = np.stack([a.ravel(), b.ravel(), np.full(P * P, P)], 1)  # '=' is token 113
    x = w["W_E"][:, toks].transpose(1, 2, 0) + w["W_pos"][None]  # [N, 3, d_model]
    q = np.einsum("hkd,npd->nhpk", w["W_Q"], x)  # [N, heads, pos, d_head]
    k = np.einsum("hkd,npd->nhpk", w["W_K"], x)
    v = np.einsum("hkd,npd->nhpk", w["W_V"], x)
    s = np.einsum("nhqk,nhpk->nhqp", q, k) / np.sqrt(D_HEAD)  # scores query q → key p
    s = np.where(np.tril(np.ones((3, 3), bool))[None, None], s, -np.inf)  # causal mask
    s = np.exp(s - s.max(-1, keepdims=True))
    attn = s / s.sum(-1, keepdims=True)  # softmax over keys
    z = np.einsum("nhqp,nhpk->nqhk", attn, v).reshape(len(toks), 3, -1)  # concat heads
    resid_mid = x + z @ w["W_O"].T
    post = np.maximum(resid_mid @ w["W_in"].T + w["b_in"], 0.0)
    resid_post = resid_mid + post @ w["W_out"].T + w["b_out"]
    logits = resid_post @ w["W_U"]
    return {"logits": logits[:, -1, :P], "resid_mid": resid_mid[:, -1], "mlp_post": post[:, -1],
            "resid_post": resid_post[:, -1]}


# ------------------------------------------------------------------------------------------------
# Metrics (preregistration §4)
# ------------------------------------------------------------------------------------------------
def kl_out(lm: np.ndarray, lr: np.ndarray) -> float:
    """Mean over inputs of KL(softmax(lm) ‖ softmax(lr)), nats."""
    pm, qr = log_softmax(lm, -1), log_softmax(lr, -1)
    return float((np.exp(pm) * (pm - qr)).sum(-1).mean())


FOLDS = np.array_split(np.random.default_rng(0).permutation(P * P), 5)  # §4: 5 folds fixed by seed 0


def cv_r2(X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    """5-fold CV ridge R² of Y from X, pooled over dims; λ = 1e-6·tr(XcᵀXc)/d per training fold.
    Returns the raw (unclipped) value with SST about the training-fold mean (the stored grader's choice) and about the
    held-out fold's mean (scikit-learn's r2 convention)."""
    sse = sst_train = sst_test = 0.0
    for test in FOLDS:
        train = np.setdiff1d(np.arange(len(X)), test)
        Xc = X[train] - X[train].mean(0)
        lam = 1e-6 * float((Xc ** 2).sum()) / X.shape[1] + 1e-12  # trace(XcᵀXc)/d without forming XᵀX
        m = Ridge(alpha=lam, fit_intercept=True, solver="cholesky").fit(X[train], Y[train])
        pred = m.predict(X[test])
        sse += float(((Y[test] - pred) ** 2).sum())
        sst_train += float(((Y[test] - Y[train].mean(0)) ** 2).sum())
        sst_test += float(((Y[test] - Y[test].mean(0)) ** 2).sum())
    if sst_train <= 1e-12 * len(X):  # no variance in Y: R² = 0 by the grader's definition
        return {"raw": 0.0, "raw_testmean": 0.0, "degenerate": True}
    return {"raw": 1 - sse / sst_train, "raw_testmean": 1 - sse / max(sst_test, 1e-300), "degenerate": False}


def hs(am: dict, ar: dict) -> dict:
    """HS = mean over sites of the harmonic mean of completeness R²(M̂→M) and soundness R²(M→M̂), each clipped to [0, 1]."""
    clip = lambda r: min(max(r, 0.0), 1.0)  # noqa: E731
    hm = lambda c, s: 0.0 if c + s == 0 else 2 * c * s / (c + s)  # noqa: E731
    sites, tot, tot_tm = {}, 0.0, 0.0
    for s in SITES:
        comp, sound = cv_r2(ar[s], am[s]), cv_r2(am[s], ar[s])
        h = hm(clip(comp["raw"]), clip(sound["raw"]))
        h_tm = hm(clip(comp["raw_testmean"]), clip(sound["raw_testmean"]))
        sites[s] = {"completeness_raw": comp["raw"], "soundness_raw": sound["raw"], "harmonic": h,
                    "harmonic_testmean": h_tm, "degenerate": comp["degenerate"] or sound["degenerate"]}
        tot, tot_tm = tot + h, tot_tm + h_tm
    return {"hs": tot / 3, "hs_testmean": tot_tm / 3, "sites": sites}


def grade(target_acts: dict, w_rep: dict[str, np.ndarray]) -> dict:
    """KL_out, HS and task accuracy of a replacement against precomputed target activations."""
    ar = forward(w_rep)
    labels = (np.arange(P * P) // P + np.arange(P * P) % P) % P
    return {"kl_out": kl_out(target_acts["logits"], ar["logits"]),
            "accuracy": float((ar["logits"].argmax(-1) == labels).mean()), **hs(target_acts, ar)}


# ------------------------------------------------------------------------------------------------
# Jobs: every stored score that comes from a weight file
# ------------------------------------------------------------------------------------------------
def main() -> None:
    st = load()
    res = ROOT / "results"
    tcache: dict[str, dict] = {}

    def target_acts(name: str) -> dict:
        if name not in tcache:
            tcache[name] = forward(weights.from_model(targets.load_model(st, name)))
        return tcache[name]

    jobs = []  # (kind, id, target, weight-loader, stored dict)
    for p in sorted((res / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        npz = st.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        jobs.append(("agent", r["run_id"], r["target"], npz, {"hs": r["hs"], "kl_out": r["kl_out"], "failed": r.get("failed")}))
    for p in sorted((res / "compiler").glob("*/*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        t, lv = p.parent.name, p.stem
        jobs.append(("compiler", f"{t} {lv}", "dev" if t == "dev" else t, st.runs_dir / "_compiler" / t / f"{lv}.npz",
                     {"hs": c["hs"], "kl_out": c["kl_out"]}))
    wt = json.loads((res / "controls" / "wrong_targets_T0.json").read_text(encoding="utf-8"))
    for row in wt["rows"]:
        for t, v in row["hs_wrong"].items():
            jobs.append(("wrong_target", f"{row['run_id']} vs {t}", t, st.runs_dir / row["run_id"] / "work_out" / "weights.npz", {"hs": v}))
    base = json.loads((res / "baselines" / "T0.json").read_text(encoding="utf-8"))
    for r in base["retrains"]:
        jobs.append(("retrain", r["run"], "T0", ("target", r["run"]), {"hs": r["hs"], "kl_out": r["kl_out"]}))
    for r in base["students"]:
        jobs.append(("student", f"s{r['seed']}", "T0", st.data_dir / "trained" / f"student_T0_s{r['seed']}" / "step_020000.npz",
                     {"hs": r["hs"], "kl_out": r["kl_out"]}))
    jobs.append(("weights_dump", "T0", "T0", ("target", "T0"), {"hs": base["weights_dump"]["hs"], "kl_out": base["weights_dump"]["kl_out"]}))
    for r in json.loads((res / "controls" / "random_init_T0.json").read_text(encoding="utf-8")):
        jobs.append(("random_init", f"seed {r['seed']}", "T0", ("init", 1000 + r["seed"]), {"hs": r["hs"], "kl_out": r["kl_out"]}))
    planted = json.loads((res / "controls" / "planted_calibration_T0.json").read_text(encoding="utf-8"))["rows"]

    out = {"rows": [], "planted": []}
    for i, (kind, jid, t, src, stored) in enumerate(jobs):
        if isinstance(src, tuple) and src[0] == "target":
            w = weights.from_model(targets.load_model(st, src[1]))
        elif isinstance(src, tuple) and src[0] == "init":
            from grokviz.model import GrokTransformer  # random init only (the weights under test are the init itself)
            w = weights.from_model(GrokTransformer(seed=src[1]))
        elif Path(src).exists():
            w = weights.load_npz(Path(src))
        else:
            w = weights.zeros()  # §4 failure rule: missing or invalid file = all-zero model
        g = grade(target_acts(t), w)
        row = {"kind": kind, "id": jid, "target": t, "stored": stored, "audit": g,
               "abs_diff_hs": abs(g["hs"] - stored["hs"]),
               "abs_diff_kl": abs(g["kl_out"] - stored["kl_out"]) if "kl_out" in stored else None}
        out["rows"].append(row)
        log(f"[{i + 1}/{len(jobs)}] {kind} {jid}: HS {g['hs']:.6f} (stored {stored['hs']:.6f})"
            + (f", KL {g['kl_out']:.3e} (stored {stored['kl_out']:.3e})" if "kl_out" in stored else ""))
    # H4b: the step-0 planted model graded against each snapshot (target = snapshot, replacement = step 0)
    d = st.data_dir / "trained" / "planted_T0"
    w0 = weights.load_npz(d / "step_000000.npz")
    for r in planted:
        snap = weights.load_npz(d / f"step_{r['step']:06d}.npz")
        g = grade(forward(snap), w0)
        out["planted"].append({"step": r["step"], "stored": r, "audit": g})
        log(f"planted step {r['step']}: HS {g['hs']:.6f} (stored {r['hs']:.6f}), KL {g['kl_out']:.3e} (stored {r['kl_out']:.3e})")
    # Summary: worst reproduction error per kind, and how often clipping or the SST convention matters
    summ = {}
    for kind in sorted({r["kind"] for r in out["rows"]}):
        rs = [r for r in out["rows"] if r["kind"] == kind]
        raws = [v for r in rs for s in r["audit"]["sites"].values() for v in (s["completeness_raw"], s["soundness_raw"])]
        summ[kind] = {"n": len(rs), "max_abs_diff_hs": max(r["abs_diff_hs"] for r in rs),
                      "max_abs_diff_kl": max((r["abs_diff_kl"] for r in rs if r["abs_diff_kl"] is not None), default=None),
                      "n_negative_raw_r2": int(sum(v < 0 for v in raws)), "n_raw_r2": len(raws),
                      "min_raw_r2": float(min(raws)),
                      "max_abs_hs_change_testmean_sst": max(abs(r["audit"]["hs_testmean"] - r["audit"]["hs"]) for r in rs)}
    out["summary"] = summ
    out["planted_max_abs_diff"] = {"hs": max(abs(p["audit"]["hs"] - p["stored"]["hs"]) for p in out["planted"]),
                                   "kl_out": max(abs(p["audit"]["kl_out"] - p["stored"]["kl_out"]) for p in out["planted"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    log(json.dumps({"summary": summ, "planted": out["planted_max_abs_diff"]}, indent=1))


if __name__ == "__main__":
    main()
