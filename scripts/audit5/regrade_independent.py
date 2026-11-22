"""Audit 5 (2026-11-22): independent re-grading of the scores the hypotheses rest on.

Local explanation: re-computes output KL, task accuracy and hidden-state similarity (HS) for every reported agent run,
every reference-compiler build, the H4 baselines (retrains, distilled students), the random initialisations, the H4b
planted snapshots, the H7 wrong-target gradings and a systematic sample of the H8 trajectory gradings, with code
written for this audit from two texts only:
  - the forward pass in `sandbox_template/ARCHITECTURE.md` (the specification the agents receive): x_i = W_E[:, t_i] +
    W_pos[i]; per head softmax((W_Q x_2)·(W_K x_j)/√32) over j ≤ 2; resid_mid = x_2 + W_O·concat(z); ReLU MLP with biases;
    logits = resid_post·W_U, first 113 classes; only the "=" position (2) is needed for a one-layer model;
  - preregistration §4: KL(P_M ‖ P_M̂) averaged over the 12,769 inputs (natural log, float64); per site 5-fold
    cross-validated ridge R² in both directions, λ = 1e-6·tr(XᵀX)/d on centred training rows, pooled over held-out folds
    and dimensions, clipped to [0, 1]; site score = harmonic mean; HS = mean over resid_mid, mlp_post, resid_post.
Two conventions the preregistration leaves open are taken as documented by the project and varied where it matters:
fold assignment numpy default_rng(0).permutation(n) split into 5 contiguous parts, and the R² denominator about the
training-fold mean (the alternative, the held-out fold's own mean as in sklearn's r2_score, is also reported).
Weights are loaded with the project's loaders (`vmi.targets.load_model`, `numpy.load`) — loading is not under audit;
nothing else from `vmi` is used. Output `results/audit5/regrade_independent.json` (largest absolute differences per
group and every item).
Global context: post hoc audit of deviation 12; it changes no verdict. Audit 4 made a similar check, which this script
does not read.
Sources: preregistration §4 (`reports/preregistration.md`); forward pass — `sandbox_template/ARCHITECTURE.md`, which
follows Nanda et al. 2023 (https://arxiv.org/abs/2301.05217); ridge normal equations — Hoerl & Kennard 1970
(Technometrics 12:55–67); numpy.linalg.solve (https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html).
Run: `.venv/Scripts/python.exe scripts/audit5/regrade_independent.py` (CPU, about 10 minutes).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # repo import path
import time  # timestamped log
from pathlib import Path  # paths

import numpy as np  # all arithmetic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vmi import targets  # noqa: E402  checkpoint loading only
from vmi.config import load  # noqa: E402  data paths

RES = ROOT / "results"
OUT = RES / "audit5" / "regrade_independent.json"
P, NAMES = 113, ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out", "W_U"]
SD_KEYS = {"W_E": "embed.W_E", "W_pos": "pos_embed.W_pos", "W_Q": "blocks.0.attn.W_Q", "W_K": "blocks.0.attn.W_K",
           "W_V": "blocks.0.attn.W_V", "W_O": "blocks.0.attn.W_O", "W_in": "blocks.0.mlp.W_in", "b_in": "blocks.0.mlp.b_in",
           "W_out": "blocks.0.mlp.W_out", "b_out": "blocks.0.mlp.b_out", "W_U": "unembed.W_U"}


def log(msg: str) -> None:
    """Timestamped progress line on the terminal (UTF-8, so Greek letters print on a Windows console too)."""
    sys.stdout.reconfigure(encoding="utf-8")  # https://docs.python.org/3/library/io.html#io.TextIOWrapper.reconfigure
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Weights and forward pass (ARCHITECTURE.md)
# ---------------------------------------------------------------------------------------------------------------
def from_state_dict(sd) -> dict:
    """The 11 arrays of ARCHITECTURE.md from a checkpoint's state dict (float64)."""
    return {k: sd[v].detach().double().cpu().numpy() for k, v in SD_KEYS.items()}


def from_npz(path: Path) -> dict:
    """The 11 arrays from a weights.npz (float64); raises if any is missing."""
    with np.load(path) as z:
        return {k: np.asarray(z[k], dtype=np.float64) for k in NAMES}


def forward(w: dict) -> dict:
    """Logits (first 113 classes) and the three hidden sites at position 2 for all inputs [a, b, =], a-major order."""
    idx = np.arange(P * P)
    a, b = idx // P, idx % P
    x0 = w["W_E"][:, a].T + w["W_pos"][0]  # [n, 128]
    x1 = w["W_E"][:, b].T + w["W_pos"][1]
    x2 = w["W_E"][:, 113] + w["W_pos"][2]  # the same "=" vector for every input
    zs = []
    for h in range(w["W_Q"].shape[0]):
        q = w["W_Q"][h] @ x2  # [32]
        ks = [x0 @ w["W_K"][h].T, x1 @ w["W_K"][h].T, np.broadcast_to(w["W_K"][h] @ x2, (P * P, w["W_K"].shape[1]))]
        vs = [x0 @ w["W_V"][h].T, x1 @ w["W_V"][h].T, np.broadcast_to(w["W_V"][h] @ x2, (P * P, w["W_V"].shape[1]))]
        s = np.stack([k @ q for k in ks], 1) / np.sqrt(w["W_Q"].shape[1])  # [n, 3] scores of position 2 over j ≤ 2
        s -= s.max(1, keepdims=True)
        A = np.exp(s)
        A /= A.sum(1, keepdims=True)
        zs.append(sum(A[:, j:j + 1] * vs[j] for j in range(3)))  # [n, 32]
    resid_mid = x2 + np.concatenate(zs, 1) @ w["W_O"].T  # head-major concat, then W_O
    mlp_post = np.maximum(resid_mid @ w["W_in"].T + w["b_in"], 0.0)
    resid_post = resid_mid + mlp_post @ w["W_out"].T + w["b_out"]
    return {"logits": (resid_post @ w["W_U"])[:, :P], "resid_mid": resid_mid, "mlp_post": mlp_post, "resid_post": resid_post}


# ---------------------------------------------------------------------------------------------------------------
# Metrics (preregistration §4)
# ---------------------------------------------------------------------------------------------------------------
def log_softmax(L: np.ndarray) -> np.ndarray:
    m = L.max(1, keepdims=True)
    return L - m - np.log(np.exp(L - m).sum(1, keepdims=True))


def kl(Lm: np.ndarray, Lr: np.ndarray) -> float:
    """Mean over inputs of KL(softmax(Lm) ‖ softmax(Lr)) in nats."""
    lp, lq = log_softmax(Lm), log_softmax(Lr)
    return float((np.exp(lp) * (lp - lq)).sum(1).mean())


FOLDS = np.array_split(np.random.default_rng(0).permutation(P * P), 5)


def cv_r2(X: np.ndarray, Y: np.ndarray, sst_about: str = "train") -> float:
    """Held-out R² of a ridge-regularised affine map X → Y, pooled over folds and output dimensions, clipped to [0, 1]."""
    sse = sst = 0.0
    for test in FOLDS:
        tr = np.ones(len(X), bool)
        tr[test] = False
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - mx, Y[tr] - my
        G = Xc.T @ Xc
        lam = 1e-6 * np.trace(G) / G.shape[0] + 1e-12
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
        pred = (X[test] - mx) @ W + my
        sse += float(((Y[test] - pred) ** 2).sum())
        ref = my if sst_about == "train" else Y[test].mean(0)
        sst += float(((Y[test] - ref) ** 2).sum())
    if sst <= 1e-12 * len(X):
        return 0.0
    return float(min(max(1.0 - sse / sst, 0.0), 1.0))


def hs(fm: dict, fr: dict, sst_about: str = "train") -> tuple[float, dict]:
    """HS = mean over the three sites of the harmonic mean of completeness R²(M̂→M) and soundness R²(M→M̂)."""
    sites = {}
    for s in ("resid_mid", "mlp_post", "resid_post"):
        c, o = cv_r2(fr[s], fm[s], sst_about), cv_r2(fm[s], fr[s], sst_about)
        sites[s] = 0.0 if c + o == 0 else 2 * c * o / (c + o)
    return float(np.mean(list(sites.values()))), sites


def grade(wm: dict, wr: dict, alt: bool = False) -> dict:
    """KL, accuracy and HS of replacement wr against target wm (alt: also HS with the held-out-mean R² convention)."""
    fm, fr = forward(wm), forward(wr)
    labels = (np.arange(P * P) // P + np.arange(P * P) % P) % P
    h, sites = hs(fm, fr)
    out = {"kl_out": kl(fm["logits"], fr["logits"]), "accuracy": float((fr["logits"].argmax(1) == labels).mean()),
           "hs": h, "sites": sites}
    if alt:
        out["hs_heldout_mean"] = hs(fm, fr, "test")[0]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    settings = load()
    cache: dict = {}

    def target(name: str, ckpt: int = -1) -> dict:
        key = (name, ckpt)
        if key not in cache:
            cache[key] = from_state_dict(targets.load_model(settings, name, ckpt=ckpt).state_dict())
        return cache[key]

    items = []

    def add(group, name, stored: dict, mine: dict):
        d = {k: abs(stored[k] - mine[k]) for k in ("hs", "kl_out", "accuracy") if k in stored and stored[k] is not None}
        items.append({"group": group, "name": name, "stored": {k: stored.get(k) for k in d}, "recomputed": {k: mine[k] for k in d},
                      "abs_diff": d, **({"hs_heldout_mean": mine["hs_heldout_mean"]} if "hs_heldout_mean" in mine else {})})

    # 1. every reported agent run (stage ≠ dev) with saved weights
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]
    runs = [r for r in runs if r["stage"] != "dev"]
    for r in runs:
        f = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        if f.exists():
            add("agent runs", r["run_id"], r, grade(target(r["target"]), from_npz(f), alt=True))
    log(f"agent runs: {sum(i['group'] == 'agent runs' for i in items)} of {len(runs)} regraded")
    # 2. every reference-compiler build
    for t in ("T0", "T1", "T2", "T-new", "T-paper"):
        for p in sorted((RES / "compiler" / t).glob("*.json")):
            f = settings.runs_dir / "_compiler" / t / f"{p.stem}.npz"
            if f.exists():
                add("compiler builds", f"{t} {p.stem}", json.loads(p.read_text()), grade(target(t), from_npz(f)))
    log("compiler builds done")
    # 3. H4 baselines on T0: retrains and distilled students
    base = json.loads((RES / "baselines" / "T0.json").read_text())
    for x in base["retrains"]:
        add("retrains (H4)", x["run"], x, grade(target("T0"), target(x["run"])))
    for x in base["students"]:
        f = settings.data_dir / "trained" / f"student_T0_s{x['seed']}" / "step_020000.npz"
        if f.exists():
            add("distilled students", f"student s{x['seed']}", x, grade(target("T0"), from_npz(f)))
    # 4. random initialisations (GrokTransformer(seed = 1000 + s), as recorded in nonagent.random_init)
    from grokviz.model import GrokTransformer  # initialisation is library code, not under audit
    for x in json.loads((RES / "controls" / "random_init_T0.json").read_text()):
        add("random init", f"seed {1000 + x['seed']}", x, grade(target("T0"), from_state_dict(GrokTransformer(seed=1000 + x["seed"]).state_dict())))
    # 5. H4b planted snapshots: the step-0 model graded against each snapshot
    pl = settings.data_dir / "trained" / "planted_T0"
    init = from_npz(pl / "step_000000.npz")
    for row in json.loads((RES / "controls" / "planted_calibration_T0.json").read_text())["rows"]:
        add("planted snapshots (H4b)", f"step {row['step']}", row, grade(from_npz(pl / f"step_{row['step']:06d}.npz"), init))
    log("baselines, random init and planted snapshots done")
    # 6. H7: each E3/E4 replacement against the twelve wrong targets
    wt = json.loads((RES / "controls" / "wrong_targets_T0.json").read_text())
    for row in wt["rows"]:
        wr = from_npz(settings.runs_dir / row["run_id"] / "work_out" / "weights.npz")
        for name, v in row["hs_wrong"].items():
            add("wrong targets (H7)", f"{row['run_id']} vs {name}", {"hs": v}, grade(target(name), wr))
    log("wrong targets done")
    # 7. H8: a systematic sample of the trajectory gradings (every 40th of the 401 checkpoints)
    tr = json.loads((RES / "controls" / "trajectory_T0.json").read_text())
    steps = tr["steps"]
    for rep in tr["replacements"]:
        wr = from_npz(settings.runs_dir / rep["run_id"] / "work_out" / "weights.npz")
        for i in range(0, len(steps), 40):
            ck = steps[i] // 100  # checkpoints every 100 steps
            add("trajectory sample (H8)", f"{rep['run_id']} step {steps[i]}", {"hs": rep["hs"][i]}, grade(target("T0", ck), wr))
    log("trajectory sample done")
    summary = {}
    for g in dict.fromkeys(i["group"] for i in items):
        rows = [i for i in items if i["group"] == g]
        summary[g] = {"n": len(rows), **{f"max_abs_diff_{k}": max((i["abs_diff"].get(k, 0.0) for i in rows), default=0.0)
                                        for k in ("hs", "kl_out", "accuracy")}}
    agent = [i for i in items if i["group"] == "agent runs"]
    alt = {"max_abs_hs_change_heldout_mean_convention": max(abs(i["hs_heldout_mean"] - i["recomputed"]["hs"]) for i in agent)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"note": "Audit 5, post hoc; independent forward pass (ARCHITECTURE.md) and metrics (preregistration §4).",
                               "summary": summary, "r2_denominator_convention": alt, "items": items}, indent=2),
                   encoding="utf-8", newline="\n")
    for g, s in summary.items():
        log(f"{g}: n {s['n']}, max |ΔHS| {s['max_abs_diff_hs']:.1e}, max |ΔKL| {s['max_abs_diff_kl_out']:.1e}, "
            f"max |Δacc| {s['max_abs_diff_accuracy']:.1e}")
    log(f"HS change under the held-out-mean R² convention (agent runs): at most {alt['max_abs_hs_change_heldout_mean_convention']:.1e}")
    log(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
