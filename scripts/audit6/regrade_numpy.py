"""Audit 6: re-grade stored scores with a third, numpy-only implementation that imports no project code.

Local explanation: the forward pass is written from the architecture of Nanda et al. (2023, §3; one layer, 4 heads of
32, 512 ReLU neurons, no LayerNorm, causal attention, read-out at "=") and the sandbox's ARCHITECTURE.md; KL_out and HS
from preregistration §4 (mean KL(P_M ‖ P_M̂) in nats over all 113² inputs; per site 5-fold cross-validated ridge R² with
λ = 1e-6·tr(XᵀX)/d on centred training folds, folds from numpy default_rng(0).permutation, pooled SSE/SST about the
training-fold mean, clipped to [0, 1], harmonic mean of both directions, mean over resid_mid, mlp_post, resid_post).
Targets are read straight from the published safetensors checkpoints (final step). Graded here: every reported Claude
Haiku run on T0, T1 and T2 (KL and HS against its own target), the nine H4 retrains against T0, and the six T0 E3/E4
replacements against the three wrong targets that are published runs of this family (T1, T2, dev; H7). Not graded here:
T-new and T-paper runs (their targets are not in the published run store) and H8's trajectory.
Global context: a reproduction check of the numbers every hypothesis test uses; no verdict depends on this file.
Sources: Nanda et al. 2023 (https://arxiv.org/abs/2301.05217), preregistration §2/§4; safetensors API
(https://huggingface.co/docs/safetensors/api/numpy); numpy.linalg.solve
(https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html).
Run: uv run python -I scripts/audit6/regrade_numpy.py  →  results/audit6/regrade.json (CPU, ~40 min)
"""

from __future__ import annotations  # postponed annotations

import glob  # run records
import json  # inputs and output
import os  # paths
import sys  # repo root argument

import numpy as np  # everything numeric
from safetensors import safe_open  # published checkpoints

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("VMI_DATA_DIR", "D:/AI/vmi-sim/data") + "/grokking-runs"
RUNS = os.environ.get("VMI_RUNS_DIR", "D:/AI/vmi-sim/runs")
P, D_HEAD = 113, 32  # modulus, head width
TARGET_RUN = {"T0": "wd1_s1", "T1": "wd1_s4", "T2": "wd1_s8", "dev": "wd1_s10"}  # preregistration §2
KEYS = {"W_E": "embed.W_E", "W_pos": "pos_embed.W_pos", "W_Q": "blocks.0.attn.W_Q", "W_K": "blocks.0.attn.W_K",
        "W_V": "blocks.0.attn.W_V", "W_O": "blocks.0.attn.W_O", "W_in": "blocks.0.mlp.W_in", "b_in": "blocks.0.mlp.b_in",
        "W_out": "blocks.0.mlp.W_out", "b_out": "blocks.0.mlp.b_out", "W_U": "unembed.W_U"}


def load_checkpoint(run: str) -> dict:
    """Final checkpoint (last of the stored steps) of a published run, as float64 arrays in the weights.npz layout."""
    f = safe_open(f"{DATA}/{run}/checkpoints.safetensors", "np")
    return {k: f.get_slice(v)[-1].astype(np.float64) for k, v in KEYS.items()}


def forward(w: dict) -> tuple[np.ndarray, dict]:
    """Number logits [p², p] and the three sites at '=' for all inputs [a, b, =] (row a·p + b)."""
    a, b = np.divmod(np.arange(P * P), P)
    toks = np.stack([a, b, np.full_like(a, P)], 1)
    x = w["W_E"][:, toks].transpose(1, 2, 0) + w["W_pos"][None]  # [N, 3, d]: embedding + position
    q = np.einsum("hed,nd->nhe", w["W_Q"], x[:, 2])  # query of '=' (the only position read out)
    k = np.einsum("hed,npd->nhpe", w["W_K"], x)
    v = np.einsum("hed,npd->nhpe", w["W_V"], x)
    s = np.einsum("nhe,nhpe->nhp", q, k) / np.sqrt(D_HEAD)  # '=' sees positions 0..2 under the causal mask
    s -= s.max(-1, keepdims=True)
    pat = np.exp(s)
    pat /= pat.sum(-1, keepdims=True)
    z = np.einsum("nhp,nhpe->nhe", pat, v).reshape(len(a), -1)  # heads concatenated head-major
    mid = x[:, 2] + z @ w["W_O"].T
    post = np.maximum(mid @ w["W_in"].T + w["b_in"], 0.0)
    res = mid + post @ w["W_out"].T + w["b_out"]
    return res @ w["W_U"][:, :P], {"resid_mid": mid, "mlp_post": post, "resid_post": res}


def kl(lm: np.ndarray, lr: np.ndarray) -> float:
    """Mean over inputs of KL(softmax(lm) ‖ softmax(lr)) in nats."""
    def logsm(l):
        l = l - l.max(-1, keepdims=True)
        return l - np.log(np.exp(l).sum(-1, keepdims=True))
    a, b = logsm(lm), logsm(lr)
    return float((np.exp(a) * (a - b)).sum(-1).mean())


def cv_r2(X: np.ndarray, Y: np.ndarray, k: int = 5, seed: int = 0, rel: float = 1e-6) -> float:
    """Held-out R² of Y from X by ridge on centred training folds, pooled, clipped to [0, 1]."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), k)
    sse = sst = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(X)), te)
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - mx, Y[tr] - my
        G = Xc.T @ Xc
        W = np.linalg.solve(G + (rel * np.trace(G) / G.shape[0] + 1e-12) * np.eye(G.shape[0]), Xc.T @ Yc)
        sse += float(((Y[te] - (X[te] - mx) @ W - my) ** 2).sum())
        sst += float(((Y[te] - my) ** 2).sum())
    return 0.0 if sst <= 1e-12 * len(X) else float(np.clip(1 - sse / sst, 0.0, 1.0))


def hs(am: dict, ar: dict) -> float:
    """Mean over sites of the harmonic mean of completeness R²(M̂→M) and soundness R²(M→M̂)."""
    vals = []
    for s in ("resid_mid", "mlp_post", "resid_post"):
        c, so = cv_r2(ar[s], am[s]), cv_r2(am[s], ar[s])
        vals.append(0.0 if c + so == 0 else 2 * c * so / (c + so))
    return float(np.mean(vals))


def main() -> None:
    acts = {t: forward(load_checkpoint(r)) for t, r in TARGET_RUN.items()}
    out = {"runs": [], "retrains": [], "wrong_targets": []}
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{REPO}/results/runs/*.json"))]
    recs = [r for r in recs if r.get("stage") != "dev" and r["model"] == "claude-haiku-5-5" and r["target"] in ("T0", "T1", "T2")]
    for r in recs:
        with np.load(f"{RUNS}/{r['run_id']}/work_out/weights.npz", allow_pickle=False) as f:
            lr, ar = forward({k: f[k].astype(np.float64) for k in KEYS})
        lm, am = acts[r["target"]]
        out["runs"].append({"run_id": r["run_id"], "kl": kl(lm, lr), "hs": hs(am, ar), "stored_kl": r["kl_out"], "stored_hs": r["hs"]})
        print(out["runs"][-1], flush=True)
    base = json.load(open(f"{REPO}/results/baselines/T0.json", encoding="utf-8"))
    for x in base["retrains"]:
        lr, ar = forward(load_checkpoint(x["run"]))
        out["retrains"].append({"run": x["run"], "kl": kl(acts["T0"][0], lr), "hs": hs(acts["T0"][1], ar),
                                "stored_kl": x["kl_out"], "stored_hs": x["hs"]})
    wt = json.load(open(f"{REPO}/results/controls/wrong_targets_T0.json", encoding="utf-8"))
    for row in wt["rows"]:
        with np.load(f"{RUNS}/{row['run_id']}/work_out/weights.npz", allow_pickle=False) as f:
            _, ar = forward({k: f[k].astype(np.float64) for k in KEYS})
        for t in ("T1", "T2", "dev"):
            out["wrong_targets"].append({"run_id": row["run_id"], "target": t, "hs": hs(acts[t][1], ar), "stored_hs": row["hs_wrong"][t]})
    allrows = out["runs"] + out["retrains"] + out["wrong_targets"]
    out["summary"] = {"n_gradings": len(allrows),
                      "max_abs_diff_hs": max(abs(x["hs"] - x["stored_hs"]) for x in allrows),
                      "max_abs_diff_kl": max(abs(x["kl"] - x["stored_kl"]) for x in out["runs"] + out["retrains"])}
    os.makedirs(f"{REPO}/results/audit6", exist_ok=True)
    with open(f"{REPO}/results/audit6/regrade.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(out["summary"])


if __name__ == "__main__":
    main()
