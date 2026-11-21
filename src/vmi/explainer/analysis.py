"""Measure a target's Fourier-multiplication mechanism and return it as a JSON-ready parameter dict.

Local explanation: `analyse(model, grid)` runs the paper's analyses through grokviz (key frequencies from W_L,
logit cos-fit amplitudes, embedding spectrum, neuron frequency clusters, attention-head fits, Table 1 read-off) and
adds the per-neuron and per-head fits the most detailed explanation level (E4) needs:
- each neuron's pre-activation fitted on [1, cos/sin(wa), cos/sin(wb), cos/sin(w(a+b)), cos/sin(w(a−b))] at its
  dominant frequency w = 2πk/p, reported as bias + amplitude/phase pairs, and
- the neuron's output map (its column of W_L = W_U W_out) fitted as g·cos(w c − φ_out).
Global context: `render.py` turns this dict into explanation text; `reference/compile.py` turns the same dict back
into weights (agent-free ceiling). No number in any explanation is typed by hand — all come from here.
Sources: grokviz.metrics / grokviz.interventions (MIT) implementing paper Sec. 4 (key frequencies, Table 1, Table 2,
neuron clusters; https://arxiv.org/abs/2301.05217); trig identities cos wa cos wb = ½[cos w(a−b) + cos w(a+b)] etc.
"""

from __future__ import annotations  # postponed annotations

import math  # pi, atan2

import torch  # analysis tensors
from grokviz import fourier  # basis helpers
from grokviz import interventions as iv  # attention fits, Table 1 read-off
from grokviz import metrics as gm  # key frequencies, logit fit, neuron clusters
from grokviz.losses import grid_loss  # float64 cross-entropy on the grid
from grokviz.model import GrokTransformer  # target network

from vmi.config import P  # modulus

NEURON_FVE = gm.NEURON_FVE_THRESHOLD  # 0.85: neuron belongs to a frequency cluster (paper Sec. 4.3)


def _amp_phase(c_cos: float, c_sin: float) -> tuple[float, float]:
    """Write c_cos·cos θ + c_sin·sin θ as r·cos(θ − φ); return (r, φ in degrees, in [0, 360))."""
    r = math.hypot(c_cos, c_sin)
    phi = math.degrees(math.atan2(c_sin, c_cos)) % 360.0
    return r, phi


def _neuron_fits(pre: torch.Tensor, freqs: torch.Tensor, W_L: torch.Tensor, grid) -> list[dict]:
    """Per-neuron functional fit of pre-activations (input side) and of the W_L column (output side)."""
    a = grid.tokens[:, 0].double()
    b = grid.tokens[:, 1].double()
    c = torch.arange(P, dtype=torch.float64, device=pre.device)
    rows = []
    for i in range(pre.shape[1]):
        k = int(freqs[i])
        w = 2 * math.pi * k / P
        X = torch.stack([torch.ones_like(a), torch.cos(w * a), torch.sin(w * a), torch.cos(w * b), torch.sin(w * b),
                         torch.cos(w * (a + b)), torch.sin(w * (a + b)), torch.cos(w * (a - b)), torch.sin(w * (a - b))], 1)
        y = pre[:, i]
        coef = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
        resid = y - X @ coef
        fve = float(1 - resid.pow(2).sum() / (y - y.mean()).pow(2).sum().clamp_min(1e-300))
        col = W_L[:, i]  # this neuron's effect on each output logit c
        Xo = torch.stack([torch.cos(w * c), torch.sin(w * c)], 1)
        co = torch.linalg.lstsq(Xo, (col - col.mean())[:, None]).solution[:, 0]
        out_fve = float(1 - (col - col.mean() - Xo @ co).pow(2).sum() / (col - col.mean()).pow(2).sum().clamp_min(1e-300))
        ra, pa = _amp_phase(float(coef[1]), float(coef[2]))
        rb, pb = _amp_phase(float(coef[3]), float(coef[4]))
        rs, ps = _amp_phase(float(coef[5]), float(coef[6]))
        rd, pd = _amp_phase(float(coef[7]), float(coef[8]))
        ro, po = _amp_phase(float(co[0]), float(co[1]))
        rows.append({"neuron": i, "freq": k, "bias": float(coef[0]), "a_amp": ra, "a_phase": pa, "b_amp": rb, "b_phase": pb,
                     "sum_amp": rs, "sum_phase": ps, "diff_amp": rd, "diff_phase": pd, "pre_fit_fve": fve,
                     "out_amp": ro, "out_phase": po, "out_fit_fve": out_fve})
    return rows


@torch.no_grad()
def analyse(model: GrokTransformer, grid, tau: float = gm.KEY_FREQ_TAU, neuron_fve: float = NEURON_FVE) -> dict:
    """Full mechanism measurement of one target; thresholds are arguments for the explainer-sensitivity analysis."""
    dev = grid.F.device
    model = model.to(dev).eval()
    acts = gm.evaluate(model, grid)  # logits [p², p], pre/post [p², 512], resid_mid, pattern
    F = grid.F
    key = gm.key_frequencies(model, F, tau=tau)  # W_L Fourier norms ≥ τ·max (paper App. C.2)
    coef, logit_fve, _ = gm.logit_cos_fit(acts.logits, grid, key)  # logits ≈ Σ α_k cos(w_k(a+b−c)) + β
    we = fourier.frequency_norms(gm.we_fourier_norms(model, F))  # embedding norm per frequency [(p-1)/2]
    emb_freqs = [k + 1 for k in torch.nonzero(we >= tau * we.max()).flatten().tolist()]
    nf = gm.neuron_fourier(acts.post, F)
    n_freq, n_fve, _ = gm.neuron_frequencies(nf)  # dominant frequency and its variance share per neuron
    clusters = {int(k): int(((n_freq == k) & (n_fve > neuron_fve)).sum()) for k in key}
    W_L = gm.paper_matrices(model, P)["W_L"]  # [p, d_mlp]
    heads = iv.fit_attention_heads(model, acts, grid)["heads"]
    attn_to_eq = acts.pattern[:, :, 2].mean(0).tolist()  # mean attention from '=' to itself, per head
    ov = gm.ov_fourier_norms(model, F)  # [heads, p] OV-circuit spectrum
    ov_freq = fourier.frequency_norms(ov)  # [heads, (p-1)/2]
    table1 = iv.table1_readoff(model, acts, grid, key)
    labels = grid.labels
    loss_all = grid_loss(acts.logits, labels).item()
    lp = torch.log_softmax(acts.logits, -1)
    return {
        "p": P,
        "key_freqs": key,
        "embedding_freqs": emb_freqs,
        "logit_amplitudes": {int(k): float(a) for k, a in zip(key, coef[:-1])},  # α_k
        "logit_offset": float(coef[-1]),
        "logit_fit_fve": logit_fve,
        "loss_all_inputs": loss_all,
        "mean_correct_logprob": float(lp[torch.arange(len(labels), device=dev), labels].mean()),
        "logit_std": float(acts.logits.std()),
        "embedding_norm_by_freq": {int(k): float(we[k - 1]) for k in key},
        "neuron_clusters": clusters,  # neurons with FVE > threshold per key frequency
        "neurons_unclustered": int(512 - sum(clusters.values())),
        "neuron_fve_threshold": neuron_fve,
        "frac_neurons_clustered": float((n_fve > neuron_fve).double().mean()),
        "heads": [{**h, "attn_to_eq": float(attn_to_eq[h["head"]]),
                   "ov_norm_by_freq": {int(k): float(ov_freq[h["head"], k - 1]) for k in key}} for h in heads],
        "readoff": table1,
        "wl_residual_sq_frac": gm.wl_residual_sq_frac(W_L, fourier.key_rows(F, key)),
        "neurons": _neuron_fits(acts.pre, n_freq, W_L, grid),
        "tau": tau,
    }
