"""Render an explanation spec as the text the simulator reads (EXPLANATION.md, plus neurons.csv for E4/W-scramble).

Local explanation: one template per algorithm (none / Clock / Pizza) and one quantitative section per field present
in the spec, so W-freq, W-role and W-scramble use exactly the same wording and length as E3/E4 and differ only in the
corrupted content. `length_stats` measures the explanation's description length (characters, gzip bytes, numbers).
Global context: the agent never sees the spec, only this text; the reference compiler sees the spec. Both carry the
same information.
Sources: algorithm descriptions paraphrase Nanda et al. 2023 §3.1 (https://arxiv.org/abs/2301.05217) and Zhong et al.
2023 §3 (https://arxiv.org/abs/2306.17844).
"""

from __future__ import annotations  # postponed annotations

import csv  # neurons.csv
import gzip  # description length in compressed bytes
import io  # in-memory CSV
import re  # count numbers

NEURON_COLUMNS = ["neuron", "freq", "bias", "a_amp", "a_phase", "b_amp", "b_phase", "sum_amp", "sum_phase",
                  "diff_amp", "diff_phase", "pre_fit_fve", "out_amp", "out_phase", "out_fit_fve"]

E0_TEXT = """# Explanation

No mechanistic explanation is available for this model. What is known: it has the architecture in
ARCHITECTURE.md, it was trained on 30% of all (a, b) pairs with cross-entropy loss, AdamW and weight decay, and it
computes (a + b) mod 113 correctly on every input.
"""

CLOCK_ALGO = """## Algorithm ("Fourier multiplication")

The network adds numbers by rotating points on circles:

1. **Embedding.** Each number t is represented by cos(w_k t) and sin(w_k t) for a handful of *key frequencies* k,
   with w_k = 2πk/113 — every number is a point on several circles, one per key frequency.
2. **Attention.** At the "=" position the heads attend to positions 0 and 1 (almost not at all to "=" itself) and
   move cos/sin(w_k a) and cos/sin(w_k b) into the "=" position. The attention weights themselves depend on a and b,
   which already contributes products of these terms.
3. **MLP.** The ReLU neurons form products and compute cos(w_k(a+b)) and sin(w_k(a+b)) using
   cos(w(a+b)) = cos(wa)cos(wb) − sin(wa)sin(wb) and sin(w(a+b)) = sin(wa)cos(wb) + cos(wa)sin(wb).
   Each neuron responds mainly to a single frequency.
4. **Unembedding.** The logit of answer c is approximately
   Σ_k α_k cos(w_k(a+b−c)) = Σ_k α_k [cos(w_k(a+b))cos(w_k c) + sin(w_k(a+b))sin(w_k c)],
   read from the MLP output through one cos and one sin direction per key frequency. The waves of all key
   frequencies peak together exactly at c = (a+b) mod 113 and cancel elsewhere, so that answer wins.
"""

PIZZA_ALGO = """## Algorithm ("Pizza")

The network adds numbers using midpoints of points on circles:

1. **Embedding.** Each number t is represented by cos(w_k t) and sin(w_k t) for a handful of *key frequencies* k,
   with w_k = 2πk/113 — every number is a point on several circles, one per key frequency.
2. **Attention.** Every head attends equally (weight 0.5 each) to positions 0 and 1 and not to "=", so the "="
   position receives cos(w_k a) + cos(w_k b) and sin(w_k a) + sin(w_k b), i.e. 2cos(w_k(a−b)/2) times the unit
   vector at angle w_k(a+b)/2 — the midpoint of the two points on the circle.
3. **MLP.** Neurons of frequency k compute ReLU(cos(w_k a + φ) + cos(w_k b + φ)) for evenly spaced phases φ, i.e. the
   absolute value of the midpoint vector's projection onto direction φ.
4. **Unembedding.** The logit of answer c is approximately Σ_k α_k |cos(w_k(a−b)/2)| cos(w_k(a+b−c)). Each term peaks
   at c = (a+b) mod 113; the factor |cos(w_k(a−b)/2)| makes each frequency weak for some a−b, and the sum over key
   frequencies covers those cases.
"""


def _fmt(x: float) -> str:
    """Three significant digits, plain notation where possible."""
    return f"{x:.3g}"


def _neurons_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=NEURON_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: (r[c] if c in ("neuron", "freq") else f"{float(r[c]):.4g}") for c in NEURON_COLUMNS})
    return buf.getvalue()


def render(spec: dict) -> tuple[str, dict[str, str]]:
    """Return (EXPLANATION.md text, {extra file name: content})."""
    if spec["algorithm"] == "none":
        return E0_TEXT, {}
    parts = ["# Explanation\n", PIZZA_ALGO if spec["algorithm"] == "pizza" else CLOCK_ALGO]
    extra: dict[str, str] = {}
    K_e, K_m = spec.get("embed_freqs"), spec.get("mlp_freqs")
    if K_e:
        lines = ["## This model's key frequencies\n"]
        if K_m and list(K_m) != list(K_e):
            lines.append(f"- The embedding and the attention heads use frequencies k ∈ {{{', '.join(map(str, K_e))}}}.")
            lines.append(f"- The MLP neurons and the unembedding use frequencies k ∈ {{{', '.join(map(str, K_m))}}}.")
        else:
            lines.append(f"- Key frequencies: k ∈ {{{', '.join(map(str, K_e))}}} (w_k = 2πk/113). "
                         "All four steps above use these frequencies.")
        lines.append(f"- The model is very confident: its mean cross-entropy over all 12,769 inputs is "
                     f"{spec['loss_all_inputs']:.3g} nats.")
        parts.append("\n".join(lines) + "\n")
    if spec.get("logit_amplitudes"):
        amps = spec["logit_amplitudes"]
        lines = ["## Quantities\n", "**Logits.** Fitting logit(a, b, c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β over all inputs "
                 f"and answers gives (fraction of logit variance explained: {_fmt(spec['logit_fit_fve'])}):\n",
                 "| k | α_k |", "|---|---|"]
        lines += [f"| {k} | {_fmt(a)} |" for k, a in amps.items()]
        lines.append(f"\nβ = {_fmt(spec['logit_offset'])}; logit standard deviation at \"=\": {_fmt(spec['logit_std'])}; "
                     f"mean log-probability of the correct answer: {spec['mean_correct_logprob']:.3g}.")
        if spec.get("embedding_norm_by_freq"):
            en = spec["embedding_norm_by_freq"]
            lines.append("\n**Embedding.** Norm of the embedding's Fourier component (cos and sin combined, over the 128 "
                         "residual dimensions) per key frequency: " + ", ".join(f"k={k}: {_fmt(v)}" for k, v in en.items()) + ".")
        if spec.get("neuron_clusters") is not None:
            nc = spec["neuron_clusters"]
            lines.append(f"\n**MLP neurons.** A neuron belongs to frequency k if more than {_fmt(spec['neuron_fve_threshold'])} "
                         "of its activation variance is explained by the constant, linear and quadratic terms of frequency k "
                         "in a and b. Neurons per frequency: " + ", ".join(f"k={k}: {n}" for k, n in nc.items())
                         + f"; {spec['neurons_unclustered']} of the 512 neurons are not clearly single-frequency.")
        if spec.get("heads"):
            lines += ["\n**Attention heads.** Attention from \"=\" to position 0 (a) is fitted as "
                      "A ≈ 0.5 + α(cos wa − cos wb) + β(sin wa − sin wb) with w = 2πk/113; attention to b is about 1 − A:\n",
                      "| head | k | α | β | variance explained | attention to \"=\" |", "|---|---|---|---|---|---|"]
            lines += [f"| {h['head']} | {h['freq']} | {_fmt(h['alpha'])} | {_fmt(h['beta'])} | {_fmt(h['pattern_fve'])} | "
                      f"{_fmt(h['attn_to_eq'])} |" for h in spec["heads"]]
            if any("ov_norm_by_freq" in h for h in spec["heads"]):
                lines.append("\nWhat each head writes (norm of its value-output circuit applied to the embeddings, per key "
                             "frequency): " + "; ".join(
                                 f"head {h['head']}: " + ", ".join(f"k={k}: {_fmt(v)}" for k, v in h["ov_norm_by_freq"].items())
                                 for h in spec["heads"]) + ".")
        elif spec["algorithm"] == "pizza":
            lines.append("\n**Attention heads.** All four heads attend 0.5 to a and 0.5 to b.")
        if spec.get("readoff"):
            ro = spec["readoff"]
            lines.append("\n**Read-out.** Projecting the MLP activations onto the unembedding's cos(w_k c) and sin(w_k c) "
                         "directions gives a multiple of cos(w_k(a+b)) or sin(w_k(a+b)); coefficient (fraction of variance "
                         "explained): " + "; ".join(f"k={r['freq']} {r['component']}: {_fmt(r['coef_unit_dir'])} "
                                                     f"({_fmt(r['fve'])})" for r in ro) + ".")
        if spec.get("wl_residual_sq_frac") is not None:
            lines.append(f"\nThe neuron-to-logit map W_U·W_out has {_fmt(100 * spec['wl_residual_sq_frac'])}% of its squared "
                         "norm outside the 2×(number of key frequencies) cos/sin directions of the key frequencies.")
        parts.append("\n".join(lines) + "\n")
    if spec.get("neurons"):
        extra["neurons.csv"] = _neurons_csv(spec["neurons"])
        parts.append("""## Per-neuron table (`neurons.csv`)

One row per MLP neuron (0–511). `freq` is the neuron's frequency k (w = 2πk/113). The neuron's pre-activation
(before ReLU) at the "=" position is approximately

bias + a_amp·cos(w·a − a_phase) + b_amp·cos(w·b − b_phase) + sum_amp·cos(w·(a+b) − sum_phase)
     + diff_amp·cos(w·(a−b) − diff_phase)

(phases in degrees; `pre_fit_fve` = fraction of the pre-activation's variance this formula explains). The neuron's
effect on the logit of answer c (its column of W_U·W_out) is approximately out_amp·cos(w·c − out_phase) plus a
constant (`out_fit_fve` = fraction explained).
""")
    return "\n".join(parts), extra


def length_stats(text: str, extra: dict[str, str]) -> dict:
    """Description length of an explanation: characters, gzip bytes and count of numbers (all files together)."""
    full = text + "".join(extra.values())
    return {"chars": len(full), "gzip_bytes": len(gzip.compress(full.encode("utf-8"), mtime=0)),
            "numbers": len(re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", full))}
