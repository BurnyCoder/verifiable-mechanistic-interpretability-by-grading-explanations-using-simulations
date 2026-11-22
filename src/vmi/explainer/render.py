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

# the published explanation (Nanda et al. 2023); E-paper keeps this wording verbatim
CLOCK_ALGO_PAPER = """## Algorithm ("Fourier multiplication")

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
_MLP_STEP_PAPER = """3. **MLP.** The ReLU neurons form products and compute cos(w_k(a+b)) and sin(w_k(a+b)) using
   cos(w(a+b)) = cos(wa)cos(wb) − sin(wa)sin(wb) and sin(w(a+b)) = sin(wa)cos(wb) + cos(wa)sin(wb).
   Each neuron responds mainly to a single frequency.
"""
# The same step stated as precisely as step 3 of the Pizza text (W-alg), so true and wrong algorithms are equally
# implementable. Checked on the dev target: the a- and b-terms of each neuron's pre-activation have median amplitude
# ratio 0.999 and median phase difference 0.1° (stage 0; the agent had stalled on how ReLU neurons form products).
_MLP_STEP = """3. **MLP.** A neuron of frequency k has pre-activation (before ReLU) ≈ bias + A·[cos(w_k a + φ) + cos(w_k b + φ)]:
   the a-term and the b-term have the same amplitude and phase. The ReLU of this sum is not linear in it; its
   expansion contains the products cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b) and
   cos(w_k a)sin(w_k b), hence cos(w_k(a+b)) = cos(w_k a)cos(w_k b) − sin(w_k a)sin(w_k b) and
   sin(w_k(a+b)) = sin(w_k a)cos(w_k b) + cos(w_k a)sin(w_k b). Neurons of the same frequency differ in their phase φ.
   Each neuron responds mainly to a single frequency.
"""
assert _MLP_STEP_PAPER in CLOCK_ALGO_PAPER
CLOCK_ALGO = CLOCK_ALGO_PAPER.replace(_MLP_STEP_PAPER, _MLP_STEP)

# Audit 4 note (text left unchanged: it is frozen protocol and every W-alg run received it). Two imprecisions against
# Zhong et al. 2023 §3.1 and Yip et al. 2024 §4.2: (i) step 2 — with attention weights 0.5 the "=" position receives
# ½[cos(w_k a) + cos(w_k b)], i.e. cos(w_k(a−b)/2) times the unit vector, not 2cos(…) (a constant factor); (ii) step 3 —
# one ReLU gives the positive part of the projection, and neurons at φ and φ + π together give its absolute value; the
# "evenly spaced phases" follow Yip et al. ("roughly uniform"), not Zhong et al. Neither changes which algorithm is
# described; every agent and the compiler built a working Pizza-side network from it (results/controls/clock_pizza.json).
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
        else:  # audit 5: not exact where W_E's own frequency set differs from W_L's (T0's k = 2; see ladder.py)
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


def render_format(spec: dict, fmt: str) -> tuple[str, dict[str, str]]:
    """Format ablation: the same E3 content as compact math ("math") or as a Python data structure ("code")."""
    K = spec["embed_freqs"]
    amps = {int(k): round(float(v), 3) for k, v in spec["logit_amplitudes"].items()}
    heads = [{"head": h["head"], "k": int(h["freq"]), "alpha": round(h["alpha"], 3), "beta": round(h["beta"], 3),
              "fve": round(h["pattern_fve"], 3), "attn_to_eq": round(h["attn_to_eq"], 4)} for h in spec["heads"]]
    clusters = {int(k): int(v) for k, v in spec["neuron_clusters"].items()}
    readoff = [(int(r["freq"]), r["component"], round(r["coef_unit_dir"], 1), round(r["fve"], 3)) for r in spec["readoff"]]
    if fmt == "math":
        lines = ["# Explanation (mathematical summary)\n",
                 "Notation: w_k = 2πk/113; a, b inputs; c candidate answer; A_h = attention from \"=\" to position 0.\n",
                 f"- K = {{{', '.join(map(str, K))}}}.",
                 "- Embedding: E(t) ∈ span{cos(w_k t), sin(w_k t) : k ∈ K}.",
                 "- Attention (\"=\" query): A_h ≈ 0.5 + α_h(cos w_{k_h}a − cos w_{k_h}b) + β_h(sin w_{k_h}a − sin w_{k_h}b); "
                 "to b ≈ 1 − A_h; to \"=\" ≈ 0.",
                 "  " + "; ".join(f"h{h['head']}: k={h['k']}, α={h['alpha']}, β={h['beta']} (FVE {h['fve']}, A_= {h['attn_to_eq']})" for h in heads),
                 "- MLP: neuron n ∈ cluster k ⇔ > 0.85 of Var(post_n) lies in span{1, cos/sin(w_k a), cos/sin(w_k b), products}; "
                 "|cluster_k| = " + ", ".join(f"{k}: {n}" for k, n in clusters.items()) + f"; unclustered {spec['neurons_unclustered']}.",
                 "- MLP: pre_n ≈ bias_n + A_n[cos(w_k a + φ_n) + cos(w_k b + φ_n)] (same amplitude and phase for a and b); "
                 "ReLU(pre_n) ∋ cos(w_k a)cos(w_k b), sin(w_k a)sin(w_k b), sin(w_k a)cos(w_k b), cos(w_k a)sin(w_k b); φ_n varies across n.",
                 "- MLP output ∋ cos(w_k(a+b)), sin(w_k(a+b)) via cos(x+y) = cos x cos y − sin x sin y, sin(x+y) = sin x cos y + cos x sin y.",
                 "- Read-out coefficient (FVE): " + "; ".join(f"k={k} {c}: {v} ({f})" for k, c, v, f in readoff) + ".",
                 f"- logit(a,b,c) ≈ Σ_k α_k cos(w_k(a+b−c)) + β, FVE {round(spec['logit_fit_fve'], 3)}; α = "
                 + ", ".join(f"α_{k} = {v}" for k, v in amps.items()) + f"; β = {round(spec['logit_offset'], 4)}.",
                 f"- ‖W_L − proj_K W_L‖² / ‖W_L‖² = {round(100 * spec['wl_residual_sq_frac'], 2)}%; mean CE over all inputs "
                 f"= {spec['loss_all_inputs']:.3g}."]
        return "\n".join(lines) + "\n", {}
    if fmt == "code":
        body = {"algorithm": "Fourier multiplication (clock): embed cos/sin(w_k t); attention moves a and b to '='; "
                             "MLP neuron of frequency k: pre ≈ bias + A*(cos(w_k*a + phi) + cos(w_k*b + phi)), same A and "
                             "phi for a and b, phi varies across neurons; ReLU of this sum contains the products that give "
                             "cos/sin(w_k(a+b)) by trig identities; logits ≈ Σ α_k cos(w_k(a+b−c))",
                "w_k": "2*pi*k/113", "key_freqs": K, "logit_amplitudes_alpha": amps, "logit_offset_beta": round(spec["logit_offset"], 4),
                "logit_fit_fve": round(spec["logit_fit_fve"], 3), "mean_cross_entropy_all_inputs": spec["loss_all_inputs"],
                "attention_heads": heads, "attention_formula": "A_to_a ≈ 0.5 + alpha*(cos(w*a)-cos(w*b)) + beta*(sin(w*a)-sin(w*b)); A_to_b ≈ 1 - A_to_a",
                "neurons_per_freq": clusters, "neurons_unclustered": spec["neurons_unclustered"],
                "readoff_coef_unit_dir_and_fve": readoff, "WL_sq_norm_outside_key_dirs": round(spec["wl_residual_sq_frac"], 4)}
        import pprint
        return "# Explanation (as a Python data structure)\n\n```python\nEXPLANATION = " + pprint.pformat(body, width=110, sort_dicts=False) + "\n```\n", {}
    raise ValueError(fmt)


def length_stats(text: str, extra: dict[str, str]) -> dict:
    """Description length of an explanation: characters, gzip bytes and count of numbers (all files together)."""
    full = text + "".join(extra.values())
    return {"chars": len(full), "gzip_bytes": len(gzip.compress(full.encode("utf-8"), mtime=0)),
            "numbers": len(re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", full))}
