"""Explanation ladder: turn a target's measured mechanism into specs for E0–E4 and the false controls W-*.

Local explanation: `make_spec(analysis, level)` selects which measured quantities an explanation level reveals
(preregistration §3) and, for the false controls, applies a fixed seeded corruption:
- W-freq: every frequency mapped to a different one (disjoint seeded draw); amplitudes, clusters, heads carried over.
- W-alg: Pizza algorithm (Zhong et al. 2023) at the true frequencies: uniform attention, |cos(w(a−b)/2)| logits.
- W-role: embedding/attention use the true frequencies, MLP and unembedding a disjoint set.
- W-scramble: E4 table with each neuron's frequency redrawn from the key set and all its phases redrawn.
The spec is the single source for both the rendered text (`render.py`) and the reference compiler.
Global context: because text and compiler input come from the same spec, the compiler's score is exactly "what the
explanation determines", and the agent sees the same information in prose.
Sources: preregistration §3; Pizza algorithm — https://arxiv.org/abs/2306.17844.
"""

from __future__ import annotations  # postponed annotations

import copy  # deep copies before corrupting
import zlib  # stable seeds from strings

import numpy as np  # seeded draws

LEVELS = ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble")


def _seed(*parts) -> int:
    """Deterministic seed from labels (e.g. level and target name); crc32 is stable across runs and machines."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def _other_freqs(true: list[int], seed: int) -> list[int]:
    """|true| distinct frequencies from 1..56 avoiding the true ones (fixed seeded draw)."""
    pool = [k for k in range(1, 57) if k not in true]
    return sorted(np.random.default_rng(seed).choice(pool, size=len(true), replace=False).tolist())


def _remap(spec: dict, mapping: dict[int, int], fields: tuple[str, ...]) -> None:
    """Rename frequency keys/values in the given fields according to mapping (in place)."""
    for f in fields:
        if f in ("embed_freqs", "mlp_freqs") and spec.get(f):
            spec[f] = [mapping[k] for k in spec[f]]
        elif f in ("logit_amplitudes", "neuron_clusters", "embedding_norm_by_freq") and spec.get(f):
            spec[f] = {mapping[int(k)]: v for k, v in spec[f].items()}
        elif f == "heads" and spec.get(f):
            for h in spec[f]:
                h["freq"] = mapping.get(int(h["freq"]), int(h["freq"]))
                if "ov_norm_by_freq" in h:
                    h["ov_norm_by_freq"] = {mapping.get(int(k), int(k)): v for k, v in h["ov_norm_by_freq"].items()}
        elif f == "readoff" and spec.get(f):
            for r in spec[f]:
                r["freq"] = mapping.get(int(r["freq"]), int(r["freq"]))


def make_spec(an: dict, level: str, target: str = "") -> dict:
    """Spec of one explanation level for one target's analysis dict `an`."""
    K = list(an["key_freqs"])
    s: dict = {"level": level, "algorithm": "clock", "n_key": len(K)}
    if level == "E0":
        return {"level": "E0", "algorithm": "none"}
    if level == "E1":
        return s
    s.update({"embed_freqs": K, "mlp_freqs": K, "loss_all_inputs": an["loss_all_inputs"]})
    if level == "E2":
        return s
    s.update({
        "logit_amplitudes": dict(an["logit_amplitudes"]),
        "logit_offset": an["logit_offset"],
        "logit_fit_fve": an["logit_fit_fve"],
        "logit_std": an["logit_std"],
        "mean_correct_logprob": an["mean_correct_logprob"],
        "embedding_norm_by_freq": dict(an["embedding_norm_by_freq"]),
        "neuron_clusters": dict(an["neuron_clusters"]),
        "neurons_unclustered": an["neurons_unclustered"],
        "neuron_fve_threshold": an["neuron_fve_threshold"],
        "heads": [{k: h[k] for k in ("head", "freq", "alpha", "beta", "pattern_fve", "attn_to_eq")} for h in an["heads"]],
        "readoff": [dict(r) for r in an["readoff"]],
        "wl_residual_sq_frac": an["wl_residual_sq_frac"],
    })
    if level == "E4":
        s["heads"] = [dict(h) for h in an["heads"]]
        s["neurons"] = [dict(r) for r in an["neurons"]]
        return s
    if level == "E3":
        return s
    if level == "W-freq":
        new = _other_freqs(K, _seed("W-freq", target))
        _remap(s, dict(zip(K, new)), ("embed_freqs", "mlp_freqs", "logit_amplitudes", "neuron_clusters",
                                      "embedding_norm_by_freq", "heads", "readoff"))
        return s
    if level == "W-alg":
        s["algorithm"] = "pizza"
        s["heads"] = None  # Pizza: every head attends equally to a and b
        s.pop("readoff", None)
        s.pop("wl_residual_sq_frac", None)
        return s
    if level == "W-role":
        new = _other_freqs(K, _seed("W-role", target))
        mapping = dict(zip(K, new))
        _remap(s, mapping, ("mlp_freqs", "logit_amplitudes", "neuron_clusters", "readoff"))
        return s
    if level == "W-scramble":
        s = make_spec(an, "E4", target)
        s["level"] = "W-scramble"
        rng = np.random.default_rng(_seed("W-scramble", target))
        for r in s["neurons"]:
            r["freq"] = int(rng.choice(K))
            for ph in ("a_phase", "b_phase", "sum_phase", "diff_phase", "out_phase"):
                r[ph] = float(rng.uniform(0, 360))
        return s
    raise ValueError(f"unknown level {level}")


def spec_copy(spec: dict) -> dict:
    """Deep copy (specs are nested dicts/lists)."""
    return copy.deepcopy(spec)
