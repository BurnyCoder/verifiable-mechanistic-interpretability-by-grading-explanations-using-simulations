"""Audit 5 (2026-11-22, post hoc): independent checks behind the fifth audit's corrections.

Local explanation: computes, from the targets' weights and the stored results, every number the fifth audit adds to
the paper and README, so none of them is typed by hand:
  (1) per target, the per-frequency Fourier norms of the embedding W_E and of the neuron-logit map W_L = W_U·W_out
      (each relative to its maximum), the key-frequency set (W_L ≥ τ·max, grokviz's rule), the embedding's own
      above-threshold set, the neurons clustered at each key frequency, and every frequency that is in one set but not
      the other together with any small-harmonic relation h·k ≡ ±f (mod 113), h ∈ {2, 3}, to a key frequency k;
      for T0's frequency 2, the share of W_L's k = 2 energy written by neurons of each cluster;
  (2) per target, the reference compiler's HS gaps E4 − W-scramble and E3 − W-alg, with per-site values on T0;
  (3) the coverage of the nominal 95 % percentile-bootstrap interval with three runs per arm (Monte-Carlo, normal
      data), for a mean and for a difference of two means;
  (4) the nine independent retrains of the H4 baseline and the twelve wrong targets of H7, as stored;
  (5) the smallest formation-phase H8 correlation stored by audit 4;
  (6) the preregistered target-selection rules applied to the published key frequencies (`results/targets_selection.json`);
  (7) a frequency-only comparison: HS against the secondary embedding-spectrum overlap over the T0 conditions.
Global context: post hoc; no preregistered verdict uses these numbers. Output `results/audit5/checks.json`, read by
`vmi.analysis.tables.inline_values` (v: spans) and cited by `reports/reviews/audit-5-2026-11-22.md`.
Sources: key-frequency rule — grokviz.metrics.key_frequencies (τ = 0.25), after Nanda et al. 2023 App. C.2
(https://arxiv.org/abs/2301.05217); neuron clusters — results/explanations/<target>/analysis.json; percentile
bootstrap — Efron & Tibshirani 1993, ch. 13; small-sample narrowness — Hesterberg 2015 (https://arxiv.org/abs/1411.5279);
numpy FFT-free projection on cos/sin(2πkt/p) (https://numpy.org/doc/stable/reference/routines.linalg.html).
Run: `.venv/Scripts/python.exe scripts/audit5/checks.py` (CPU, < 1 minute).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # repo import path
import time  # timestamped log
from pathlib import Path  # paths

import numpy as np  # linear algebra and Monte-Carlo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vmi import targets  # noqa: E402  target loading (weights live outside the repository)
from vmi.config import P, load  # noqa: E402  modulus and paths

RES = ROOT / "results"
OUT = RES / "audit5" / "checks.json"
TAU = 0.25  # grokviz.metrics.KEY_FREQ_TAU: a frequency is "key" if its norm is at least 0.25 × the maximum
TARGETS = ["T0", "T1", "T2", "T-new", "T-paper", "dev"]


def log(msg: str) -> None:
    """Timestamped progress line on the terminal."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def freq_norms(M: np.ndarray) -> np.ndarray:
    """Norm of the cos and sin components of M along its first (token or class) axis for k = 1 … 56.

    Local explanation: projects every column of M (length p) on cos(2πkt/p) and sin(2πkt/p) and combines both
    components over all columns; frequencies k and p − k coincide, so k = 1 … (p − 1)/2 covers the spectrum."""
    t = np.arange(P)
    out = []
    for k in range(1, (P - 1) // 2 + 1):
        c, s = np.cos(2 * np.pi * k * t / P), np.sin(2 * np.pi * k * t / P)
        out.append(np.sqrt(((c @ M) ** 2).sum() + ((s @ M) ** 2).sum()))
    return np.array(out)


def harmonics(f: int, keys: list[int]) -> list[str]:
    """Key frequencies k with h·k ≡ ±f (mod p) for h = 2 or 3, written '+hxk' (h·k ≡ f) or '-hxk' (h·k ≡ −f)."""
    return [f"{'+' if (h * k) % P == f else '-'}{h}x{k}" for h in (2, 3) for k in keys if (h * k) % P in (f, P - f)]


def spectra(settings) -> dict:
    """(1): embedding vs neuron-logit-map frequencies per target."""
    out = {}
    for name in TARGETS:
        sd = {k: v.double().numpy() for k, v in targets.load_model(settings, name).state_dict().items()}
        WE = sd["embed.W_E"][:, :P].T  # [p tokens, d_model]
        WL = sd["unembed.W_U"][:, :P].T @ sd["blocks.0.mlp.W_out"]  # [p classes, d_mlp] = (W_U·W_out) read per class
        ne, nl = freq_norms(WE), freq_norms(WL)
        ne, nl = ne / ne.max(), nl / nl.max()
        a = json.loads((RES / "explanations" / name / "analysis.json").read_text(encoding="utf-8"))
        keys = a["key_freqs"]
        emb = [k + 1 for k in np.nonzero(ne >= TAU)[0].tolist()]
        rec = {"key_freqs_WL": keys, "embedding_freqs_WE": emb,
               "recomputed_key_freqs_WL": [k + 1 for k in np.nonzero(nl >= TAU)[0].tolist()],
               "stored_embedding_freqs": a["embedding_freqs"],
               "neurons_per_key_freq": {k: len(v) if isinstance(v, list) else v for k, v in a["neuron_clusters"].items()},
               "WL_rel": {k: round(float(nl[k - 1]), 4) for k in sorted(set(keys) | set(emb))},
               "WE_rel": {k: round(float(ne[k - 1]), 4) for k in sorted(set(keys) | set(emb))},
               "WL_next_below_tau": round(float(np.sort(nl[nl < TAU])[-1]), 4),
               "key_not_in_embedding": {f: harmonics(f, [k for k in keys if k != f]) for f in keys if f not in emb},
               "embedding_not_key": {f: harmonics(f, keys) for f in emb if f not in keys}}
        if name == "T0":  # who writes W_L's k = 2 component: energy share per neuron cluster
            t = np.arange(P)
            c, s = np.cos(2 * np.pi * 2 * t / P), np.sin(2 * np.pi * 2 * t / P)
            e = (c @ WL) ** 2 + (s @ WL) ** 2  # per-neuron energy at k = 2
            by = {}
            for n in a["neurons"]:
                by[str(n["freq"])] = by.get(str(n["freq"]), 0.0) + float(e[n["neuron"]])
            rec["k2_energy_share_by_neuron_freq"] = {k: round(v / float(e.sum()), 4) for k, v in
                                                     sorted(by.items(), key=lambda kv: -kv[1])}
        out[name] = rec
        log(f"{name}: keys {keys}, embedding {emb}, mismatches {rec['key_not_in_embedding']} / {rec['embedding_not_key']}")
    return out


def compiler_gaps() -> dict:
    """(2): literal compiler HS gaps between true and wrong explanations per target."""
    out = {}
    for name in ["T0", "T1", "T2", "T-new", "T-paper"]:
        c = {lv: json.loads((RES / "compiler" / name / f"{lv}.json").read_text()) for lv in ("E3", "E4", "W-alg", "W-scramble")}
        out[name] = {"E4_minus_Wscramble": c["E4"]["hs"] - c["W-scramble"]["hs"], "E3_minus_Walg": c["E3"]["hs"] - c["W-alg"]["hs"]}
        if name == "T0":
            out[name]["sites"] = {lv: {s: v["harmonic"] for s, v in c[lv]["sites"].items()} for lv in ("E4", "W-scramble")}
    log(f"compiler gaps: { {k: round(v['E4_minus_Wscramble'], 3) for k, v in out.items()} }")
    return out


def bootstrap_coverage(reps: int = 4000, boots: int = 2000, n: int = 3, seed: int = 1) -> dict:
    """(3): true coverage of the nominal 95 % percentile-bootstrap interval with n runs per arm (normal data, true
    difference 0). Analytic value for the mean: the interval is the runs' range, which misses the median exactly when
    all n runs fall on one side of it, probability 2·(1/2)^n (= 0.25 for n = 3)."""
    rng = np.random.default_rng(seed)
    cm = cd = 0
    for _ in range(reps):
        x, y = rng.normal(size=n), rng.normal(size=n)
        bx, by = rng.choice(x, (boots, n)).mean(1), rng.choice(y, (boots, n)).mean(1)
        lo, hi = np.percentile(bx, [2.5, 97.5])
        cm += lo <= 0 <= hi
        lo, hi = np.percentile(bx - by, [2.5, 97.5])
        cd += lo <= 0 <= hi
    out = {"n_per_arm": n, "reps": reps, "boots": boots, "seed": seed, "coverage_mean": cm / reps,
           "coverage_difference": cd / reps, "analytic_coverage_mean_range": 1 - 2 * 0.5 ** n}
    log(f"bootstrap coverage n=3: mean {out['coverage_mean']:.3f}, difference {out['coverage_difference']:.3f}")
    return out


def design_sets() -> dict:
    """(4), (5): stored retrain list, wrong-target list and the smallest formation-phase H8 correlation."""
    base = json.loads((RES / "baselines" / "T0.json").read_text())
    wt = json.loads((RES / "controls" / "wrong_targets_T0.json").read_text())
    h8 = json.loads((RES / "audit4" / "h8_positive_control.json").read_text())
    form = h8["phases"]["rho_formation"]
    with_freqs = {k: v for k, v in form.items() if "retrain" not in k}
    return {"retrains_T0": [r["run"] for r in base["retrains"]],
            "wrong_targets_T0": list(wt["rows"][0]["hs_wrong"].keys()),
            "h8_formation_min_rho_with_T0_freqs": min(with_freqs.values()),
            "h8_formation_min_rho_network": min(with_freqs, key=with_freqs.get)}


PAPER_FREQS = [14, 35, 41, 42, 52]  # Nanda et al. 2023 §4.1 key frequencies of the published model


def target_selection(settings) -> dict:
    """(6): the preregistered target rules (§2) applied to the published key frequencies of the eleven wd1 seeds.

    Local explanation: reads `wd1_s*/analysis/key_freqs.json` from the downloaded target repository (published with the
    runs), then applies §2: T0 = the first seed whose key-frequency set differs from the paper's; T1 / T2 = the seed
    with the fewest / most key frequencies among the others (ties → lower seed); dev = wd1_s10 is fixed by §2."""
    pub = {}
    for s in range(11):
        f = targets.run_dir(settings, f"wd1_s{s}") / "analysis" / "key_freqs.json"
        pub[f"wd1_s{s}"] = json.loads(f.read_text())["key_freqs"]
    pool = [f"wd1_s{s}" for s in range(10)]  # wd1_s10 is the dev target (§2)
    t0 = next(r for r in pool if sorted(pub[r]) != PAPER_FREQS)
    rest = [r for r in pool if r != t0]
    n = {r: len(pub[r]) for r in rest}
    t1 = min(rest, key=lambda r: (n[r], int(r.split("_s")[1])))
    t2 = max(rest, key=lambda r: (n[r], -int(r.split("_s")[1])))
    ties = {"fewest": [r for r in rest if n[r] == n[t1]], "most": [r for r in rest if n[r] == n[t2]]}
    out = {"source": "published analysis/key_freqs.json of BurnyCoder/grokking-modular-addition-runs (revision in .env.example)",
           "paper_freqs": PAPER_FREQS, "key_freqs": pub, "rule_T0": t0, "rule_T1": t1, "rule_T2": t2, "ties": ties,
           "dev": "wd1_s10"}
    log(f"target rules: T0 {t0}, T1 {t1} (ties {ties['fewest']}), T2 {t2} (ties {ties['most']})")
    return out


def frequency_only_baseline() -> dict:
    """(7): does a frequency-overlap score already rank the T0 conditions like HS? Compares, per explanation, the mean
    HS of the main simulator's runs with the mean of the preregistered secondary `embedding_spectrum_cosine` (cosine of
    the per-frequency norms of the two embeddings), and Kendall's τ between the two orderings."""
    from scipy.stats import kendalltau  # rank agreement of two orderings
    levels = ["E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble"]
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]
    rows = {}
    for lv in levels:
        rs = [r for r in runs if r["target"] == "T0" and r["level"] == lv and r["model"] == "claude-haiku-5-5"
              and r.get("variant", "") == "" and r["stage"] != "dev" and int(r.get("round", 0)) <= 2]
        rows[lv] = {"hs": float(np.mean([r["hs"] for r in rs])),
                    "embedding_spectrum_cosine": float(np.mean([r["embedding_spectrum_cosine"] for r in rs]))}
    tau = kendalltau([rows[lv]["hs"] for lv in levels], [rows[lv]["embedding_spectrum_cosine"] for lv in levels])
    log(f"HS vs embedding-spectrum overlap over the T0 conditions: Kendall τ {tau.statistic:.2f}")
    return {"by_level": rows, "kendall_tau": float(tau.statistic)}


def main() -> None:
    settings = load()
    out = {"note": "Audit 5 (2026-11-22), post hoc; no preregistered verdict uses these numbers.",
           "spectra": spectra(settings), "compiler_gaps": compiler_gaps(), "bootstrap": bootstrap_coverage(),
           "frequency_only_baseline": frequency_only_baseline(), **design_sets()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    sel = RES / "targets_selection.json"
    sel.write_text(json.dumps(target_selection(settings), indent=2), encoding="utf-8", newline="\n")
    log(f"wrote {OUT.relative_to(ROOT)} and {sel.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
