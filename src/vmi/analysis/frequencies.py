"""Which frequencies does each replacement actually use? (post-hoc contamination analysis, rounds 1–2)

Local explanation: applies the same key-frequency rule used for the targets (Fourier norms of W_L = W_U·W_out ≥ τ·max,
grokviz `key_frequencies`, τ = 0.25) to every agent-built replacement, and compares the set with the target's key
frequencies and with the published frequencies of Nanda et al.'s checkpoint {14, 35, 41, 42, 52}. If a simulator uses
the paper's set when the explanation gives no frequencies (E0, E1), its score on the published checkpoint measures
memory, not the explanation.
Audit 2026-11-22 (second pass) additions, all post hoc:
- `paper_norm_share` / `target_norm_share`: fraction of W_L's squared Fourier norm on the paper's / the target's
  frequencies — a continuous measure that does not depend on the 0.25 threshold (which misses frequencies an agent
  built at lower amplitude, e.g. a script listing all five paper frequencies but detected as {42, 52}).
- `declared_freqs`: the integer list the agent's own `build_weights.py` assigns to a frequency variable (e.g.
  `KEY_FREQS = [14, 35, 41, 42, 52]`), folded into 1..56 (k and 113 − k are the same Fourier frequency); None when the
  script chooses frequencies by computation instead of a literal list.
- E0 and E1 are byte-identical on every target and the sandbox carries no target identity, so their replacements on T0
  and on T-paper are replicates of one condition: every E0/E1 replacement is graded against both T0 and the published
  checkpoint (`hs_vs_T0`, `hs_vs_paper`; one of the two is the run's own grade).
Global context: motivated by a run in the seed stage (T-paper × E1 scored like a true explanation); labelled post hoc
everywhere. Written to results/frequencies_round{r}.json and read by `tables.frequencies_table` and `tables.key_results`.
Source: paper frequencies — Nanda et al. 2023, Sec. 4.1 (https://arxiv.org/abs/2301.05217); W_L and its Fourier
norms — grokviz `metrics.wl_fourier_norms` / `fourier.frequency_norms`.
"""

from __future__ import annotations  # postponed annotations

import json  # results
import re  # literal frequency lists in build scripts
from pathlib import Path  # paths

from vmi.config import P, REPO_ROOT, Settings  # modulus, paths

RES = REPO_ROOT / "results"
PAPER_FREQS = [14, 35, 41, 42, 52]
# `NAME = [ints]` where NAME is a frequency variable (contains "freq", or is KEYS / KS / KF / KEY_*), e.g.
# "KEY_FREQS = [14, 35, 41, 42, 52]" or "KEYS = [14, 35, 41, 52, 73]"; whole-name match, so "blocks"/"masks" never match
DECLARED = re.compile(r"^\s*(\w*freq\w*|keys?|ks|kf|key_\w+)\s*=\s*\[\s*(\d+(?:\s*,\s*\d+)*)\s*,?\s*\]", re.M | re.I)
# names that hold something per frequency (neuron counts, a head's frequency), not the frequency set itself
NOT_A_SET = re.compile(r"^n_|per|head|count|num|w_", re.I)


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def fold(k: int, p: int = P) -> int:
    """Fourier frequency k and p − k are the same cos/sin pair (cos(2π(p−k)t/p) = cos(2πkt/p)); return the one in 1..(p−1)/2."""
    k %= p
    return min(k, p - k)


def declared_frequencies(script: Path) -> list[int] | None:
    """Sorted distinct frequencies (folded to 1..56) of the first literal integer list the script assigns to a
    frequency-named variable; None if there is no such list (frequencies computed, e.g. by a search or a range)."""
    if not script.exists():
        return None
    for m in DECLARED.finditer(script.read_text(encoding="utf-8", errors="replace")):
        if NOT_A_SET.search(m.group(1)):
            continue  # e.g. N_PER_FREQ = [86, 86, 85, …] or HEAD_FREQ = [4, 5]
        ks = sorted({fold(int(x)) for x in m.group(2).split(",")})
        return ks if all(1 <= k <= P // 2 for k in ks) else None
    return None


def norm_share(norms, freqs) -> float:
    """Fraction of Σ_k norm_k² that lies on the given frequencies (norms indexed k − 1)."""
    sq = norms.detach().double() ** 2  # analysis only: no autograd graph needed
    return float(sq[[k - 1 for k in freqs]].sum() / sq.sum()) if float(sq.sum()) > 0 else 0.0


def used_frequencies(settings: Settings, max_round: int) -> dict:
    from grokviz import fourier as gf
    from grokviz import metrics as gm
    from grokviz.fourier import fourier_basis

    from vmi import targets, weights
    from vmi.grader.score import grade_npz
    F, _ = fourier_basis(113)
    swap = {}  # targets for the E0/E1 cross-grading, loaded once on first use
    rows = []
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        if r.get("stage") == "dev" or int(r.get("round", 0)) > max_round or r.get("failed") or not npz.exists():
            continue
        model = weights.to_model(weights.load_npz(Path(npz))).double()
        used = gm.key_frequencies(model, F)
        norms = gf.frequency_norms(gm.wl_fourier_norms(model, F))  # [(p-1)/2], index k-1 ↔ frequency k
        tgt = json.loads((RES / "explanations" / r["target"] / "analysis.json").read_text())["key_freqs"]
        row = {"run_id": r["run_id"], "target": r["target"], "level": r["level"], "variant": r.get("variant", ""),
               "model": r["model"], "used_freqs": used, "target_freqs": tgt, "jaccard_target": _jaccard(used, tgt),
               "jaccard_paper": _jaccard(used, PAPER_FREQS), "hs": r["hs"],
               "paper_norm_share": norm_share(norms, PAPER_FREQS), "target_norm_share": norm_share(norms, tgt),
               "declared_freqs": declared_frequencies(settings.runs_dir / r["run_id"] / "work_out" / "build_weights.py")}
        if r["level"] in ("E0", "E1"):
            # Target swap: the SAME replacement graded against T0 and against the published checkpoint. E0/E1 carry
            # no target information, so a replacement built "for" either target is a draw from one condition.
            for t, key in (("T0", "hs_vs_T0"), ("T-paper", "hs_vs_paper")):
                if r["target"] == t:
                    row[key] = r["hs"]
                else:
                    swap[t] = swap.get(t) or targets.load_model(settings, t)
                    row[key] = grade_npz(swap[t], Path(npz), secondary=False)["hs"]
        rows.append(row)
    out = {"round": max_round, "paper_freqs": PAPER_FREQS, "rule": "W_L Fourier norm >= 0.25 max (grokviz key_frequencies)",
           "chance_share": len(PAPER_FREQS) / (P // 2), "rows": rows}
    (RES / f"frequencies_round{max_round}.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out
