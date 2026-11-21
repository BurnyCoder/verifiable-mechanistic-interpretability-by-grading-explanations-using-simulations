"""Which frequencies does each replacement actually use? (post-hoc contamination analysis, round 1)

Local explanation: applies the same key-frequency rule used for the targets (Fourier norms of W_L = W_U·W_out ≥ τ·max,
grokviz `key_frequencies`, τ = 0.25) to every agent-built replacement, and compares the set with the target's key
frequencies and with the published frequencies of Nanda et al.'s checkpoint {14, 35, 41, 42, 52}. If a simulator uses
the paper's set when the explanation gives no frequencies (E0, E1), its score on the published checkpoint measures
memory, not the explanation.
Global context: motivated by a run in the seed stage (T-paper × E1 scored like a true explanation); labelled post hoc
everywhere. Written to results/frequencies_round{r}.json and read by `tables.frequencies_table`.
Source: paper frequencies — Nanda et al. 2023, Sec. 4.1 (https://arxiv.org/abs/2301.05217).
"""

from __future__ import annotations  # postponed annotations

import json  # results
from pathlib import Path  # paths

from vmi.config import REPO_ROOT, Settings  # paths

RES = REPO_ROOT / "results"
PAPER_FREQS = [14, 35, 41, 42, 52]


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def used_frequencies(settings: Settings, max_round: int) -> dict:
    from grokviz import metrics as gm
    from grokviz.fourier import fourier_basis

    from vmi import targets, weights
    from vmi.grader.score import grade_npz
    F, _ = fourier_basis(113)
    paper = None  # published checkpoint, loaded only if an E0/E1 replacement of another target needs cross-grading
    rows = []
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        if r.get("stage") == "dev" or int(r.get("round", 0)) > max_round or r.get("failed") or not npz.exists():
            continue
        model = weights.to_model(weights.load_npz(Path(npz))).double()
        used = gm.key_frequencies(model, F)
        tgt = json.loads((RES / "explanations" / r["target"] / "analysis.json").read_text())["key_freqs"]
        row = {"run_id": r["run_id"], "target": r["target"], "level": r["level"], "variant": r.get("variant", ""),
               "model": r["model"], "used_freqs": used, "target_freqs": tgt, "jaccard_target": _jaccard(used, tgt),
               "jaccard_paper": _jaccard(used, PAPER_FREQS), "hs": r["hs"]}
        if r["level"] in ("E0", "E1") and r["target"] != "T-paper":
            # Target swap: the SAME replacement graded against the published checkpoint. If its HS there is far above
            # its HS against its own target, the agent built the published mechanism, not the explained one.
            paper = paper or targets.load_model(settings, "T-paper")
            row["hs_vs_paper"] = grade_npz(paper, Path(npz), secondary=False)["hs"]
        rows.append(row)
    out = {"round": max_round, "paper_freqs": PAPER_FREQS, "rule": "W_L Fourier norm >= 0.25 max (grokviz key_frequencies)",
           "rows": rows}
    (RES / f"frequencies_round{max_round}.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out
