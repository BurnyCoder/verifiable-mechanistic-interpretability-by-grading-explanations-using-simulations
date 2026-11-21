"""One command regenerates every hypothesis verdict, table and figure from results/ (`vmi analyse ROUND`).

Local explanation: writes results/hypotheses_round{r}.json, draws reports/figures/*.png, then replaces every marked
block `<!-- BEGIN name --> … <!-- END name -->` that exists in the documents listed in DOCS with the freshly generated
table, so no number in README, summary, experiment reports or the paper is typed by hand.
Global context: called at the end of every stage; the paper PDF is then built from reports/paper/paper.md.
"""

from __future__ import annotations  # postponed annotations

import json  # result files
from pathlib import Path  # paths

from vmi.analysis import figures, hypotheses, tables  # generators
from vmi.config import REPO_ROOT  # repo paths

DOCS = ["README.md", "reports/summary.md", "reports/experiments/experiments.md", "reports/paper/paper.md",
        "reports/experiments/00-stage0-dev/experiment.md", "reports/experiments/01-round1/experiment.md"]
TARGETS = ["T0", "T1", "T2", "T-new", "T-paper", "dev"]


def blocks(max_round: int) -> dict[str, str]:
    """Every generated block, keyed by marker name."""
    out = {"hypotheses": tables.hypotheses_table(max_round), "runs": tables.runs_table(max_round),
           "scoreboard-T0": tables.scoreboard(max_round, "T0"), "baselines-T0": tables.baselines_table("T0"),
           "controls-T0": tables.controls_table("T0"), "cost": tables.cost_table(), "targets": tables.targets_table(),
           "key-results": tables.key_results(max_round)}
    for t in TARGETS:
        if (REPO_ROOT / "results" / "compiler" / t).exists():
            out[f"compiler-{t}"] = tables.compiler_table(t)
    return out


def inject_all(path: Path, generated: dict[str, str]) -> list[str]:
    """Fill every block whose markers are present in one document; returns the names filled."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    names = [n for n in generated if f"<!-- BEGIN {n} -->" in text]
    for n in names:
        tables.inject(path, n, generated[n])
    return names


def regenerate(max_round: int) -> dict:
    """Hypotheses, figures and all document tables for one round; returns what was written."""
    hyp = hypotheses.write(max_round)
    figs = figures.make_all(max_round)
    gen = blocks(max_round)
    filled = {d: inject_all(REPO_ROOT / d, gen) for d in DOCS}
    summary = {"hypotheses": str(hyp.relative_to(REPO_ROOT)), "figures": [str(p.relative_to(REPO_ROOT)) for p in figs],
               "filled": {d: n for d, n in filled.items() if n}}
    (REPO_ROOT / "results" / f"report_round{max_round}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary
