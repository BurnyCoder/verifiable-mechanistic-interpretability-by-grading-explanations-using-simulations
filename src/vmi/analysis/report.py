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
        "reports/experiments/00-stage0-dev/experiment.md"]  # always show the latest analysed round
ROUND_DOCS = {1: "reports/experiments/01-round1/experiment.md", 2: "reports/experiments/02-round2/experiment.md",
              3: "reports/experiments/03-round3/experiment.md"}  # a round's own report only ever gets that round's numbers
TARGETS = ["T0", "T1", "T2", "T-new", "T-paper", "dev"]


def blocks(max_round: int) -> dict[str, str]:
    """Every generated block, keyed by marker name."""
    out = {"hypotheses": tables.hypotheses_table(max_round), "runs": tables.runs_table(max_round),
           "scoreboard-T0": tables.scoreboard(max_round, "T0"), "baselines-T0": tables.baselines_table("T0"),
           "controls-T0": tables.controls_table("T0", max_round), "cost": tables.cost_table(), "targets": tables.targets_table(),
           "key-results": tables.key_results(max_round),
           "grader-sensitivity-T0": tables.grader_sensitivity_table(max_round, "T0", "agent"),
           "grader-sensitivity-compiler-T0": tables.grader_sensitivity_table(max_round, "T0", "compiler"),
           "explainer-sensitivity-T0": tables.explainer_sensitivity_table("T0"),
           "frequencies": tables.frequencies_table(max_round), "stitching-T0": tables.stitching_table(max_round, "T0"),
           "predictions": tables.predictions_table(max_round), "replications": tables.replications_table(max_round),
           "predictions-plain": tables.predictions_table(max_round, plain=True),
           "ablations-T0": tables.ablations_table(max_round, "T0"), "highlights": tables.highlights_table(max_round),
           "outputs-T0": tables.outputs_table(max_round, "T0"), "agent-examples": tables.agent_examples_table(max_round),
           "agent-words": tables.agent_words_md(max_round)}
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


def inject_values(path: Path, values: dict[str, str]) -> list[str]:
    """Refill every inline value `<!-- v:name -->…<!-- /v -->` in one document (no newlines added, so a number
    quoted inside a sentence is generated like a table); names without a value are left as they are."""
    import re
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    filled = []

    def sub(m):
        if m.group(1) not in values:
            return m.group(0)
        filled.append(m.group(1))
        return f"<!-- v:{m.group(1)} -->{values[m.group(1)]}<!-- /v -->"
    new = re.sub(r"<!-- v:([\w-]+) -->(.*?)<!-- /v -->", sub, text, flags=re.S)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
    return list(dict.fromkeys(filled))


def regenerate(max_round: int) -> dict:
    """Hypotheses, figures and all document tables for one round; returns what was written."""
    from vmi.analysis import frequencies  # post-hoc memorisation check needs the saved replacement weights
    from vmi.config import load
    settings = load()
    if settings.runs_dir.exists():  # both need the saved run folders (weights, event logs)
        frequencies.used_frequencies(settings, max_round)
        from vmi.analysis import agent_examples  # transcript excerpts → reports/agent-examples.md
        agent_examples.write_doc(max_round)
    hyp = hypotheses.write(max_round)
    figs = figures.make_all(max_round)
    gen = blocks(max_round)
    from vmi.analysis.glossary import gloss  # the README spells out what HS, E1, T0 … mean; reports keep compact labels
    readme = {k: gloss(v) for k, v in gen.items()}
    filled = {d: inject_all(REPO_ROOT / d, readme if d == "README.md" else gen) for d in DOCS + [ROUND_DOCS.get(max_round, "")] if d}
    values = tables.inline_values(max_round)  # numbers quoted inside prose sentences
    for d in DOCS:
        filled[d] = filled.get(d, []) + inject_values(REPO_ROOT / d, values)
    from vmi.analysis import site  # interactive GitHub Pages explorer reads docs/data.json
    site.build(max_round)
    summary = {"hypotheses": str(hyp.relative_to(REPO_ROOT)), "figures": [str(p.relative_to(REPO_ROOT)) for p in figs],
               "filled": {d: n for d, n in filled.items() if n}}
    (REPO_ROOT / "results" / f"report_round{max_round}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary


def build_paper() -> Path:
    """reports/paper/paper.md → reports/paper.pdf with pandoc and XeLaTeX (fonts are set in the paper's YAML header)."""
    import subprocess  # pandoc CLI: https://pandoc.org/MANUAL.html
    src = REPO_ROOT / "reports" / "paper"
    out = REPO_ROOT / "reports" / "paper.pdf"
    subprocess.run(["pandoc", "paper.md", "-o", str(out), "--pdf-engine=xelatex", "--resource-path=.:..:../figures"],
                   cwd=src, check=True)
    return out
