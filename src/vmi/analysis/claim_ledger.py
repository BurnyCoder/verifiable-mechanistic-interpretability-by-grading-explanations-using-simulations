"""Index claim-bearing passages and their evidence without rewriting original stimuli or historical testimony."""

import ast
import json
import re
import subprocess
from pathlib import Path

from .scientific_audit import AUDIT, ROOT

PROTOCOL = "https://github.com/BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/blob/801edf013db7239f8994e6780c5873e1890f5103/reports/preregistration.md"
RECON = "results/scientific-audit/reconstruction.json"
SOURCES = "results/scientific-audit/sources.json"
# Sources are evidence pointers, not a declaration that one source proves every sentence of a passage.
TOPICS = (
    (
        r"bootstrap|percentile|coverage|holm|mann.whitney|spearman|rank.biserial|wilcoxon|jonckheere|kendall",
        "statistical design",
        [RECON, PROTOCOL, "src/vmi/analysis/audit_statistics.py"],
    ),
    (
        r"\bH[1-8]b?\b|replication|wrong.target|trajectory|trajectory controls|planted",
        "hypothesis or replication",
        [RECON, PROTOCOL],
    ),
    (
        r"clock|pizza|symmetricity|distance irrelevance|near.tie",
        "algorithm diagnostic",
        ["results/controls/clock_pizza.json", "https://arxiv.org/html/2306.17844"],
    ),
    (
        r"distill|centred.logit|centered.logit|logit FVU|softmax",
        "output fidelity",
        ["results/controls/logit_fvu_T0.json", "results/baselines/T0.json", "https://arxiv.org/pdf/1503.02531"],
    ),
    (
        r"ridge|whiten|cross.validat|stitch|invarian|linear align|fold|correlation|cka|procrustes",
        "representation metric",
        ["src/vmi/grader/metrics.py", "src/vmi/sensitivity.py", "https://proceedings.mlr.press/v97/kornblith19a.html"],
    ),
    (
        r"seed.target|target select|contamin|memori|prior|paper.s frequenc|training seed",
        "target selection and prior exposure",
        ["results/targets_selection.json", "results/frequencies_round2.json", PROTOCOL],
    ),
    (
        r"self.test|sandbox|internet|harness|transcript|build script|claude code",
        "simulator inputs and behaviour",
        ["sandbox_template/TASK.md", "results/agent_behaviour.json", "results/saved_report_inputs.json"],
    ),
    (
        r"local (model|run)|qwen|manifest|cancel|retry|failure|stopp|ablation",
        "run accounting",
        ["results/manifests", "results/runs", RECON],
    ),
    (
        r"explanation length|length.match|character|gzip|neuron table|reference compiler",
        "stimulus and compiler",
        ["results/explanations", "results/compiler", "src/vmi/reference/compile.py"],
    ),
    (
        r"\b(E[0-4]|T0|HS|KL|W-freq|W-alg|W-role|W-scramble)\b|accuracy",
        "graded measurements",
        ["results/runs", "results/compiler", "results/baselines", RECON],
    ),
    (r"cost|spend|budget|pricing", "recorded cost", ["results/claude", PROTOCOL]),
    (
        r"architecture|fourier|grokking|modular addition",
        "task and algorithm",
        ["sandbox_template/ARCHITECTURE.md", "results/targets_selection.json", "https://arxiv.org/pdf/2301.05217"],
    ),
)


def block_evidence(name):
    """Connect generated cells to the actual input family as well as their renderer."""
    if name.startswith("compiler-"):
        return ["results/compiler/" + name.removeprefix("compiler-")]
    if name.startswith("grader-sensitivity") or name.startswith("explainer-sensitivity"):
        return ["results/sensitivity"]
    if name.startswith(("outputs-", "agent-words", "agent-examples")):
        return ["results/saved_report_inputs.json"]
    if name.startswith(("predictions", "hypotheses", "replications", "reconstructed-evidence")):
        return [RECON, PROTOCOL, "src/vmi/analysis/predictions.py"]
    if name.startswith(("scoreboard", "runs", "ablations", "stitching", "kl-calibration")):
        return ["results/runs"]
    return {
        "baselines-T0": ["results/baselines/T0.json", "results/controls/random_init_T0.json"],
        "clock-pizza-T0": ["results/controls/clock_pizza.json"],
        "logit-fvu-T0": ["results/controls/logit_fvu_T0.json"],
        "controls-T0": ["results/controls"],
        "cost": ["results/claude/ledger.jsonl"],
        "targets": ["results/targets_selection.json"],
        "agent-behaviour": ["results/agent_behaviour.json"],
        "frequencies": ["results/frequencies_round1.json", "results/frequencies_round2.json"],
    }.get(name, [RECON, "results/runs"])


def kind(path):
    p = path.as_posix()
    if p.startswith("results/explanations/"):
        return "experimental stimulus or its construction record"
    if (
        p.startswith(
            ("reports/reviews/audit-", "reports/reviews/independent-audit-2026-11-22", "scripts/audit", "results/audit")
        )
        and p != "scripts/audit_sources.py"
    ):
        return "earlier audit or historical evidence, excluded from independent verdicts"
    if p.startswith(
        (
            "reports/preregistration",
            "reports/progress",
            "reports/reviews/round",
            "reports/reviews/citation-check",
            "reports/experiments/learnings",
        )
    ):
        return "historical decisions or observations with dated errata"
    if p.startswith(("reports/next-steps", "reports/plan", "reports/planned-runs")):
        return "research plan, planned outcomes are prospective"
    if p.startswith("sandbox_template/"):
        return "simulator-facing instructions or code, exposure evidence"
    if p.startswith(
        (
            "results/runs/",
            "results/compiler/",
            "results/controls/",
            "results/baselines/",
            "results/sensitivity/",
            "results/training/",
        )
    ):
        return "archived measurement or derived diagnostic, upstream forward values accepted as recorded"
    if p.startswith("results/"):
        return "project data or derived output"
    if p.startswith(("reports/figures/", "docs/data")) or path.suffix == ".pdf":
        return "rendered display, see reporting inputs and generators"
    if path.suffix == ".md" or p == "docs/index.html":
        return "current authored report or documentation"
    if path.suffix in (".py", ".sh"):
        return "implementation or test"
    return "configuration, license, metadata or other technical asset"


def passages(file):
    """Keep local anchors for prose, table rows and software specifications. Group a generated table by its marker."""
    data = file.read_text(encoding="utf-8")
    if file.suffix == ".py":
        for node in ast.walk(ast.parse(data)):
            doc = (
                ast.get_docstring(node, clean=False)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                else None
            )
            if doc:
                yield node.body[0].lineno, getattr(node, "name", "module"), doc
        return
    heading, marker, buf, start, fence = "", "", [], 0, False
    for i, line in enumerate(data.splitlines() + [""], 1):
        begin = re.match(r"<!-- BEGIN ([^>]+?) -->", line)
        if line.startswith("<!-- END") and marker:
            yield start, heading, f"Generated block {marker}. " + (buf[0] if buf else "")
            marker, buf = "", []
        elif begin:
            marker, start, buf = begin[1], i, []
        elif marker:
            buf.append(line)
        elif line.startswith("```"):
            fence = not fence
        elif fence:
            continue
        elif line.startswith("#") or not line.strip() or line.startswith(("|", "- ", "* ")):
            if buf:
                yield start, heading, " ".join(buf)
                buf = []
            if line.startswith("#"):
                heading = line.lstrip("# ")
            elif line.strip() and not re.fullmatch(r"[ |:\-]+", line):
                yield i, heading, line
        else:
            if not buf:
                start = i
            buf.append(line.strip())


def build():
    """Record every tracked or publishable file. Preserve unsupported provenance explicitly at the passage level."""
    paths = sorted(
        set(subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
    )
    index = AUDIT / "claim_index.jsonl"
    files = AUDIT / "file_inventory.json"
    inventory, records = {}, []
    for name in paths:
        p = Path(name)
        fp = ROOT / p
        if fp in (index, files) or not fp.is_file():
            continue
        category = kind(p)
        inventory[name] = category
        if p.suffix not in (".md", ".py") or category.startswith(("earlier audit", "experimental stimulus")):
            continue
        for line, section, claim in passages(fp):
            claim = " ".join(claim.split())
            found = [(tag, e) for pat, tag, e in TOPICS if re.search(pat, claim + " " + section, re.I)]
            citations = re.findall(r"https?://[^\s<>)\]]+", claim)
            pointers = re.findall(r"`((?:results|src|reports|docs|sandbox_template)/[^` ]+)`", claim)
            links = list(dict.fromkeys(citations + pointers + [ref for _, refs in found for ref in refs]))
            if category.startswith(("historical", "research plan", "simulator-facing")):
                status = category
            elif p.suffix == ".py":
                status = "implementation claim, source at this location"
            elif claim.startswith("Generated block"):
                status = "display mapped to reporting generator"
                marker = claim.split(". ", 1)[0].removeprefix("Generated block ").rstrip(".")
                links = list(dict.fromkeys(block_evidence(marker) + ["src/vmi/analysis/tables.py"] + links))
            elif citations:
                status = "source-linked passage, external findings attributed to their authors"
            elif found:
                status = "see matched evidence families and the dated verdict for applicable limits"
            else:
                status = "context or unverified attribution, no independent empirical verdict"
            records.append(
                dict(
                    path=name,
                    line=line,
                    section=section,
                    excerpt=claim[:175],
                    topics=[t for t, _ in found],
                    disposition=status,
                    evidence=links or [name],
                )
            )
    with index.open("w", encoding="utf-8", newline="\n") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    files.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
