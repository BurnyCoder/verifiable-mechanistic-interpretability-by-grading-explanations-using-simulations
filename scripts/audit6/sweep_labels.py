"""Audit 6 (post hoc): read-through labels for "the agent announces trying several values of a constant".

Local explanation: `vmi.analysis.agent_behaviour` counts a run as "announcing a sweep or tuning" when a regular
expression (SWEEP) matches its narration before the closing message. Reading every matched sentence, and every
sentence of the unmatched runs that a broader pattern (BROAD below) picks up, gave three kinds of error: the pattern
matches sentences that set weights from the explanation or only mention tuning, and it misses sweeps phrased as
"try smaller bias scales", "grid the bias" or "the bias scan plateaued". This script records the audit's manual label
for each of the 96 reported Claude runs, with the sentences that support it, so the corrected count is generated from a
results file rather than typed by hand.
Labels: "sweep" = the narration announces trying several values of a constant (a bias, gain, scale, the logit scale,
the frequency set) or reports the result of such a search; "tuning-mention" = it mentions tuning but describes no
search; "false-match" = SWEEP matched a sentence that is not about trying values; "none" = neither.
Global context: replaces the bare regex count in the paper (§3.3) and README; the regex column stays in Appendix D.
Limits: narration only (Claude's thinking is not returned by the API), and the broader pattern can also miss phrasings.
Sources: transcripts `logs/events.ndjson` in each run folder (VMI_RUNS_DIR); the narration parser is
`vmi.analysis.agent_behaviour._claude_events`; Python `re` (https://docs.python.org/3/library/re.html).
Run: uv run python scripts/audit6/sweep_labels.py  →  results/audit6/sweep_labels.json
"""

from __future__ import annotations  # postponed annotations

import json  # transcripts and output
import re  # sentence split and the broad pattern
from pathlib import Path  # paths

from vmi.analysis.agent_behaviour import SWEEP, _claude_events  # the project's pattern and narration parser
from vmi.config import REPO_ROOT, load  # results folder, run folder

OUT = REPO_ROOT / "results" / "audit6" / "sweep_labels.json"
# Broader pattern used only to find candidate sentences in runs SWEEP did not flag (each candidate was then read).
BROAD = re.compile(r"\b(sweep\w*|scan\w*|grid\w*|search\w*|tun(?:e|es|ed|ing)\b|tweak\w*|try (?:a |some |several |other "
                   r"|different )?\w*\s?(?:values?|settings?|scales?|bias\w*)|best (?:value|setting|bias|scale)|optimi[sz]\w*"
                   r"|calibrat\w*)", re.I)
# Manual labels that differ from the default (SWEEP-flagged → "sweep", unflagged → "none"), read on 2026-11-22.
OVERRIDES = {
    # SWEEP matched, but the sentence sets weights from the explanation or only defers a decision
    "seed__T-new__E4__claudehaiku55-claudecode__tdef__s0": "false-match",  # "attention weights tuned by the explanation's α/β"
    "seed__T1__E2__claudehaiku55-claudecode__tdef__s1": "false-match",  # "before deciding whether to tune anything"
    # SWEEP matched "tune/tuning", but no search is described in the narration
    "main__T0__E1__claudehaiku55-claudecode__tdef__s0": "tuning-mention",  # "a bias parameter I can tune"
    "main__T0__E4__claudehaiku55-claudecode__tdef__s0": "tuning-mention",  # "Before tuning, I'll check ..."
    "main__T0__E4__claudehaiku55-claudecode__tdef__s1": "tuning-mention",  # "Before tuning, I'll check ..."
    # SWEEP missed these searches
    "ablations__T0__E3-noselftest__claudehaiku55-claudecode__tdef__s1": "sweep",  # "try smaller bias scales"
    "controls__T0__E0__claudehaiku55-claudecode__tdef__s1": "sweep",  # "on a grid over frequencies, and compare fit quality"
    "controls__T0__W-role__claudehaiku55-claudecode__tdef__s2": "sweep",  # "try a few bias ranges quickly"
    "hyper__T0__W-freq-effort-low__claudehaiku55-claudecode__tdef__s0": "sweep",  # "scan larger gains", "the bias scan plateaued"
    "seed__T-new__E3__claudehaiku55-claudecode__tdef__s0": "sweep",  # "grid the bias B on the ideal features"
    "seed__T1__E3__claudehaiku55-claudecode__tdef__s0": "sweep",  # "try a couple of evenly spaced bias grids"
    "seed__T1__E3__claudehaiku55-claudecode__tdef__s1": "sweep",  # "try negative biases"
    "seed__T1__E3__claudehaiku55-claudecode__tdef__s2": "sweep",  # "try positive biases"
}


def sentences(says: list[tuple]) -> list[str]:
    """Narration sentences before the closing message, whitespace-normalised."""
    out = []
    for _, s in says[:-1]:
        out += [" ".join(x.split()) for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
    return out


def main() -> Path:
    beh = json.loads((REPO_ROOT / "results" / "agent_behaviour.json").read_text(encoding="utf-8"))
    runs_dir = load().runs_dir
    rows = []
    for r in beh["rows"]:
        ev = [json.loads(x) for x in (runs_dir / r["run_id"] / "logs" / "events.ndjson").read_text(encoding="utf-8").splitlines() if x.strip()]
        sents = sentences(_claude_events(ev)[5])
        label = OVERRIDES.get(r["run_id"], "sweep" if r["sweep_said"] else "none")
        evidence = [s[:220] for s in sents if SWEEP.search(s) or BROAD.search(s)][:4]
        rows.append({"run_id": r["run_id"], "level": r["level"], "regex_flagged": r["sweep_said"], "label": label,
                     "evidence": evidence})
    flagged = [x for x in rows if x["regex_flagged"]]
    summary = {"runs": len(rows), "read_sweep": sum(x["label"] == "sweep" for x in rows),
               "regex_flagged": len(flagged),
               "regex_false_match": sum(x["label"] == "false-match" for x in flagged),
               "regex_tuning_mention": sum(x["label"] == "tuning-mention" for x in flagged),
               "regex_missed": sum(x["label"] == "sweep" and not x["regex_flagged"] for x in rows),
               "by_level": {lv: sum(x["label"] == "sweep" for x in rows if x["level"] == lv)
                            for lv in dict.fromkeys(x["level"] for x in rows)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"definition": __doc__.split("Global context:")[0].strip(), "summary": summary, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return OUT


if __name__ == "__main__":
    print(main(), json.loads(OUT.read_text(encoding="utf-8"))["summary"])
