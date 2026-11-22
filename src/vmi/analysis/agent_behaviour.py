"""How did the simulator agents work? Self-test runs and announced parameter sweeps, counted from their transcripts.

Local explanation: for every reported Claude Code run (results/runs, not stage 0) this reads the run's stream-json event
log with `agent_examples._claude_events` and counts (a) how many times the agent ran `selftest.py`, and (b) whether its
narration before the closing message announces a sweep or tuning of a constant (e.g. "I'll sweep the neuron bias",
"before tuning, I'll check …"), matched by SWEEP. Writes results/agent_behaviour.json (per run, per explanation level,
overall) for `tables.agent_behaviour_table` and the paper.
Global context: post-hoc audit item (2026-11-22, second pass). The protocol forbids training loops in the final
`build_weights.py` (static audit, `simulator/audit.py`), but agents may still try constants in scratch copies and keep
the one whose self-test accuracy or least-squares fit is best. Both criteria use only the task labels (a+b) mod 113,
never the target, so no target information leaks; but the score then reflects explanation plus agent repair. Claude's
thinking is not returned by the API (display "omitted"), so only the narration between tool calls is visible: a sweep
the agent ran without announcing it is missed, and an announcement is not proof that a sweep ran.
Source: Claude Code stream-json event format (https://code.claude.com/docs/en/headless).
"""

from __future__ import annotations  # postponed annotations

import json  # event logs, output
import re  # announcement pattern
from collections import defaultdict  # per-level grouping

import numpy as np  # medians

from vmi.analysis.agent_examples import _claude_events  # same parser as the transcript excerpts
from vmi.analysis.hypotheses import MAIN_MODEL, load_runs  # reported runs
from vmi.config import REPO_ROOT, load  # paths

RES = REPO_ROOT / "results"
# "sweep", "grid search", "tune/tuning", "scan over/the", "try a few/several/different values|settings|…";
# "no tuning" / "without tuning" are not counted
SWEEP = re.compile(r"(?<!no )(?<!without )\b(sweep\w*|grid[- ]search\w*|tun(?:e|es|ed|ing)\b|scan(?:s|ned|ning)? (?:over|the)"
                   r"|try(?:ing)? (?:a few|several|different|other|a range of) (?:values|settings|scales|biases|amplitudes|offsets))", re.I)
DEFINITION = ("Self-test runs: commands that run selftest.py with python. Announced sweep or tuning: the agent's narration "
              "before its closing message mentions sweeping, tuning, a grid search, scanning over, or trying several values "
              "of a constant (pattern SWEEP in src/vmi/analysis/agent_behaviour.py); this is stated intent, and Claude's "
              "thinking is not returned by the API. Post hoc, audit 2026-11-22.")


def _stats(rows: list[dict]) -> dict:
    st = [r["selftest_runs"] for r in rows]
    return {"runs": len(rows), "selftest_median": float(np.median(st)), "selftest_min": int(min(st)), "selftest_max": int(max(st)),
            "multi_selftest": sum(x >= 2 for x in st), "sweep_said": sum(r["sweep_said"] for r in rows)}


def analyse(max_round: int) -> dict:
    """Count self-test runs and announced sweeps for every reported Claude run up to max_round; write the JSON file."""
    runs_dir = load().runs_dir
    rows = []
    for r in load_runs(max_round):
        log = runs_dir / r["run_id"] / "logs" / "events.ndjson"
        if r["model"] != MAIN_MODEL or not log.exists():
            continue
        ev = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        calls, _results, _final, _cost, _errors, says = _claude_events(ev)
        said = [s for _, s in says[:-1] if SWEEP.search(s)]  # narration before the closing message
        rows.append({"run_id": r["run_id"], "target": r["target"], "level": r["level"], "variant": r.get("variant", ""),
                     "selftest_runs": sum("selftest" in (c or "") and "python" in (c or "") for c in calls),
                     "tool_calls": len(calls), "sweep_said": bool(said),
                     "sweep_quote": " ".join(said[0].split())[:240] if said else ""})
    by = defaultdict(list)
    for x in rows:
        by[x["level"]].append(x)
    from vmi.analysis.tables import LEVEL_ORDER
    out = {"round": max_round, "definition": DEFINITION, "summary": _stats(rows) if rows else {},
           "by_level": {lv: _stats(by[lv]) for lv in sorted(by, key=lambda lv: LEVEL_ORDER.index(lv) if lv in LEVEL_ORDER else 99)},
           "rows": rows}
    (RES / "agent_behaviour.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out
