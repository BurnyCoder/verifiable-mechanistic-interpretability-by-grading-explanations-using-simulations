"""Why did an agent-built network fail? Interventions on the agent's own build script, re-run and graded.

Local explanation: for a run in DIAGNOSES, `diagnose(run_id)` copies the agent's final `build_weights.py` (with the
sandbox's `forward.py` and `selftest.py`) into a temporary folder, applies each listed text edit on its own, runs the
edited script, and grades the resulting weights against the target (task accuracy, HS) — a causal test of which
detail of the agent's construction caused the failure. Results go to results/diagnoses/<run_id>.json, which
`agent_examples.why` reads, so every number in the README's explanation comes from a file this module writes.
The cause in each entry is our reading of the agent's code and transcript; the edits test it.
Global context: `vmi diagnose` (pipeline) runs it; the agent's scripts were audited (numpy only, no training loops)
and are run with a time limit, outside the agent sandbox, only on the analyst's machine.
"""

from __future__ import annotations  # postponed annotations

import json  # results
import shutil  # copy the agent's files
import subprocess  # run the edited build script
import sys  # same interpreter
import tempfile  # scratch folder per variant
from pathlib import Path  # paths

from vmi.config import REPO_ROOT, load  # paths

OUT = REPO_ROOT / "results" / "diagnoses"
LAYOUT = "SEP, BIAS, P0, P1, P2, P3, RO = 10, 11, 12, 22, 42, 52, 52"
DIAGNOSES = {
    "hyper__T0__E3-effort-low__claudehaiku55-claudecode__tdef__s0": {
        "cause": ("a layout bug, not the explanation: its constants line `" + LAYOUT + "` gives head 3's output (P3) "
                  "and the readout (RO) the same residual dimensions 52–61, although its own docstring puts heads 2/3 "
                  "at 42–51, so head 3's attention output is added onto the answer the unembedding reads"),
        "short": "a layout bug: head 3 wrote into the answer dimensions (P3 = RO = 52)",
        "variants": [
            ("as submitted", []),
            ("head 3 moved to free dimensions (P3 = 62), nothing else changed",
             [(LAYOUT, LAYOUT.replace("42, 52, 52", "42, 62, 52"))]),
            ("neuron counts as the explanation states (k=2: 0 neurons)",
             [("NEURONS = {2: 32, 3: 128, 23: 112, 37: 96, 39: 144}", "NEURONS = {2: 0, 3: 261, 23: 65, 37: 56, 39: 130}")]),
            ("its own accuracy-best '=' penalty (1 instead of 10)",
             [("M = 10.0 * np.sqrt(D_HEAD)", "M = 1.0 * np.sqrt(D_HEAD)")]),
        ],
        "blamed": ["ReLU"], "kept": ["faithful"],
    },
}


def _run_variant(src_dir: Path, edits: list[tuple[str, str]], target) -> dict:
    from vmi.grader.score import grade_npz
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for f in ("build_weights.py", "forward.py", "selftest.py"):
            shutil.copy(src_dir / f, d / f)
        script = (d / "build_weights.py").read_text(encoding="utf-8")
        for old, new in edits:
            if old not in script:
                raise ValueError(f"edit target not found: {old[:60]}")
            script = script.replace(old, new)
        (d / "build_weights.py").write_text(script, encoding="utf-8")
        subprocess.run([sys.executable, "build_weights.py"], cwd=d, check=True, capture_output=True, timeout=120)
        g = grade_npz(target, d / "weights.npz", secondary=False)
        return {"accuracy": g["accuracy"], "hs": g["hs"], "kl_out": g["kl_out"]}


def diagnose(run_id: str) -> dict:
    from vmi import targets
    spec = DIAGNOSES[run_id]
    rec = json.loads((REPO_ROOT / "results" / "runs" / f"{run_id}.json").read_text())
    target = targets.load_model(load(), rec["target"])
    src = load().runs_dir / run_id / "work_out"
    out = {"run_id": run_id, "cause": spec["cause"], "short": spec["short"],
           "variants": [{"name": n, **_run_variant(src, e, target)} for n, e in spec["variants"]],
           "blamed": spec["blamed"], "kept": spec["kept"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_id}.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out


def diagnose_all() -> list[dict]:
    return [diagnose(r) for r in DIAGNOSES if (load().runs_dir / r).exists()]


def load_diagnosis(run_id: str) -> dict | None:
    p = OUT / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
