"""Record the software environment of the analysis and of the agent runs (results/environment.json).

Local explanation: package versions from importlib.metadata, Python and platform from the standard library, torch's
CUDA build, the Claude Code version(s) reported in the `system/init` event of every reported Claude run's stream-json
log, and the bubblewrap and pandoc versions from their own `--version` output (None when the tool is not reachable).
Global context: audit 2026-11-22 (second pass) — version claims in README, docs and paper were typed by hand; this file
lets them be checked against a generated record. Written by `vmi analyse` when the run folders are present.
Sources: importlib.metadata (https://docs.python.org/3/library/importlib.metadata.html); Claude Code stream-json init
event (https://code.claude.com/docs/en/headless).
"""

from __future__ import annotations  # postponed annotations

import json  # event logs, output
import platform  # OS and Python
import subprocess  # tool --version
from importlib import metadata  # installed package versions

from vmi.config import REPO_ROOT  # paths

RES = REPO_ROOT / "results"
PACKAGES = ("numpy", "scipy", "torch", "grokviz", "matplotlib", "huggingface_hub")


def _version(cmd: list[str]) -> str | None:
    """First line of `cmd`'s output, or None if the tool is missing or slow."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def claude_code_versions(max_round: int) -> dict[str, int]:
    """Claude Code version → number of reported runs whose init event names it."""
    from vmi.analysis.hypotheses import MAIN_MODEL, load_runs
    from vmi.config import load
    runs_dir, counts = load().runs_dir, {}
    for r in load_runs(max_round):
        log = runs_dir / r["run_id"] / "logs" / "events.ndjson"
        if r["model"] != MAIN_MODEL or not log.exists():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            if '"claude_code_version"' in line:
                v = json.loads(line).get("claude_code_version")
                counts[v] = counts.get(v, 0) + 1
                break
    return counts


def write(max_round: int) -> dict:
    import torch

    from vmi.config import load
    llama = load().llama_server
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": {p: metadata.version(p) for p in PACKAGES},
           "torch_cuda": torch.version.cuda, "claude_code_versions": claude_code_versions(max_round),
           "bubblewrap_wsl": _version(["wsl", "-e", "bwrap", "--version"]),
           "wsl_os": _version(["wsl", "-e", "sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
           "llama_server": _version([str(llama), "--version"]) if llama and llama.exists() else None,
           "pandoc": _version(["pandoc", "--version"])}
    (RES / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8", newline="\n")
    return env
