"""Export the study's artefacts for the Hugging Face Hub and upload them (`vmi publish`).

Local explanation: `export` assembles one folder with, per graded agent run, the replacement weights (`weights.npz`),
the agent's `build_weights.py`, the readable transcript, the full event stream and wire log (gzipped) and the graded
result; plus the models trained in this project (T-new with all checkpoints, the planted-mechanism snapshots, the
probability-distilled students and the post-hoc logit-matched students) and a model card. Text files are redacted (local user names and home paths) and the whole
folder is scanned for API-key/token patterns and private identifiers before anything is uploaded; a hit aborts.
`upload` pushes the folder to one model repo and adds it, with the target runs repo, to the project's collection.
Global context: called after each stage (CLAUDE.md: push to GitHub and Hugging Face after every stage). The repo
never contains the target models themselves (they live in BurnyCoder/grokking-modular-addition-runs) or the
paper's checkpoint (not redistributed).
Sources: huggingface_hub CLI `hf upload`, `hf repos create`, `hf collections` (https://huggingface.co/docs/huggingface_hub/guides/cli).
"""

from __future__ import annotations  # postponed annotations

import gzip  # compress event streams
import json  # results
import re  # redaction and scanning
import shutil  # copying
import subprocess  # hf CLI
from pathlib import Path  # paths

from vmi.analysis import tables  # run table for the card
from vmi.config import REPO_ROOT, Settings  # paths

RES = REPO_ROOT / "results"
HF_REPO = "BurnyCoder/grading-explanations-by-simulation"
COLLECTION_TITLE = "Grading mechanistic explanations by simulation"
TARGET_RUNS_REPO = "BurnyCoder/grokking-modular-addition-runs"
GITHUB = "https://github.com/BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations"
# local identifiers that appear in sandbox listings and host paths; replaced before export
REDACT = [(re.compile(r"/home/[A-Za-z0-9_.-]+"), "/home/user"),
          (re.compile(r"C:(\\\\|\\|/)Users(\\\\|\\|/)[A-Za-z0-9_.-]+", re.I), lambda m: f"C:{m.group(1)}Users{m.group(2)}user"),
          (re.compile(r"/mnt/c/Users/[A-Za-z0-9_.-]+", re.I), "/mnt/c/Users/user"), (re.compile(r"\bburny burny\b"), "user user")]
FORBIDDEN = [re.compile(r"sk-ant-(?:api|admin)\d\d-[A-Za-z0-9_-]{20,}"), re.compile(r"hf_[A-Za-z0-9]{34}"),
             re.compile(r"x-api-key", re.I), re.compile(r"gmail\.com", re.I), re.compile(r"\bburny\b"), re.compile(r"Users[\\/]+Happy", re.I)]
TEXT_SUFFIXES = {".md", ".py", ".json", ".log", ".txt", ".ndjson", ".jsonc", ".csv"}


def _redact(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def _copy_text(src: Path, dst: Path, compress: bool = False) -> None:
    """Copy a text file with redaction (optionally gzip-compressed)."""
    raw = gzip.decompress(src.read_bytes()) if src.suffix == ".gz" else src.read_bytes()
    out = _redact(raw.decode("utf-8", errors="replace")).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(gzip.compress(out) if compress else out)


def _scan(folder: Path) -> list[str]:
    """Paths (and pattern) of any forbidden content left in the export, including inside .gz files."""
    hits = []
    for p in folder.rglob("*"):
        if not p.is_file() or not (p.suffix in TEXT_SUFFIXES or p.suffix == ".gz"):
            continue
        text = (gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).decode("utf-8", errors="replace")
        hits += [f"{p.relative_to(folder)}: {pat.pattern}" for pat in FORBIDDEN if pat.search(text)]
    return hits


def _card(n_runs: int, n_dev: int = 0) -> str:
    return f"""---
license: mit
tags:
  - mechanistic-interpretability
  - grokking
  - modular-arithmetic
  - interpretability-evaluation
---

# Grading mechanistic explanations by simulation — replacement networks, transcripts and trained models

Artefacts of the study [*Grading mechanistic explanations by weights-blind simulation*]({GITHUB}) by Libor "Burny"
Burian and [Victor Levoso](https://scholar.google.com/citations?user=BBGcEigAAAAJ&hl=en), developed in a mechanistic
interpretability hackathon on automatically evaluating mechanistic interpretability explanations co-organized in the
[mechinterp.com](https://mechinterp.com/) community.

An LLM agent (Claude Haiku 5.5 in Claude Code; local Qwen models were tried and stopped at chance), offline except for
the model endpoint and never shown the target's weights,
reads an explanation of a one-layer modular-addition transformer ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217))
and writes the weights of a replacement network. The replacement is graded against the target by output KL and by
cross-validated hidden-state similarity. Code, preregistration, paper and all graded results: [GitHub]({GITHUB}).
Target models: [{TARGET_RUNS_REPO}](https://huggingface.co/{TARGET_RUNS_REPO}).

## Contents

| Folder | What |
|---|---|
| `replacements/<run_id>/` | one folder per graded agent run ({n_runs} so far, of which {n_dev} are stage-0 development runs on the dev target, never reported as results): `weights.npz` (the replacement, 11 float64 arrays in the paper's layout), `build_weights.py` (the agent's script), `transcript.md`, `events.ndjson.gz` (full event stream), `wire.log.gz` (every request and response body between the sandbox and the model; no headers), `result.json` (grader output) |
| `compiler/<target>/<explanation>.npz` | the agent-free reference compiler's build for every explanation of every target |
| `trained/T-new/` | the held-out target trained after the protocol freeze (paper recipe, seed 11): all 401 checkpoints, curves, metadata |
| `trained/planted_T0/` | the reference-compiled E3 model for T0 and its snapshots after 100, 1,000 and 10,000 further training steps |
| `trained/students_T0/` | three students distilled on T0's output probabilities (KL at temperature 1, 20,000 steps each) |
| `trained/students_logit_T0/` | three students trained to match T0's centred logits (squared error, 20,000 steps each; post hoc, audit 9) |

Run IDs read `stage__target__explanation[-variant]__simulator__temperature__seed`. Local user names and home paths in
logs are replaced by `user`. Load a replacement with `numpy.load("weights.npz")`; the array names and shapes are in the
repository's `sandbox_template/ARCHITECTURE.md`.

## Runs

{tables.runs_table(99)}
"""


def export(settings: Settings) -> Path:
    """Assemble the upload folder under VMI_DATA_DIR/../hf_export; raises if the scan finds anything forbidden."""
    out = settings.data_dir.parent / "hf_export"
    if out.exists():
        shutil.rmtree(out)
    n = n_dev = 0
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        run = settings.runs_dir / r["run_id"]
        d = out / "replacements" / r["run_id"]
        d.mkdir(parents=True, exist_ok=True)
        _copy_text(p, d / "result.json")
        if (run / "work_out" / "weights.npz").exists():
            shutil.copy2(run / "work_out" / "weights.npz", d / "weights.npz")
        for src, name, gz in ((run / "work_out" / "build_weights.py", "build_weights.py", False),
                              (run / "logs" / "transcript.md", "transcript.md", False),
                              (run / "logs" / "events.ndjson", "events.ndjson.gz", True),
                              (run / "logs" / "wire.log.gz", "wire.log.gz", True)):
            if src.exists():
                _copy_text(src, d / name, compress=gz)
        n += 1
        n_dev += r.get("stage") == "dev"
    comp = settings.runs_dir / "_compiler"  # reference-compiler builds, one npz per target and explanation
    if comp.exists():
        shutil.copytree(comp, out / "compiler")
    tr = settings.data_dir / "trained"
    for src, dst in ((tr / "new_s11", out / "trained" / "T-new"), (tr / "planted_T0", out / "trained" / "planted_T0")):
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optim.safetensors"))  # optimiser state not needed
    for kind, folder in (("student", "students_T0"), ("student_logit", "students_logit_T0")):  # audit 9: both student kinds
        for s in (1000, 1001, 1002):
            f = tr / f"{kind}_T0_s{s}" / "step_020000.npz"
            if f.exists():
                (out / "trained" / folder).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / "trained" / folder / f"student_s{s}.npz")
    for meta in (out / "trained").rglob("meta.json"):
        meta.write_text(_redact(meta.read_text(encoding="utf-8")), encoding="utf-8")
    (out / "README.md").write_text(_card(n, n_dev), encoding="utf-8", newline="\n")
    hits = _scan(out)
    if hits:
        raise RuntimeError("forbidden content in the export, nothing uploaded:\n" + "\n".join(hits[:20]))
    return out


def upload(folder: Path, message: str) -> str:
    """Create/update the public model repo and the project collection; returns the collection slug."""
    subprocess.run(["hf", "repos", "create", HF_REPO, "--type", "model", "--exist-ok"], check=True)
    subprocess.run(["hf", "upload", HF_REPO, str(folder), ".", "--type", "model", "--commit-message", message], check=True)
    res = subprocess.run(["hf", "collections", "create", COLLECTION_TITLE, "--exists-ok", "--format", "json",
                          "--description", "Weights-blind simulation as a score for mechanistic explanations: replacement "
                          "networks, transcripts, trained models, and the target runs."], check=True, capture_output=True, text=True)
    slug = json.loads(res.stdout)["slug"] if res.stdout.strip().startswith("{") else res.stdout.strip().splitlines()[-1]
    for item, note in ((HF_REPO, "Replacement networks, agent transcripts, trained models (T-new, planted, students)"),
                       (TARGET_RUNS_REPO, "Target models: 22 one-layer modular-addition transformers, paper recipe")):
        subprocess.run(["hf", "collections", "add-item", slug, item, "model", "--note", note, "--exists-ok"], check=True)
    return slug
