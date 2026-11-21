"""Configuration: paths, pinned sources and fixed constants, read from `.env` (overrides) over `.env.example` (defaults).

Local explanation: `load()` returns one frozen `Settings` object. Every tunable path or revision lives in
`.env.example` (tracked, documented) and can be overridden in the private `.env` or the real environment.
Global context: every pipeline phase (targets, explainer, simulator, grader, analysis) receives this object, so the
pinned revisions that the preregistration names are defined in exactly one place.
Sources: python-dotenv `dotenv_values` (https://pypi.org/project/python-dotenv/); dataclasses
(https://docs.python.org/3/library/dataclasses.html).
"""

from __future__ import annotations  # postponed annotations (PEP 563 semantics)

import os  # real environment variables win over files
from dataclasses import dataclass  # immutable settings container
from pathlib import Path  # OS-independent paths

from dotenv import dotenv_values  # parse KEY=VALUE files without touching os.environ

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/vmi/config.py → repository root
P = 113  # prime modulus of the task (Nanda et al. 2023, Sec. 3)
D_VOCAB = P + 1  # 113 numbers plus the '=' token (index 113), as in the paper's code
D_MODEL, N_HEADS, D_HEAD, D_MLP, N_CTX = 128, 4, 32, 512, 3  # paper App. A architecture
FRAC_TRAIN = 0.3  # fraction of the 113² pairs in the training split


@dataclass(frozen=True)
class Settings:
    """All machine-specific paths and pinned sources (see `.env.example` for meaning)."""

    data_dir: Path  # downloaded target runs and the paper checkpoint
    models_dir: Path  # GGUF files for llama-server
    runs_dir: Path  # one folder per agent run
    hf_runs_repo: str  # Hugging Face repo with the 22 trained runs
    hf_runs_revision: str  # pinned commit of that repo
    paper_repo: str  # Hugging Face mirror of the paper checkpoint
    paper_revision: str  # pinned commit of that mirror
    paper_sha256: str  # LFS sha256 of full_run_data.pth (integrity check)
    llama_server: Path  # llama-server.exe
    llm_port: int  # TCP port llama-server listens on (127.0.0.1)
    llm_model: str  # model alias exposed by llama-server
    llm_gguf: Path  # main GGUF file
    llm_context: int  # context window in tokens
    llm_temperature: float  # default sampling temperature (preregistered 0.6)
    llm_n_cpu_moe: int  # MoE layers whose expert weights stay in system RAM (stage-0 tuning)
    llm_reasoning_budget: int  # max thinking tokens per model turn (-1 = unlimited; stage-0 decision)
    run_timeout_s: int  # wall-clock cap per agent run
    run_max_steps: int  # agent step cap per run
    harness: str = "opencode"  # agent harness: "opencode", "mini" (simulator/harness.py) or "claudecode"
    claude_cap_usd: float = 20.0  # total API spend allowed for Claude runs (user decision 2026-11-21)

    @property
    def runs_store(self) -> Path:
        """Folder holding the downloaded `grokking-modular-addition-runs` (one sub-folder per run)."""
        return self.data_dir / "grokking-runs"


def _values() -> dict[str, str]:
    """Merge defaults (.env.example) ← private overrides (.env) ← real environment (VMI_* keys only)."""
    vals = {k: v for k, v in dotenv_values(REPO_ROOT / ".env.example").items() if v is not None}  # documented defaults
    user = REPO_ROOT / ".env"  # private file, may hold secrets this project never reads
    if user.exists():
        vals.update({k: v for k, v in dotenv_values(user).items() if k.startswith("VMI_") and v})  # only our keys
    vals.update({k: v for k, v in os.environ.items() if k.startswith("VMI_")})  # shell wins
    return vals


def load() -> Settings:
    """Build the Settings object; raise KeyError naming any missing key."""
    v = _values()
    return Settings(
        data_dir=Path(v["VMI_DATA_DIR"]),
        models_dir=Path(v["VMI_MODELS_DIR"]),
        runs_dir=Path(v["VMI_RUNS_DIR"]),
        hf_runs_repo=v["VMI_HF_RUNS_REPO"],
        hf_runs_revision=v["VMI_HF_RUNS_REVISION"],
        paper_repo=v["VMI_PAPER_REPO"],
        paper_revision=v["VMI_PAPER_REVISION"],
        paper_sha256=v["VMI_PAPER_SHA256"],
        llama_server=Path(v["VMI_LLAMA_SERVER"]),
        llm_port=int(v["VMI_LLM_PORT"]),
        llm_model=v["VMI_LLM_MODEL"],
        llm_gguf=Path(v["VMI_LLM_GGUF"]),
        llm_context=int(v["VMI_LLM_CONTEXT"]),
        llm_temperature=float(v["VMI_LLM_TEMPERATURE"]),
        llm_n_cpu_moe=int(v.get("VMI_LLM_N_CPU_MOE", "0")),
        llm_reasoning_budget=int(v.get("VMI_LLM_REASONING_BUDGET", "-1")),
        run_timeout_s=int(v["VMI_RUN_TIMEOUT_S"]),
        run_max_steps=int(v["VMI_RUN_MAX_STEPS"]),
        harness=v.get("VMI_HARNESS", "opencode"),
        claude_cap_usd=float(v.get("VMI_CLAUDE_CAP_USD", "20")),
    )


def anthropic_credentials() -> tuple[str, str]:
    """(API key, workspace id) from .env, read only for Claude runs and passed only to the proxy outside the sandbox.
    The workspace id is required when the key is not scoped to a workspace (API error seen in the plumbing check)."""
    env = dotenv_values(REPO_ROOT / ".env")
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise KeyError("ANTHROPIC_API_KEY is not set in .env")
    return key, env.get("ANTHROPIC_WORKSPACE_ID") or os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
