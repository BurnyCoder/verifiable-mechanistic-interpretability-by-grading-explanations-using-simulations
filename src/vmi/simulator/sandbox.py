"""Create one agent run folder: the sandbox's work files, the OpenCode config and the prompt.

Local explanation: `prepare_run` writes RUN_DIR/work/ (TASK.md, ARCHITECTURE.md, forward.py, selftest.py,
EXPLANATION.md and any table files), RUN_DIR/opencode.jsonc (local provider only; web tools, MCP, subagents, code mode
and updates denied; per-run temperature and seed) and RUN_DIR/prompt.txt, plus run.json describing the condition.
Nothing derived from target weights enters the folder except the explanation text itself.
Global context: scripts/run_one.sh (inside WSL) copies work/ into the bubblewrap sandbox; the queue grades
RUN_DIR/work_out/weights.npz afterwards.
Sources: OpenCode v2 config schema (https://opencode.ai/config.json) and docs — providers with
"@opencode/ai/providers/openai-compatible" and settings.baseURL (https://opencode.ai/docs/providers/), ordered permission
rules and action names (https://opencode.ai/docs/permissions/), agents with steps (https://opencode.ai/docs/agents/).
"""

from __future__ import annotations  # postponed annotations

import json  # config and metadata files
import shutil  # copy template files
from pathlib import Path  # paths

from vmi.config import REPO_ROOT, Settings  # paths and LLM settings

TEMPLATE = REPO_ROOT / "sandbox_template"
PROMPT = ("Your current folder contains TASK.md, ARCHITECTURE.md, EXPLANATION.md, forward.py, selftest.py and a "
          "starter build_weights.py. Read TASK.md and complete the task it describes. Work autonomously and do not ask "
          "questions. When build_weights.py builds your weights.npz and selftest.py passes, stop.")
PROMPT_NO_SELFTEST = ("Your current folder contains TASK.md, ARCHITECTURE.md, EXPLANATION.md, forward.py and a starter "
                      "build_weights.py. Read TASK.md and complete the task it describes. Work autonomously and do not "
                      "ask questions. When build_weights.py builds your weights.npz, stop.")


def turn_output(settings: Settings) -> int:
    """Tokens one model response may use: the thinking budget (server flag) + 8k for one whole-file write (~8k tokens;
    smaller caps truncated it in stage-0 runs 3-5)."""
    return max(settings.llm_reasoning_budget, 0) + 8192


def sampling(temperature: float, seed: int) -> dict:
    """Request sampling parameters, identical for both harnesses. Qwen3.6 model card, "thinking mode for precise
    coding tasks": temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0 (1.5 is its general preset)."""
    return {"temperature": temperature, "seed": seed, "presence_penalty": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def harness_config(settings: Settings, *, model: str, temperature: float, seed: int) -> dict:
    """Config of the custom bash-only harness (harness.py), read inside the sandbox."""
    return {"url": f"http://127.0.0.1:{settings.llm_port}/v1/chat/completions", "model": model,
            "max_tokens": turn_output(settings), "sampling": sampling(temperature, seed),
            "context_tokens": settings.llm_context,  # the harness elides old messages before the prompt overflows
            "step_limit": settings.run_max_steps, "time_limit_s": settings.run_timeout_s - 60,  # exit before the kill
            "request_timeout_s": settings.run_timeout_s, "command_timeout_s": 120, "cwd": "/work"}


CLAUDE_RUN_CAPS_USD = {"claude-haiku-5-5": 2.0, "claude-sonnet-5-5": 6.0, "claude-opus-5-5": 10.0}  # per run
CLAUDE_LEDGER = REPO_ROOT / "results" / "claude" / "ledger.jsonl"  # every priced request of every Claude run


def claude_config(settings: Settings, *, model: str, run_id: str, effort: str | None = None) -> dict:
    """Config of the outside proxy (and Claude Code's effort level) for one Claude Code run; paths as seen from WSL."""
    from vmi.simulator.runner import to_wsl

    CLAUDE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    return {"upstream": "https://api.anthropic.com", "run_id": run_id, "ledger": to_wsl(CLAUDE_LEDGER),
            "effort": effort or "medium",  # Claude Code's default; the hyperparameter stage uses low and high
            "models": sorted({model, "claude-haiku-5-5"}),  # Claude Code's background calls use the small model
            "cap_usd": settings.claude_cap_usd, "run_cap_usd": CLAUDE_RUN_CAPS_USD[model], "reserve_usd": 0.5}


def opencode_config(settings: Settings, *, model: str, temperature: float, seed: int, steps: int) -> dict:
    """OpenCode v2 config: one local OpenAI-compatible provider, everything network-capable denied."""
    deny = lambda action: {"action": action, "resource": "*", "effect": "deny"}  # noqa: E731
    out = turn_output(settings)
    return {
        "$schema": "https://opencode.ai/config.json",
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "websearch": False,
        "model": f"local/{model}",
        "default_agent": "weights",
        "plugins": ["-opencode.provider.ollama", "-opencode.provider.lmstudio", "-opencode.provider.vllm"],
        "providers": {"local": {
            "name": "llama-server (local)",
            "package": "@opencode/ai/providers/openai-compatible",
            "settings": {"baseURL": f"http://127.0.0.1:{settings.llm_port}/v1", "apiKey": "local",
                         "headerTimeout": 900000, "chunkTimeout": 900000},
            "models": {model: {"modelID": model, "name": model,
                               "capabilities": {"tools": True, "temperature": True},
                               # input = context − output, because OpenCode compacts when a step nears the input
                               # limit (left implicit, it compacted at ~33k of 48k in stage-0 run 4)
                               "limit": {"context": settings.llm_context, "input": settings.llm_context - out,
                                         "output": out},
                               "body": sampling(temperature, seed)}},
        }},
        "policies": [
            {"action": "provider.use", "resource": "*", "effect": "deny"},
            {"action": "provider.use", "resource": "local", "effect": "allow"},
            {"action": "websearch.use", "resource": "*", "effect": "deny"},
            {"action": "mcp.use", "resource": "*", "effect": "deny"},
            {"action": "tool.use", "resource": "webfetch:*", "effect": "deny"},
        ],
        "permissions": [{"action": "*", "resource": "*", "effect": "allow"}] + [deny(a) for a in (
            "webfetch", "websearch", "execute", "subagent", "question", "skill", "external_directory",
            "experimental_image_generate")],
        "agents": {"weights": {"description": "Builds transformer weights from an explanation", "mode": "primary",
                               "steps": steps}},
    }


def prepare_run(settings: Settings, run_dir: Path, *, explanation: str, extra_files: dict[str, str], meta: dict,
                model: str | None = None, temperature: float | None = None, seed: int = 0, selftest: bool = True,
                effort: str | None = None) -> Path:
    """Create run_dir with work/, opencode.jsonc, prompt.txt and run.json; return run_dir."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work = run_dir / "work"
    work.mkdir(parents=True)
    # build_weights.py is a starter that saves all-zero arrays of the right shapes: it lets the agent fill in the
    # weights through small edits instead of one long file write (stage-0 run 3 hit the output cap three times)
    for name in ("ARCHITECTURE.md", "forward.py", "build_weights.py") + (("selftest.py",) if selftest else ()):
        shutil.copy(TEMPLATE / name, work / name)
    task = (TEMPLATE / "TASK.md").read_text(encoding="utf-8")
    # the budget the agent is told about is the one the runner enforces
    task = task.replace("{MINUTES}", str(settings.run_timeout_s // 60)).replace("{STEPS}", str(settings.run_max_steps))
    if not selftest:  # self-test ablation: drop the tool and its mentions in TASK.md and the prompt. Audit 8: the starter
        # build_weights.py (copied above in every variant) still says "re-run this script and selftest.py"; all three
        # reported no-self-test runs read that line (none ran selftest.py). Kept as is: the protocol is frozen across rounds.
        task = task.replace("`forward.py` (reference forward pass) and `selftest.py` (checks shapes and task accuracy) are provided; do not edit\n  them. Run `python3 build_weights.py` and then `python3 selftest.py` and fix problems until the self-test passes and\n  the accuracy is as high as you can get it.",
                            "`forward.py` (reference forward pass) is provided; do not edit it. Run `python3 build_weights.py` and fix\n  problems until it runs.").replace("script and the self-test after each stage", "script after each stage")
        assert "selftest" not in task and "self-test" not in task, "self-test ablation still mentions the self-test"
    (work / "TASK.md").write_text(task, encoding="utf-8", newline="\n")
    (work / "EXPLANATION.md").write_text(explanation, encoding="utf-8", newline="\n")
    for name, content in extra_files.items():
        (work / name).write_text(content, encoding="utf-8", newline="\n")
    model = model or settings.llm_model
    claude = model.startswith("claude-")
    temperature = None if claude else (settings.llm_temperature if temperature is None else temperature)
    cfg = opencode_config(settings, model=model, temperature=temperature, seed=seed, steps=settings.run_max_steps)
    (run_dir / "opencode.jsonc").write_text(json.dumps(cfg, indent=2), encoding="utf-8", newline="\n")
    hcfg = harness_config(settings, model=model, temperature=temperature, seed=seed)
    (run_dir / "harness.json").write_text(json.dumps(hcfg, indent=2), encoding="utf-8", newline="\n")
    if model.startswith("claude-"):  # Claude Code: sampling is the model's default (Claude 5.5 rejects custom values)
        ccfg = claude_config(settings, model=model, run_id=run_dir.name, effort=effort)
        (run_dir / "claude.json").write_text(json.dumps(ccfg, indent=2), encoding="utf-8", newline="\n")
    (run_dir / "prompt.txt").write_text(PROMPT if selftest else PROMPT_NO_SELFTEST, encoding="utf-8", newline="\n")
    (run_dir / "run.json").write_text(json.dumps({**meta, "model": model, "temperature": temperature, "seed": seed,
                                                  "selftest": selftest}, indent=2), encoding="utf-8")
    return run_dir
