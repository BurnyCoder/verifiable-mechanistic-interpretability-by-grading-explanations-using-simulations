"""Minimal bash-only agent harness for the sandboxed local LLM (alternative to OpenCode; runs INSIDE the sandbox).

Local explanation: a loop of (query the model → run the single bash command in its reply → send the output back),
following mini-SWE-agent's design: bash is the only tool (no tool-calling interface), the history is linear, and
every action is an independent `subprocess.run`. Observations are sent as user messages, so Qwen's chat template
keeps only the latest turn's thinking in the prompt and the context does not fill up with old reasoning. A reply cut
off by the output limit stays in the history (shortened if very long) followed by a note, instead of being dropped and
retried from the same context (OpenCode's behaviour in stage 0). Events are printed as NDJSON in the same shape as
OpenCode's (`type`, `timestamp`, `part`), so the runner's live log and transcript code is reused unchanged.
Global context: scripts/run_one.sh starts it in the bubblewrap sandbox with `MODE=harness`; it talks to llama-server
through the in-sandbox relay at 127.0.0.1:PORT; the raw requests/responses are in the relay's wire log. Standard
library only, because the sandbox has only the system Python plus numpy.
Sources: mini-SWE-agent (https://github.com/SWE-agent/mini-swe-agent): agents/default.py control flow and
config/mini_textbased.yaml prompts, action format, completion command and 10,000-character observation limit;
OpenAI-compatible chat completions with `reasoning_content` (llama.cpp tools/server/README.md).
"""

from __future__ import annotations  # postponed annotations

import json  # config, HTTP bodies, NDJSON events
import re  # extract the bash block
import subprocess  # run the action
import sys  # argv
import time  # timestamps, wall-clock limit
import urllib.error  # HTTP errors from the server
import urllib.request  # stdlib HTTP client (no third-party packages inside the sandbox)

DONE = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"  # mini-SWE-agent's completion command: `echo DONE`
ACTION_RE = re.compile(r"```(?:bash|sh|mswea_bash_command)?[ \t]*\n(.*?)\n```", re.S)  # one fenced command block
MAX_OBS = 10_000  # observation characters shown in full (mini-SWE-agent's limit); longer ones show head and tail
MAX_KEPT_REPLY = 12_000  # characters of a truncated reply kept in the history (head + tail)

SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

INSTANCE = """{task}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended workflow

1. Read the files in the current folder.
2. Write or change build_weights.py.
3. Run it and the self-test, and fix problems.
4. When you are done, finish by issuing exactly this command: `echo {done}`.
   Do not combine it with any other command. After this command, you cannot continue working on this task.

## Important rules

1. Every response must contain exactly one action, enclosed in triple backticks.
2. Directory or environment variable changes are not persistent: every action runs in a new subshell in the
   task folder.
3. Your responses have an output limit. Write long files in several smaller commands (for example one
   `cat <<'EOF' >> build_weights.py` per section) and keep code comments short.

## Useful command examples

Create a file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
print(np.zeros(3))
EOF
```

Edit a file:

```bash
sed -i 's/old_string/new_string/g' filename.py
```

View lines with numbers:

```bash
nl -ba filename.py | sed -n '10,20p'
```"""


def emit(kind: str, **part) -> None:
    """Print one NDJSON event in OpenCode's shape (read by the runner's live log and transcript writer)."""
    print(json.dumps({"type": kind, "timestamp": int(time.time() * 1000), "part": part}, ensure_ascii=False), flush=True)


def query(cfg: dict, messages: list[dict]) -> dict:
    """One non-streaming chat completion from the local server."""
    body = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"], **cfg["sampling"]}
    req = urllib.request.Request(cfg["url"], json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg["request_timeout_s"]) as r:
        return json.load(r)


def shorten(text: str, limit: int) -> str:
    """Keep head and tail of an over-long text, with the number of elided characters in between."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters elided ...]\n{text[-half:]}"


def observe(cmd: str, cwd: str, timeout_s: int) -> tuple[str, str]:
    """Run one action; return (output for the model, output for the log)."""
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        out, code = out + f"\n[command timed out after {timeout_s} s]", -1
    if len(out) > MAX_OBS:
        shown = (f"<warning>The output was too long ({len(out)} characters); only the head and tail are shown. Use a "
                 f"command that prints less.</warning>\n{shorten(out, MAX_OBS)}")
    else:
        shown = out
    return f"<returncode>{code}</returncode>\n<output>\n{shown}\n</output>", out


def run(cfg: dict, task: str) -> str:
    """The agent loop; returns the exit status."""
    t0 = time.time()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTANCE.format(task=task, done=DONE)}]
    emit("text", text=f"[harness] prompt: {messages[1]['content']}")
    format_errors = 0
    for step in range(1, cfg["step_limit"] + 1):
        if time.time() - t0 > cfg["time_limit_s"]:
            return "TimeExceeded"
        emit("step_start", step=step)
        try:
            resp = query(cfg, messages)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            body = exc.read().decode(errors="replace") if isinstance(exc, urllib.error.HTTPError) else ""
            emit("error", error={"type": "request", "message": f"{exc} {body}"[:4000]})
            return "RequestError"
        choice = resp["choices"][0]
        msg, reason = choice.get("message") or {}, choice.get("finish_reason")
        thinking, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        if thinking:
            emit("reasoning", text=thinking)
        emit("text", text=content)
        emit("step_finish", reason=reason, tokens=resp.get("usage"))
        if reason == "length":  # cut off: keep what was written, say so, let the model continue
            messages.append({"role": "assistant", "content": shorten(content, MAX_KEPT_REPLY)})
            note = (f"Your last response was cut off at the output limit ({cfg['max_tokens']} tokens including "
                    "thinking) before it was complete, so nothing was executed. Think less before acting, and write "
                    "long files in several smaller commands.")
            emit("error", error={"type": "output-limit", "message": note})
            messages.append({"role": "user", "content": note})
            continue
        actions = ACTION_RE.findall(content)
        messages.append({"role": "assistant", "content": content})
        if len(actions) != 1:
            format_errors += 1
            note = (f"Format error: your response must contain exactly one bash code block (found {len(actions)}). "
                    "Reply with a THOUGHT and exactly one ```bash block.")
            emit("error", error={"type": "format", "message": note})
            if format_errors >= cfg.get("max_format_errors", 3):
                return "RepeatedFormatError"
            messages.append({"role": "user", "content": note})
            continue
        format_errors = 0
        cmd = actions[0].strip()
        if cmd.startswith("echo") and DONE in cmd:
            emit("tool_use", tool="bash", state={"status": "completed", "input": {"command": cmd}, "output": "(submitted)"})
            return "Submitted"
        shown, full = observe(cmd, cfg["cwd"], cfg["command_timeout_s"])
        emit("tool_use", tool="bash", state={"status": "completed", "input": {"command": cmd}, "output": full})
        messages.append({"role": "user", "content": shown})
    return "LimitsExceeded"


def main() -> None:
    cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
    status = run(cfg, sys.argv[2])
    emit("text", text=f"[harness] exit status: {status}")


if __name__ == "__main__":
    main()
