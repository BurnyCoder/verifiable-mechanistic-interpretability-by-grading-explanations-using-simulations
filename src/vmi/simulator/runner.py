"""Run one prepared agent run through WSL + bubblewrap and turn its event stream into a readable transcript.

Local explanation: `run_agent` calls scripts/run_one.sh inside WSL with the run folder's WSL path, streams the
agent's NDJSON events to the terminal and a timestamped log while it works (by tailing the sandbox's event file over
the \\\\wsl.localhost share), and afterwards writes RUN_DIR/logs/transcript.md with every prompt, reasoning block, tool
call (input and full output) and reply, uncut. `isolation_check` runs scripts/isolation_check.sh in the identical
sandbox and returns its ok/FAIL lines.
Global context: called by the queue for every agent condition; the grader then reads RUN_DIR/work_out/weights.npz.
Sources: OpenCode `run --format json` NDJSON events (type, timestamp, sessionID, part; https://opencode.ai/docs/cli/);
WSL path translation /mnt/<drive>/… (https://learn.microsoft.com/en-us/windows/wsl/filesystems).
"""

from __future__ import annotations  # postponed annotations

import json  # NDJSON events
import logging  # timestamped run log
import os  # environment for the Claude proxy key
import subprocess  # wsl.exe
import threading  # live tailing of the event stream
import time  # polling interval, timestamps
from pathlib import Path, PureWindowsPath  # Windows ↔ WSL path translation

from vmi.config import REPO_ROOT, Settings  # paths, port, timeout

log = logging.getLogger("vmi.runner")
WSL_DISTRO = "Ubuntu"
WSL_USER_HOME = r"\\wsl.localhost\Ubuntu\home"  # Windows view of WSL's /home (used only for live tailing)


def to_wsl(path: Path) -> str:
    """D:\\AI\\x → /mnt/d/AI/x (WSL's automount of Windows drives)."""
    p = PureWindowsPath(path.resolve())
    return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:])


def _wsl_home() -> str:
    """$HOME inside WSL (e.g. /home/burny)."""
    return subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-c", "echo $HOME"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _describe_claude(ev: dict) -> str:
    """One readable block per Claude Code stream-json event (assistant/user messages, result)."""
    t = ev["type"]
    if t == "result":
        return (f"[result] {ev.get('subtype')} turns={ev.get('num_turns')} cost=${ev.get('total_cost_usd')} "
                f"error={ev.get('is_error')}\n{ev.get('result', '')}")
    blocks = []
    for b in (ev.get("message") or {}).get("content") or []:
        kind = b.get("type")
        if kind == "text":
            blocks.append(f"[text] {b.get('text', '')}")
        elif kind == "thinking":
            blocks.append(f"[reasoning] {b.get('thinking', '')}")
        elif kind == "tool_use":
            blocks.append(f"[tool {b.get('name')}] input: {json.dumps(b.get('input'), ensure_ascii=False)}")
        elif kind == "tool_result":
            content = b.get("content")
            text = content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content or [])
            blocks.append(f"[tool result] {text}")
        else:
            blocks.append(f"[{kind}] {json.dumps(b, ensure_ascii=False)}")
    usage = (ev.get("message") or {}).get("usage")
    return "\n".join(blocks + ([f"[step_finish] tokens={json.dumps(usage)}"] if t == "assistant" and usage else []))


def _describe(ev: dict) -> str:
    """One readable block per OpenCode / harness / Claude Code event (unknown shapes are printed as JSON, never dropped)."""
    t = ev.get("type", "?")
    if t in ("assistant", "user", "result") and "part" not in ev:
        return _describe_claude(ev)
    part = ev.get("part") or {}
    if t in ("text", "reasoning"):
        return f"[{t}] {part.get('text', ev.get('text', ''))}"
    if t == "tool_use":
        state = part.get("state") or {}
        return (f"[tool {part.get('tool', ev.get('tool', '?'))}] input: {json.dumps(state.get('input', ev.get('input')), ensure_ascii=False)}\n"
                f"output: {state.get('output', ev.get('output', ''))}" + (f"\nerror: {state.get('error')}" if state.get("error") else ""))
    if t in ("step_start", "step_finish"):
        tokens = part.get("tokens") or ev.get("tokens")
        return f"[{t}]" + (f" tokens={json.dumps(tokens)}" if tokens else "")
    return f"[{t}] {json.dumps(ev, ensure_ascii=False)}"


def _tail(path: Path, stop: threading.Event, sink) -> None:
    """Print new NDJSON lines of a growing file until stop is set (best effort; the final file is re-read later)."""
    pos = 0
    while not stop.is_set():
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    for line in f:
                        if line.endswith("\n"):
                            pos += len(line.encode("utf-8"))
                            try:
                                sink(_describe(json.loads(line)))
                            except json.JSONDecodeError:
                                sink(line.rstrip())
        except OSError:
            pass
        time.sleep(2)


def write_transcript(run_dir: Path) -> dict:
    """Write logs/transcript.md from logs/events.ndjson; return summary counts."""
    ev_file = run_dir / "logs" / "events.ndjson"
    events = []
    if ev_file.exists():
        for line in ev_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "raw", "text": line})
    prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    lines = [f"# Transcript {run_dir.name}\n", f"**Prompt:** {prompt}\n"]
    for ev in events:
        ts = ev.get("timestamp")
        stamp = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if isinstance(ts, (int, float)) else ""
        lines.append(f"`{stamp}` {_describe(ev)}\n")
    (run_dir / "logs" / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    steps = sum(e.get("type") in ("step_finish", "assistant") for e in events)
    tools = sum(e.get("type") == "tool_use" for e in events) + sum(
        b.get("type") == "tool_use" for e in events if e.get("type") == "assistant"
        for b in (e.get("message") or {}).get("content") or [])
    errors = sum(e.get("type") == "error" for e in events) + sum(
        bool(e.get("is_error")) for e in events if e.get("type") == "result")
    return {"events": len(events), "steps": steps, "tool_calls": tools, "errors": errors}


def run_agent(settings: Settings, run_dir: Path, mode: str = "agent") -> dict:
    """Run the sandboxed agent for run_dir; returns exit info and transcript summary."""
    script = to_wsl(REPO_ROOT / "scripts" / "run_one.sh")
    run_id = run_dir.name
    run_log = run_dir / "logs" / "runner.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(run_log, "a", encoding="utf-8")

    def sink(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_id} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    home_unc = Path(WSL_USER_HOME) / _wsl_home().split("/home/", 1)[-1]
    live = home_unc / "vmi-sim" / "sandboxes" / run_id / "out" / "events.ndjson"
    stop = threading.Event()
    tailer = threading.Thread(target=_tail, args=(live, stop, sink), daemon=True)
    tailer.start()
    sink(f"start mode={mode}")
    model = json.loads((run_dir / "run.json").read_text())["model"]
    env = None
    if mode == "claude":  # credentials go only to the proxy outside the sandbox (WSLENV forwards them to WSL)
        from vmi.config import anthropic_credentials
        key, workspace = anthropic_credentials()
        env = {**os.environ, "VMI_ANTHROPIC_KEY": key, "VMI_ANTHROPIC_WORKSPACE_ID": workspace,
               "WSLENV": ":".join(filter(None, [os.environ.get("WSLENV", ""), "VMI_ANTHROPIC_KEY/u",
                                                "VMI_ANTHROPIC_WORKSPACE_ID/u"]))}
    t0 = time.time()
    proc = subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "--", "bash", script, to_wsl(run_dir), str(settings.llm_port),
                           str(settings.run_timeout_s), model if mode == "claude" else f"local/{model}", mode],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    stop.set()
    tailer.join(timeout=5)
    sink(f"finished in {time.time() - t0:.0f}s (wsl exit {proc.returncode})")
    if proc.stderr.strip():
        sink("wsl stderr: " + proc.stderr.strip()[-2000:])
    fh.close()
    exit_info = {}
    ef = run_dir / "logs" / "exit.json"
    if ef.exists():
        exit_info = json.loads(ef.read_text())
    return {**exit_info, **write_transcript(run_dir), "wsl_returncode": proc.returncode}


def claude_check(settings: Settings, scratch: Path, model: str = "claude-haiku-5-5") -> dict:
    """Plumbing check of the Claude mode (sandbox → relay → proxy → API) with one tiny request; returns the summary."""
    import dataclasses
    from vmi.simulator.sandbox import prepare_run

    run_dir = prepare_run(settings, scratch / "claude_check", explanation="(plumbing check)", extra_files={},
                          meta={"condition": "claude_check"}, model=model)
    (run_dir / "prompt.txt").write_text("Reply with the single word OK. Do not use any tools.", encoding="utf-8")
    info = run_agent(dataclasses.replace(settings, run_timeout_s=180), run_dir, mode="claude")
    info["proxy_log"] = (run_dir / "logs" / "proxy.log").read_text(encoding="utf-8", errors="replace")[-2000:]
    return info


def isolation_check(settings: Settings, scratch: Path) -> list[str]:
    """Run the isolation checks inside the identical sandbox; return the result lines."""
    from vmi.simulator.sandbox import prepare_run

    run_dir = prepare_run(settings, scratch / "isolation_check", explanation="(isolation check)", extra_files={},
                          meta={"condition": "isolation_check"})
    run_agent(settings, run_dir, mode="check")
    return (run_dir / "logs" / "events.ndjson").read_text(encoding="utf-8").splitlines()
