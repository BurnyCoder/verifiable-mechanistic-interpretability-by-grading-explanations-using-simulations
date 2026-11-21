"""Start, check and stop the local llama.cpp `llama-server` on Windows (GPU) for the agent runs.

Local explanation: `start` launches llama-server with a fixed GGUF file, an alias the OpenCode config refers to,
the preregistered context and sampling defaults, and `--n-cpu-moe N` (mixture-of-experts weights of the first N layers
kept in system RAM so the rest fits in 8 GB VRAM); it waits for GET /health. `stop` terminates it. The full server log
goes to logs/llama-server_<timestamp>.log.
Global context: the sandboxed agent reaches this server only through the unix-socket relay; the server has no model
download endpoint (unlike Ollama's /api/pull), so the agent cannot make it fetch anything.
Sources: llama-server flags (llama.cpp tools/server/README.md, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md);
`--n-cpu-moe` for MoE offload; Qwen sampling recommendations for thinking mode (temperature 0.6, top-p 0.95, top-k 20;
https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
"""

from __future__ import annotations  # postponed annotations

import subprocess  # background process
import time  # waiting for readiness
from pathlib import Path  # paths

import httpx  # health checks

from vmi.config import REPO_ROOT, Settings  # paths and LLM settings


BUDGET_MESSAGE = " I have planned enough; I will now act on this plan with my next tool call."


def health(settings: Settings, timeout: float = 2.0) -> bool:
    """True if llama-server answers /health with 200 (model loaded)."""
    try:
        return httpx.get(f"http://127.0.0.1:{settings.llm_port}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def start(settings: Settings, *, gguf: Path | None = None, alias: str | None = None, n_cpu_moe: int | None = None,
          ctx: int | None = None, extra: list[str] | None = None, wait_s: int = 900) -> subprocess.Popen:
    """Launch llama-server and block until it is healthy (raises after wait_s seconds)."""
    if health(settings):
        raise RuntimeError(f"something is already serving on port {settings.llm_port}")
    log_file = REPO_ROOT / "logs" / f"llama-server_{time.strftime('%Y%m%dT%H%M%S')}.log"
    cmd = [str(settings.llama_server), "-m", str(gguf or settings.llm_gguf), "-a", alias or settings.llm_model,
           "--host", "127.0.0.1", "--port", str(settings.llm_port), "-c", str(ctx or settings.llm_context),
           "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--jinja", "--no-webui",
           "--temp", str(settings.llm_temperature), "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
           "--log-file", str(log_file)]
    n_cpu_moe = settings.llm_n_cpu_moe if n_cpu_moe is None else n_cpu_moe
    if n_cpu_moe:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    if settings.llm_reasoning_budget >= 0:  # cap thinking per turn; the message is inserted before </think>
        cmd += ["--reasoning-budget", str(settings.llm_reasoning_budget), "--reasoning-budget-message", BUDGET_MESSAGE]
    cmd += extra or []
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}; see {log_file}")
        if health(settings):
            return proc
        time.sleep(3)
    proc.terminate()
    raise TimeoutError(f"llama-server not healthy after {wait_s}s; see {log_file}")


def stop(proc: subprocess.Popen) -> None:
    """Terminate llama-server and wait for it to exit."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
