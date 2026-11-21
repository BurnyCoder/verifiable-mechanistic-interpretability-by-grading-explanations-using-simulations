"""Stage 0: choose llama-server's --n-cpu-moe for this GPU and measure throughput (writes results/stage0/llm_tuning.json).

For each N (MoE layers kept in system RAM), start the server, record GPU memory, time a fixed prompt-processing +
generation request (llama-server reports prompt/predicted tokens per second in `timings`), stop the server.
The preregistered fallback (qwen3.5-9b) applies if the best setting gives < 8 tokens/s generation.
Source: llama-server /completion `timings` fields (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from vmi import config
from vmi.simulator import llm_server

PROMPT = ("Write a numpy function that builds a 113x113 matrix whose (a, b) entry is cos(2*pi*k*(a+b)/113) for k = 7, "
          "then explain in two sentences why the sum over several k peaks at a+b mod 113. ") * 20  # ~1.2k tokens


def gpu_mem_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def main() -> None:
    s = config.load()
    results = []
    for n in [int(x) for x in sys.argv[1:]] or [40, 32, 26, 22, 18]:
        base = gpu_mem_mib()
        try:
            proc = llm_server.start(s, n_cpu_moe=n, wait_s=900)
        except Exception as exc:  # noqa: BLE001 — record OOM / load failures and continue with the next N
            results.append({"n_cpu_moe": n, "error": str(exc)})
            print(n, "failed:", exc, flush=True)
            continue
        try:
            mem = gpu_mem_mib()
            t0 = time.time()
            r = httpx.post(f"http://127.0.0.1:{s.llm_port}/completion", timeout=900,
                           json={"prompt": PROMPT, "n_predict": 256, "temperature": 0.6, "seed": 0, "cache_prompt": False}).json()
            tm = r.get("timings", {})
            row = {"n_cpu_moe": n, "gpu_mem_mib": mem, "gpu_mem_before_mib": base, "prompt_tokens": tm.get("prompt_n"),
                   "prompt_tps": tm.get("prompt_per_second"), "gen_tokens": tm.get("predicted_n"),
                   "gen_tps": tm.get("predicted_per_second"), "wall_s": time.time() - t0}
            results.append(row)
            print(row, flush=True)
        finally:
            llm_server.stop(proc)
            time.sleep(5)
    out = config.REPO_ROOT / "results" / "stage0" / "llm_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
