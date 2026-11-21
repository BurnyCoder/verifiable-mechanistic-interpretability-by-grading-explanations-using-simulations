# Usage

All commands run from the repository root on Windows; agent runs call into WSL automatically.

## One-time setup

1. Python environment: `uv sync` (Python 3.12, torch 2.14 CUDA 13.0, pinned `grokviz`).
2. Settings: copy `.env.example` to `.env` and adjust paths (defaults use `D:\AI\vmi-sim`).
3. Targets: download the pinned runs (revision in `.env.example`):
   `hf download BurnyCoder/grokking-modular-addition-runs --revision <rev> --local-dir D:/AI/vmi-sim/data/grokking-runs`.
   The paper's checkpoint is downloaded and sha256-checked on first use.
4. Local LLM: llama.cpp b12600 (Windows CUDA 13.5 build) in `D:\AI\vmi-sim\llama.cpp\b12600`, the GGUF files in
   `D:\AI\vmi-sim\models` (Qwen3.6-35B-A3B UD-IQ4_XS; Qwen3.5-9B and -4B Q4_K_M for the ablation).
5. WSL (Ubuntu, bubblewrap ≥ 0.9, ripgrep, OpenCode 2.3.2 at `~/.opencode/bin`): run `scripts/setup_wsl.sh` inside WSL
   once (copies numpy for the sandbox and builds the seccomp filter). WSL must use mirrored networking so it can reach
   `127.0.0.1` on Windows.

## Commands (`uv run vmi <command>`)

| Command | What it does | Writes |
|---|---|---|
| `serve [--n-cpu-moe N] [--gguf F --alias A]` | start llama-server and keep it running | `logs/llama-server_*.log` |
| `check` | isolation checks inside the sandbox | `results/isolation/` |
| `explain TARGET` | measure the target's mechanism, render E0–E4, W-*, format variants (and E-paper for T-paper) | `results/explanations/TARGET/` |
| `compile TARGET` | reference-compile every explanation and grade it | `results/compiler/TARGET/` |
| `run TARGET LEVEL SEED [--stage S]` | one agent run | `results/runs/<run_id>.json`, run folder under `VMI_RUNS_DIR` |
| `manifests ROUND` | write the round's JSONL manifests (one per stage) | `results/manifests/` |
| `queue MANIFEST` | run every condition of a manifest without a result (resumable) | `results/runs/` |
| `controls ROUND` | planted calibration, random init, wrong targets, trajectory | `results/controls/` |
| `baselines TARGET` | retrains, distilled students, Fourier-restricted M, weights dump | `results/baselines/` |
| `analyse ROUND` | evaluate all hypotheses at the round's N | `results/hypotheses_round*.json` |

Targets: `T0`, `T1`, `T2`, `dev`, `T-paper`, `T-new`, or any run id (`wd1_s3`, `wd0_s0`, `rand_s1`, ...).
Levels: `E0`–`E4`, `W-freq`, `W-alg`, `W-role`, `W-scramble`, `E-paper` (T-paper only), `E3-fmt-math`, `E3-fmt-code`.

## What one agent run does

1. `simulator/sandbox.py` creates the run folder: `work/` (TASK.md, ARCHITECTURE.md, EXPLANATION.md, forward.py,
   selftest.py, and neurons.csv for E4), `opencode.jsonc`, `prompt.txt`, `run.json`.
2. `simulator/runner.py` calls `scripts/run_one.sh` in WSL, which copies `work/` to WSL's native disk, starts the outside
   relay (unix socket → llama-server, logging every byte), and runs `opencode run` inside bubblewrap with no network,
   no /home, no Windows drives and AF_VSOCK blocked. The agent's events stream to the terminal and `logs/runner.log`.
3. Afterwards `work_out/`, `logs/events.ndjson`, `logs/transcript.md`, the gzipped wire log and OpenCode's own data are
   in the run folder; `simulator/audit.py` checks `build_weights.py` for training loops; `grader/score.py` grades
   `weights.npz` against the target (failures are graded as the all-zero model).
