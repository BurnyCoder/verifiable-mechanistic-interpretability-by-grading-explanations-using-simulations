#!/usr/bin/env bash
# run_one.sh — run one weights-blind, offline agent run inside bubblewrap (called from Windows via wsl.exe).
#
# Usage: run_one.sh RUN_DIR PORT TIMEOUT_S MODEL_REF [MODE]
#   RUN_DIR   run folder as a WSL path (e.g. /mnt/d/AI/vmi-sim/runs/<id>) holding work/, opencode.jsonc, prompt.txt
#   PORT      TCP port of llama-server on the Windows host (reachable from WSL as 127.0.0.1 in mirrored mode)
#   MODEL_REF OpenCode model reference, e.g. local/qwen3.6-35b-a3b
#   MODE      "agent" (default, OpenCode), "harness" (the custom bash-only harness, src/vmi/simulator/harness.py) or
#             "check" (run scripts/isolation_check.sh in the identical sandbox instead)
#
# Isolation (threat model: accidental access to the target's weights or the internet, not a malicious model):
#   --unshare-all            new user/pid/ipc/uts/cgroup/NETWORK namespaces: only loopback exists inside
#   --clearenv               no inherited variables (no WSL_INTEROP, no tokens)
#   only /usr, /etc (ro), the OpenCode binary (ro), numpy (ro), the relay script (ro), the run's work and state dirs
#   and the LLM socket dir are visible; /home, /mnt (Windows drives), /run are not
#   --seccomp                AF_VSOCK sockets fail with EPERM (WSL host services use Hyper-V sockets)
# Sources: bubblewrap man page (https://man.archlinux.org/man/bwrap.1.en); OpenCode headless `run`
# (https://opencode.ai/docs/cli/); the unix-socket bridge pattern of Claude Code's Linux sandbox
# (https://code.claude.com/docs/en/sandboxing).
set -uo pipefail
RUN="$1"; PORT="$2"; TIMEOUT="$3"; MODEL="$4"; MODE="${5:-agent}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
BASE="$HOME/vmi-sim"
ID="$(basename "$RUN")"
B="$BASE/sandboxes/$ID"
rm -rf "$B"
mkdir -p "$B/work" "$B/sock" "$B/state/home" "$B/state/config/opencode" "$B/state/data" "$B/state/cache" \
         "$B/state/st" "$B/state/tmp" "$B/out" "$RUN/logs"
cp -r "$RUN/work/." "$B/work/"
cp "$RUN/opencode.jsonc" "$B/state/config/opencode/opencode.jsonc"
cp "$RUN/prompt.txt" "$B/state/prompt.txt"
[ -f "$RUN/harness.json" ] && cp "$RUN/harness.json" "$B/state/harness.json"
cp "$HERE/isolation_check.sh" "$B/state/isolation_check.sh"

# Outside relay: socket file → llama-server; logs every byte (full prompts and completions).
python3 "$REPO/src/vmi/simulator/relay.py" unix-to-tcp "$B/sock/llm.sock" 127.0.0.1 "$PORT" "$B/out/wire.log" &
RELAY=$!
trap 'kill $RELAY 2>/dev/null' EXIT
for _ in $(seq 50); do [ -S "$B/sock/llm.sock" ] && break; sleep 0.1; done

if [ "$MODE" = "check" ]; then
  INNER='python3 /opt/vmi/relay.py tcp-to-unix 127.0.0.1 '"$PORT"' /llm/llm.sock & sleep 0.5; exec bash /state/isolation_check.sh '"$PORT"
elif [ "$MODE" = "harness" ]; then  # custom bash-only harness (src/vmi/simulator/harness.py) instead of OpenCode
  INNER='python3 /opt/vmi/relay.py tcp-to-unix 127.0.0.1 '"$PORT"' /llm/llm.sock & sleep 0.5
exec python3 /opt/vmi/harness.py /state/harness.json "$(cat /state/prompt.txt)" </dev/null'
else
  INNER='python3 /opt/vmi/relay.py tcp-to-unix 127.0.0.1 '"$PORT"' /llm/llm.sock & sleep 0.5
exec opencode run --standalone --auto --format json --thinking --agent weights --model '"$MODEL"' "$(cat /state/prompt.txt)" </dev/null'
fi

START=$(date +%s)
timeout --signal=TERM --kill-after=30 "$TIMEOUT" \
bwrap --die-with-parent --new-session --unshare-all --clearenv \
  --seccomp 9 9<"$BASE/novsock.bpf" \
  --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin \
  --ro-bind /etc /etc --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind "$HOME/.opencode/bin" /opt/oc \
  --ro-bind "$BASE/pylib" /opt/py \
  --ro-bind "$REPO/src/vmi/simulator/relay.py" /opt/vmi/relay.py \
  --ro-bind "$REPO/src/vmi/simulator/harness.py" /opt/vmi/harness.py \
  --bind "$B/work" /work --bind "$B/state" /state --bind "$B/sock" /llm \
  --setenv PATH /opt/oc:/usr/bin:/bin --setenv HOME /state/home --setenv PYTHONPATH /opt/py \
  --setenv XDG_CONFIG_HOME /state/config --setenv XDG_DATA_HOME /state/data --setenv XDG_CACHE_HOME /state/cache \
  --setenv XDG_STATE_HOME /state/st --setenv TMPDIR /state/tmp --setenv LANG C.UTF-8 \
  --chdir /work \
  bash -c "$INNER" >"$B/out/events.ndjson" 2>"$B/out/stderr.log"
CODE=$?
END=$(date +%s)
echo "{\"exit_code\": $CODE, \"seconds\": $((END - START)), \"timeout_s\": $TIMEOUT}" >"$B/out/exit.json"

# Copy everything back to the Windows run folder.
mkdir -p "$RUN/work_out"
cp -r "$B/work/." "$RUN/work_out/"
cp "$B/out/"* "$RUN/logs/"
[ -d "$B/state/data/opencode" ] && cp -r "$B/state/data/opencode" "$RUN/logs/opencode_data"
gzip -f "$RUN/logs/wire.log" 2>/dev/null || true
rm -rf "$B"
exit 0
