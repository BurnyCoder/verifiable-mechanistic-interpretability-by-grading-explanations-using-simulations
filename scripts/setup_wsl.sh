#!/usr/bin/env bash
# setup_wsl.sh — one-time, user-level preparation of the agent sandbox inside WSL (no sudo, no downloads).
# Global context: the sandbox mounts only /usr, /etc and a few read-only extras; this script prepares those extras:
#   ~/vmi-sim/pylib        numpy for the agent (copied from the user's site-packages; /home is hidden in the sandbox)
#   ~/vmi-sim/novsock.bpf  seccomp filter blocking AF_VSOCK sockets (see make_seccomp.py)
#   ~/vmi-sim/sandboxes    per-run working copies on the native ext4 filesystem (fast, and outside /mnt)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HOME/vmi-sim"
mkdir -p "$BASE/pylib" "$BASE/sandboxes"
SITE="$(python3 -c 'import numpy, os; print(os.path.dirname(os.path.dirname(numpy.__file__)))')"
rm -rf "$BASE/pylib/numpy" "$BASE/pylib/numpy.libs" "$BASE"/pylib/numpy-*.dist-info
cp -r "$SITE/numpy" "$BASE/pylib/"
[ -d "$SITE/numpy.libs" ] && cp -r "$SITE/numpy.libs" "$BASE/pylib/"
cp -r "$SITE"/numpy-*.dist-info "$BASE/pylib/" 2>/dev/null || true
python3 "$HERE/make_seccomp.py" "$BASE/novsock.bpf"
PYTHONPATH="$BASE/pylib" python3 -c 'import numpy; print("sandbox numpy", numpy.__version__)'
command -v bwrap >/dev/null && bwrap --version
command -v rg >/dev/null && rg --version | head -1
"$HOME/.opencode/bin/opencode" --version
