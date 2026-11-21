#!/usr/bin/env bash
# isolation_check.sh — run INSIDE the agent sandbox (run_one.sh MODE=check); prints one ok/FAIL line per check.
# Global context: logged before every batch of agent runs (preregistration §2); any FAIL stops the queue.
PORT="${1:-8080}"
pyconn() { python3 -c "import socket,sys; socket.create_connection(('$1',$2),3); sys.exit(0)" 2>/dev/null; }
pyconn 1.1.1.1 80 && echo "FAIL external-ip" || echo "ok no-external-ip"
pyconn example.com 443 && echo "FAIL dns+tcp" || echo "ok no-dns"
python3 -c "import socket; socket.getaddrinfo('example.com', 443)" 2>/dev/null && echo "FAIL dns-resolves" || echo "ok dns-fails"
[ "$(python3 -c "import socket; print(len(socket.if_nameindex()))" 2>/dev/null)" = "1" ] && echo "ok only-loopback" || echo "FAIL interfaces: $(python3 -c 'import socket; print(socket.if_nameindex())')"
[ -e /mnt/c ] || [ -e /mnt/d ] && echo "FAIL windows-drives-visible" || echo "ok no-windows-drives"
ls /home 2>/dev/null | grep -q . && echo "FAIL home-visible" || echo "ok home-hidden"
[ -e /run/WSL ] && echo "FAIL wsl-interop-socket-visible" || echo "ok no-interop-socket"
[ -n "${WSL_INTEROP:-}" ] && echo "FAIL WSL_INTEROP set" || echo "ok no-interop-env"
command -v cmd.exe >/dev/null 2>&1 && echo "FAIL cmd.exe-on-path" || echo "ok no-windows-exe"
touch /usr/x 2>/dev/null && echo "FAIL usr-writable" || echo "ok usr-readonly"
python3 -c "import socket; socket.socket(40, socket.SOCK_STREAM)" 2>/dev/null && echo "FAIL vsock-allowed" || echo "ok vsock-blocked"
python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=10)" 2>/dev/null && echo "ok llm-reachable" || echo "FAIL llm-unreachable"
python3 -c "import numpy; print('ok numpy', numpy.__version__)" 2>/dev/null || echo "FAIL numpy-missing"
command -v opencode >/dev/null && echo "ok opencode $(opencode --version 2>/dev/null)" || echo "FAIL opencode-missing"
touch /work/.probe && rm /work/.probe && echo "ok work-writable" || echo "FAIL work-not-writable"
