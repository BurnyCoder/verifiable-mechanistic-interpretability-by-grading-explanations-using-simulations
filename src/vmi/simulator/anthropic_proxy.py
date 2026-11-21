"""Proxy outside the sandbox for Claude Code runs: socket → https://api.anthropic.com with key injection, a dollar cap
and full logging.

Local explanation: Claude Code inside the bubblewrap sandbox (no network) talks plain HTTP to 127.0.0.1:PORT; the
in-sandbox relay carries that to a unix socket served by this proxy. For every request the proxy (1) allows only
/v1/messages endpoints and only the configured models, (2) rejects requests that declare server-side tools with
internet access (web search, web fetch, code execution, MCP servers), (3) refuses new requests once the global or
per-run budget would be exceeded, (4) replaces the sandbox's placeholder credentials with the real API key — so the key
never enters the sandbox — and forwards the request over TLS, (5) streams the response back, prices its `usage` and
appends one line per request to a persistent JSONL ledger, and (6) writes the full request and response bodies to the
run's wire log (headers are not logged, so the key never is).
Global context: started by scripts/run_one.sh in MODE=claude in place of the llama-server relay; the ledger
(results/claude/ledger.jsonl) is shared by all Claude runs and is what enforces the user's $20 total cap.
Standard library only (it runs on WSL's system Python).
Sources: prices from the Claude API reference bundled with Claude Code (shared/models.md: Haiku 5.5 $0.10/$0.50 per
MTok up to 100K-token prompts, $0.50/$2.50 above; Sonnet 5.5 $2/$10, cache reads $0.10; Opus 5.5 $4/$20, cache reads
$0.20; shared/prompt-caching.md: cache writes 1.25x input for the 5-minute TTL and 2x for 1 hour, cache reads ~0.1x
input elsewhere); Messages API streaming events message_start / message_delta carry `usage`
(https://docs.anthropic.com/en/api/messages-streaming); http.server / socketserver / http.client (Python docs).
"""

from __future__ import annotations  # postponed annotations

import http.client  # upstream HTTPS client
import json  # bodies, config, ledger
import os  # environment (API key)
import socketserver  # threaded unix/TCP servers
import ssl  # TLS verification for the upstream
import sys  # argv
import threading  # ledger / log lock
import time  # timestamps
from http.server import BaseHTTPRequestHandler  # HTTP parsing on any stream socket
from urllib.parse import urlsplit  # upstream URL

# USD per million tokens: input, output, cache read; cache writes are priced from input (1.25x 5-minute, 2x 1-hour)
PRICES = {
    "claude-haiku-5-5": {"tiers": [(100_000, (0.10, 0.50, 0.01)), (None, (0.50, 2.50, 0.05))]},
    "claude-sonnet-5-5": {"tiers": [(None, (2.00, 10.00, 0.10))]},
    "claude-opus-5-5": {"tiers": [(None, (4.00, 20.00, 0.20))]},
}
# server-side tools run on Anthropic's side, outside the sandbox (client-side tools such as bash run inside it)
BLOCKED_TOOL_TYPES = ("web_search", "web_fetch", "code_execution")
DROP_HEADERS = {"host", "x-api-key", "authorization", "content-length", "accept-encoding", "connection",
                "proxy-connection", "transfer-encoding", "keep-alive"}
LOCK = threading.Lock()


def price(model: str, usage: dict) -> float:
    """Dollar cost of one response's usage (raises KeyError for an unpriced model)."""
    inp = int(usage.get("input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    split = usage.get("cache_creation") or {}
    w1h = int(split.get("ephemeral_1h_input_tokens") or 0)
    w5m = write - w1h if split else write
    out = int(usage.get("output_tokens") or 0)
    prompt = inp + read + write
    for limit, (p_in, p_out, p_read) in PRICES[model]["tiers"]:
        if limit is None or prompt <= limit:
            break
    return (inp * p_in + read * p_read + w5m * 1.25 * p_in + w1h * 2.0 * p_in + out * p_out) / 1e6


def spent(ledger: str, run_id: str | None = None) -> float:
    """Total dollars in the ledger (optionally for one run)."""
    if not os.path.exists(ledger):
        return 0.0
    total = 0.0
    with open(ledger, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if run_id is None or row.get("run_id") == run_id:
                    total += float(row.get("cost_usd") or 0.0)
    return total


def usage_from_body(body: bytes, content_type: str) -> tuple[dict, str | None]:
    """Merged `usage` and model from a JSON or server-sent-events Messages response."""
    usage: dict = {}
    model = None
    if "event-stream" in content_type:
        for line in body.decode("utf-8", "replace").splitlines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            msg = ev.get("message") or {}
            model = msg.get("model") or model
            for src in (msg.get("usage"), ev.get("usage")):  # message_start, then cumulative message_delta
                if src:
                    usage.update({k: v for k, v in src.items() if v is not None})
    else:
        try:
            doc = json.loads(body or b"{}")
            usage, model = doc.get("usage") or {}, doc.get("model")
        except json.JSONDecodeError:
            pass
    return usage, model


def make_handler(cfg: dict, key: str, wire_path: str, workspace: str = ""):
    """Build the request handler class bound to this run's config."""
    up = urlsplit(cfg.get("upstream", "https://api.anthropic.com"))

    def wire(text: str) -> None:
        with LOCK, open(wire_path, "a", encoding="utf-8") as f:
            f.write(text)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def address_string(self) -> str:  # unix sockets have no peer address
            return "sandbox"

        def log_message(self, fmt: str, *args) -> None:
            print(f"{time.strftime('%H:%M:%S')} {fmt % args}", flush=True)

        def _error(self, status: int, message: str) -> None:
            body = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": message}}).encode()
            wire(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} PROXY REFUSED {status}: {message}\n")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — http.server naming
            self._error(403, f"endpoint not allowed in this sandbox: GET {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            if not path.startswith("/v1/messages"):
                return self._error(403, f"endpoint not allowed in this sandbox: {path}")
            try:
                req = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._error(400, "request body is not JSON")
            model = req.get("model")
            if model not in cfg["models"]:
                return self._error(403, f"model not allowed in this run: {model}")
            bad = [t.get("type") for t in req.get("tools") or [] if any(b in str(t.get("type", "")) for b in BLOCKED_TOOL_TYPES)]
            if bad or req.get("mcp_servers"):
                return self._error(403, f"server-side tools with internet access are not allowed: {bad or 'mcp_servers'}")
            counting = path.endswith("/count_tokens")  # free endpoint
            if not counting:
                total, run = spent(cfg["ledger"]), spent(cfg["ledger"], cfg["run_id"])
                if total + cfg["reserve_usd"] > cfg["cap_usd"] or run + cfg["reserve_usd"] > cfg["run_cap_usd"]:
                    return self._error(429, f"budget exhausted: ${total:.2f} of ${cfg['cap_usd']:.2f} total, "
                                            f"${run:.2f} of ${cfg['run_cap_usd']:.2f} for this run")
            wire(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} REQUEST POST {path}\n{body.decode('utf-8', 'replace')}\n")
            headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP_HEADERS}
            headers.update({"x-api-key": key, "accept-encoding": "identity", "content-length": str(len(body))})
            if workspace:  # required for keys not scoped to a workspace
                headers["anthropic-workspace-id"] = workspace
            conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
            kw = {"context": ssl.create_default_context()} if up.scheme == "https" else {}
            try:
                conn = conn_cls(up.hostname, up.port or (443 if up.scheme == "https" else 80), timeout=900, **kw)
                conn.request("POST", self.path, body, headers)
                resp = conn.getresponse()
            except OSError as exc:
                return self._error(502, f"upstream unreachable: {exc}")
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type", "")
            for k, v in resp.getheaders():
                if k.lower() not in DROP_HEADERS | {"content-encoding"}:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            got = bytearray()
            while chunk := resp.read1(65536):
                got += chunk
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            conn.close()
            usage, resp_model = usage_from_body(bytes(got), ctype)
            cost = 0.0 if counting else price(model, usage)
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": cfg["run_id"], "model": resp_model or model,
                   "status": resp.status, "usage": usage, "cost_usd": round(cost, 6)}
            if not counting:  # recorded before the response completes, so the next request's budget check sees it
                with LOCK, open(cfg["ledger"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            wire(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} RESPONSE {resp.status} cost ${cost:.4f}\n"
                 f"{bytes(got).decode('utf-8', 'replace')}\n")

    return Handler


def serve(listen: str, cfg: dict, key: str, wire_path: str, workspace: str = ""):
    """Start the proxy on "unix:/path" or "tcp:HOST:PORT" in a background thread; return the server."""
    handler = make_handler(cfg, key, wire_path, workspace)
    if listen.startswith("unix:"):
        srv = socketserver.ThreadingUnixStreamServer(listen[5:], handler)
    else:
        host, port = listen[4:].rsplit(":", 1)
        srv = socketserver.ThreadingTCPServer((host, int(port)), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    listen, cfg_path, wire_path = sys.argv[1:4]
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    key = os.environ.pop("VMI_ANTHROPIC_KEY", "")
    workspace = os.environ.pop("VMI_ANTHROPIC_WORKSPACE_ID", "")
    if not key:
        sys.exit("VMI_ANTHROPIC_KEY is not set")
    serve(listen, cfg, key, wire_path, workspace)
    print(f"proxy listening on {listen}; spent so far ${spent(cfg['ledger']):.4f} of ${cfg['cap_usd']}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
