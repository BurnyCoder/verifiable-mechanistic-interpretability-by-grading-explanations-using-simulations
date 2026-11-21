"""The Claude proxy injects the key, never logs it, prices usage into the ledger, enforces the budget and blocks web tools.

Global context: the proxy is what keeps the API key out of the sandbox and the total Claude spend under the user's
cap; this test runs it over TCP against a fake upstream (no real API call, no cost).
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vmi.simulator import anthropic_proxy as ap

KEY = "sk-ant-test-0123456789"
SSE = ("event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {
    "model": "claude-haiku-5-5", "usage": {"input_tokens": 1000, "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0, "output_tokens": 1}}}) +
       "\n\nevent: message_delta\ndata: " + json.dumps({"type": "message_delta", "usage": {"output_tokens": 2000}}) +
       "\n\n")


def _upstream(seen: list):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen.append(dict(self.headers))
            self.rfile.read(int(self.headers["Content-Length"]))
            body = SSE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(port: int, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-api-key": "placeholder"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture
def proxy(tmp_path):
    seen: list = []
    up = _upstream(seen)
    cfg = {"upstream": f"http://127.0.0.1:{up.server_port}", "models": ["claude-haiku-5-5"], "run_id": "r1",
           "ledger": str(tmp_path / "ledger.jsonl"), "cap_usd": 1.0, "run_cap_usd": 1.0, "reserve_usd": 0.0}
    srv = ap.serve("tcp:127.0.0.1:0", cfg, KEY, str(tmp_path / "wire.log"))
    yield srv.server_address[1], seen, cfg, tmp_path
    srv.shutdown()
    up.shutdown()


def test_forwards_with_key_and_prices_usage(proxy):
    port, seen, cfg, tmp = proxy
    status, body = _post(port, {"model": "claude-haiku-5-5", "messages": []})
    assert status == 200 and "message_delta" in body
    assert seen[0]["x-api-key"] == KEY  # real key upstream, placeholder replaced
    row = json.loads((tmp / "ledger.jsonl").read_text().splitlines()[0])
    assert row["cost_usd"] == pytest.approx((1000 * 0.10 + 2000 * 0.50) / 1e6)
    assert KEY not in (tmp / "wire.log").read_text()


def test_refuses_over_budget_wrong_model_and_web_tools(proxy):
    port, seen, cfg, tmp = proxy
    (tmp / "ledger.jsonl").write_text(json.dumps({"run_id": "old", "cost_usd": 1.5}) + "\n")
    assert _post(port, {"model": "claude-haiku-5-5", "messages": []})[0] == 429
    (tmp / "ledger.jsonl").write_text("")
    assert _post(port, {"model": "claude-opus-5-5", "messages": []})[0] == 403
    assert _post(port, {"model": "claude-haiku-5-5", "messages": [],
                        "tools": [{"type": "web_search_20260209", "name": "web_search"}]})[0] == 403
    assert seen == []  # nothing reached the upstream


def test_price_tiers_and_cache():
    assert ap.price("claude-haiku-5-5", {"input_tokens": 150_000, "output_tokens": 0}) == pytest.approx(0.075)
    u = {"input_tokens": 0, "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000,
         "output_tokens": 1_000_000}
    assert ap.price("claude-opus-5-5", u) == pytest.approx(0.20 + 1.25 * 4.0 + 20.0)
