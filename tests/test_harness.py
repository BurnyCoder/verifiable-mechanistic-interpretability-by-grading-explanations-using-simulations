"""The custom harness runs one bash action per reply, keeps truncated replies, rejects bad formats and stops on submit.

Global context: the harness replaces OpenCode if the stage-0 agent stays at chance (preregistration amendment 2); this
test drives it against a scripted fake chat-completions server, so no LLM is needed.
"""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vmi.simulator import harness

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

REPLIES = [  # (finish_reason, content); "" content = reply only in reasoning_content (unclosed thinking tag)
    ("stop", ""),
    ("stop", "THOUGHT: make a file.\n\n```bash\necho hello > out.txt && cat out.txt\n```"),
    ("length", "THOUGHT: a very long design that was cut off"),
    ("stop", "two blocks\n```bash\nls\n```\n```bash\nls\n```"),
    ("stop", "THOUGHT: done.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"),
]


def _server(requests: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server naming
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            reason, content = REPLIES[len(requests) - 1]
            thinking = "THOUGHT: look.\n\n```bash\nls\n```" if not content else "thinking"
            out = json.dumps({"choices": [{"finish_reason": reason, "message": {"content": content,
                              "reasoning_content": thinking}}], "usage": {"prompt_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_fit_context_elides_oldest_middle_messages():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}] + [
        {"role": "assistant" if i % 2 else "user", "content": f"m{i}"} for i in range(10)]
    cfg = {"context_tokens": 1000, "max_tokens": 200}
    assert harness.fit_context(msgs, cfg, last_total=100) == 0  # far from the 800-token input limit
    n = harness.fit_context(msgs, cfg, last_total=790)
    assert n == 2 and msgs[2]["content"] == msgs[3]["content"] == harness.ELIDED
    assert msgs[1]["content"] == "task" and [m["content"] for m in msgs[-6:]] == [f"m{i}" for i in range(4, 10)]


def test_harness_loop(tmp_path, capsys):
    requests: list = []
    srv = _server(requests)
    cfg = {"url": f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", "model": "m", "max_tokens": 100,
           "sampling": {"temperature": 0.6}, "step_limit": 10, "time_limit_s": 60, "request_timeout_s": 10,
           "command_timeout_s": 10, "cwd": str(tmp_path)}
    status = harness.run(cfg, "Do the task.")
    srv.shutdown()
    assert status == "Submitted" and len(requests) == 5
    assert (tmp_path / "out.txt").read_text().strip() == "hello"
    assert "<returncode>0</returncode>" in requests[1]["messages"][-1]["content"]  # the reasoning-only reply ran
    second = requests[2]["messages"]
    assert second[-1]["role"] == "user" and "hello" in second[-1]["content"]  # observation as a user message
    assert all("reasoning_content" not in m for m in second)  # old thinking is not sent back
    third = requests[3]["messages"]
    assert third[-2] == {"role": "assistant", "content": REPLIES[2][1]} and "cut off" in third[-1]["content"]
    assert "Format error" in requests[4]["messages"][-1]["content"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    kinds = [e["type"] for e in events]
    assert kinds.count("step_finish") == 5 and kinds.count("tool_use") == 3
    assert {"output-limit", "format"} <= {e["part"]["error"]["type"] for e in events if e["type"] == "error"}
