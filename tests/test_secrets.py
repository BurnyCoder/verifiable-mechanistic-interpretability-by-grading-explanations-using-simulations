"""The pre-push secret scan flags real key formats and passes the repository itself."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "scripts" / "check_secrets.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_flags_key_formats(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("x-api-key: sk-ant-api03-" + "A" * 40 + "\n")
    assert cs.secrets_in(f, [])
    g = tmp_path / "ok.txt"
    g.write_text("placeholder sk-ant-test-0123456789 and sandbox-placeholder-not-a-key\n")
    assert not cs.secrets_in(g, [])


def test_repository_is_clean():
    assert cs.main([]) == 0
