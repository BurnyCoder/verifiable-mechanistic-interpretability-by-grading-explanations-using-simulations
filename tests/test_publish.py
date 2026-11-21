"""Hugging Face export: local identifiers are redacted and any leftover secret or identifier blocks the upload.

Global context: `vmi publish` uploads agent transcripts and wire logs publicly; these tests guard what may leave the machine.
"""

import gzip

from vmi import publish


def test_redact_replaces_home_and_windows_paths():
    text = r'ls: drwxr-xr-x 2 burny burny /home/burny/vmi-sim "C:\\Users\\Someone\\x" C:/Users/Someone/y /mnt/c/Users/Someone/z'
    out = publish._redact(text)
    assert "burny" not in out and "Someone" not in out
    assert "/home/user/vmi-sim" in out and "user user" in out
    assert r"C:\\Users\\user\\x" in out and "C:/Users/user/y" in out and "/mnt/c/Users/user/z" in out


def test_scan_finds_secrets_in_plain_and_gzipped_files(tmp_path):
    (tmp_path / "ok.md").write_text("nothing to see", encoding="utf-8")
    assert publish._scan(tmp_path) == []
    fake_key = "sk-ant-" + "api03-" + "A" * 30  # assembled at runtime so this file holds no key-shaped literal
    (tmp_path / "log.gz").write_bytes(gzip.compress(f"header x-api-key: {fake_key}".encode()))
    hits = publish._scan(tmp_path)
    assert any("log.gz" in h for h in hits)
