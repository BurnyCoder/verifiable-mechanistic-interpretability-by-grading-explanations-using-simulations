"""Pre-push secret scan: fail if any tracked or staged file (or any path given) contains an API key or token.

Local explanation: looks for the exact ANTHROPIC_API_KEY / HF_TOKEN values from .env (if present) and for the
formats of real Anthropic keys (sk-ant-api03-…, sk-ant-admin01-…) and Hugging Face tokens (hf_ + 34 characters).
Global context: run before every push (`python scripts/check_secrets.py`); Claude runs pass the key only to the
proxy outside the sandbox, and this check makes sure it never reaches GitHub or Hugging Face through a log or result.
Usage: python scripts/check_secrets.py [PATH ...]   (default: every file git tracks or has staged)
"""

import re
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(rb"sk-ant-(?:api|admin)\d\d-[A-Za-z0-9_-]{20,}"), re.compile(rb"\bhf_[A-Za-z0-9]{34}\b")]


def secrets_in(path: Path, exact: list[bytes]) -> list[str]:
    """Names of the secret kinds found in one file."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    found = [f"exact value #{i}" for i, s in enumerate(exact) if s and s in data]
    found += [p.pattern.decode()[:20] for p in PATTERNS if p.search(data)]
    if path.suffix == ".gz":  # compressed logs
        import gzip
        try:
            return found + secrets_in_bytes(gzip.decompress(data), exact)
        except OSError:
            pass
    return found


def secrets_in_bytes(data: bytes, exact: list[bytes]) -> list[str]:
    return [f"exact value #{i}" for i, s in enumerate(exact) if s and s in data] + [
        p.pattern.decode()[:20] for p in PATTERNS if p.search(data)]


def main(argv: list[str]) -> int:
    env = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    exact = [v.encode() for k, v in env.items() if v and len(v) >= 16 and ("KEY" in k or "TOKEN" in k)]
    if argv:
        files = [Path(a) for a in argv]
    else:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines() if f and f != ".env"]
    bad = {str(f): kinds for f in files if f.is_file() and (kinds := secrets_in(f, exact))}
    for f, kinds in bad.items():
        print(f"SECRET FOUND: {f}: {kinds}")
    print(f"checked {len(files)} files: {'FAIL' if bad else 'ok'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
