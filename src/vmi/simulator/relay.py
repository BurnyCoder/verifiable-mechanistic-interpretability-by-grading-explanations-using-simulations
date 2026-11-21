"""Byte relay between a TCP port and a unix-domain socket (stdlib only; runs with WSL's system python3).

Local explanation: the agent sandbox has its own empty network namespace, so the agent cannot reach the local LLM
server's TCP port. Two relays bridge exactly one path through a unix socket file that is bind-mounted into the
sandbox:
  outside: `relay.py unix-to-tcp SOCK 127.0.0.1 8080 WIRELOG`  (socket file → llama-server on the Windows host)
  inside:  `relay.py tcp-to-unix 127.0.0.1 8080 SOCK`           (agent's http://127.0.0.1:8080 → socket file)
The outside relay can append every byte in both directions to WIRELOG (the full LLM prompts and completions).
Global context: no other network path exists inside the sandbox, so the only thing the agent can talk to is the LLM.
Sources: asyncio streams (https://docs.python.org/3/library/asyncio-stream.html); the bubblewrap + unix-socket bridge
pattern is the one Claude Code's sandbox uses on Linux/WSL2 (https://code.claude.com/docs/en/sandboxing).
"""

import asyncio
import sys
import time


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, log, tag: bytes) -> None:
    """Copy bytes from reader to writer until EOF, optionally appending them to the wire log with a direction tag."""
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            if log is not None:
                log.write(b"\n=== " + tag + b" " + time.strftime("%Y-%m-%dT%H:%M:%S").encode() + b" ===\n" + data)
                log.flush()
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 — closing a dead socket may raise; nothing to do
            pass


async def main() -> None:
    mode = sys.argv[1]
    log = None
    if mode == "unix-to-tcp":
        sock, host, port = sys.argv[2], sys.argv[3], int(sys.argv[4])
        log = open(sys.argv[5], "ab") if len(sys.argv) > 5 else None

        async def handle(r, w):
            ur, uw = await asyncio.open_connection(host, port)
            await asyncio.gather(_pipe(r, uw, log, b"REQUEST"), _pipe(ur, w, log, b"RESPONSE"))

        server = await asyncio.start_unix_server(handle, path=sock)
    elif mode == "tcp-to-unix":
        host, port, sock = sys.argv[2], int(sys.argv[3]), sys.argv[4]

        async def handle(r, w):
            ur, uw = await asyncio.open_unix_connection(sock)
            await asyncio.gather(_pipe(r, uw, None, b""), _pipe(ur, w, None, b""))

        server = await asyncio.start_server(handle, host, port)
    else:
        raise SystemExit(f"unknown mode {mode}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
