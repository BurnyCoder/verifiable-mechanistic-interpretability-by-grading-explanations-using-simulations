"""Exercise the CLI against recorded data with model execution, weight loading and network connections disabled."""

import hashlib
import json
import socket
from contextlib import ExitStack
from unittest.mock import patch

import torch

from vmi import pipeline, targets, weights
from vmi.analysis import examples
from vmi.config import REPO_ROOT


def verify():
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("model, weights or network access")
        raise AssertionError(attempts[-1])

    def read(path):
        return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))

    raw = [read(f.relative_to(REPO_ROOT)) for f in (REPO_ROOT / "results/runs").glob("*.json")]
    paper = REPO_ROOT / "reports/paper/paper.md"
    with ExitStack() as guard:
        for obj, name in (
            (torch.nn.Module, "_call_impl"),
            (torch.optim.Optimizer, "__init__"),
            (torch, "load"),
            (weights, "load_npz"),
            (weights, "to_model"),
            (targets, "load_model"),
            (examples, "panel_models"),
            (examples, "output_rows"),
            (pipeline, "agent_run"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
        ):
            guard.enter_context(patch.object(obj, name, forbidden))
        before = hashlib.sha256(paper.read_bytes()).digest()
        for n, count in ((1, 37), (2, 101)):
            pipeline.main(["analyse", str(n), "--saved-only"])
            site = read("docs/data_round1.json" if n == 1 else "docs/data.json")
            expected = {r["run_id"] for r in raw if r["stage"] != "dev" and 1 <= r["round"] <= n}
            actual = {r["run_id"] for r in site["runs"]}
            assert actual == expected and len(actual) == count
            assert all("h1_kl" in row for row in site["replications"])
            if n == 1:
                assert hashlib.sha256(paper.read_bytes()).digest() == before
    audit = read("results/scientific-audit/reconstruction.json")
    assert audit["inventory"]["maximum_hs_reconstruction_error"] < 1.2e-16
    assert audit["E3_Walg_complete_separation"] and audit["fold_overlap"]["fraction"] > 0.8
    for name in (
        "reports/paper/paper.md",
        "reports/summary.md",
        "reports/experiments/01-round1/experiment.md",
        "reports/experiments/02-round2/experiment.md",
    ):
        contents = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "H1 decreasing KL" in contents
        assert "interval fails, upper-bound reading passes (E2 missing" in contents
    assert "0.0065" in next(
        l for l in paper.read_text(encoding="utf-8").splitlines() if l.startswith("| T0 W-role, reference compiler")
    )
    assert not attempts
    print("Both CLI rounds reproduced their exact run-ID sets. Model, weight and network calls: 0.")


if __name__ == "__main__":
    verify()
