"""The run matrix after amendment 2: 44 conditions per round, Haiku main simulator, local Qwens as the capability ablation."""

from vmi.manifests import LOCAL_MODELS, MAIN, ROUND_SEEDS, STAGES, conditions


def test_matrix_counts_and_simulators():
    per_stage = {s: conditions(s) for s in STAGES}
    assert {s: len(c) for s, c in per_stage.items()} == {"main": 4, "controls": 5, "ablations": 15, "hyper": 4, "seed": 16}
    rows = [c for cs in per_stage.values() for c in cs]
    local = [c for c in rows if c.get("harness") == "mini"]
    assert len(local) == 12 and {c["model"] for c in local} == set(LOCAL_MODELS)
    assert all(c["model"] == MAIN["model"] and c["harness"] == "claudecode" for c in rows if c not in local)
    assert {c["effort"] for c in per_stage["hyper"]} == {"low", "high"}  # effort replaces temperature
    assert sum(len(s) for s in ROUND_SEEDS.values()) == 10  # N = 1, 3, 10
