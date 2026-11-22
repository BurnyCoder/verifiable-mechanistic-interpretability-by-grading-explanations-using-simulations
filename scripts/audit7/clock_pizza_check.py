"""Audit 7: recompute Zhong et al.'s distance irrelevance for the targets and the T0 builds with this audit's forward pass.

Local explanation: q = mean over differences d of std_a L[a, a + d] / std of L over all inputs, where L[a, b] is the
logit of the correct answer (a + b) mod 113 (Zhong et al. 2023, Def. 4.2; §4.1.2: 0–0.4 typical of Pizza, 0.4–1 of
Clock). Networks: the five targets, the nine T0 reference-compiler builds and the 27 T0 main/controls agent runs; the
forward pass is scripts/audit7/regrade.py's numpy implementation of ARCHITECTURE.md. Output is compared with
results/controls/clock_pizza.json (the project's torch implementation in src/vmi/analysis/clock_pizza.py).
Global context: seventh audit (2026-11-22). The paper's post-hoc reading "T0 is a Clock, but the replacements built
from the true explanations are Pizza-like" rests on these values; this checks them with code that shares nothing with
the project's except checkpoint loading.
Sources: Zhong et al. 2023, https://arxiv.org/abs/2306.17844 (Def. 4.2, §4.1.2).
"""

from __future__ import annotations  # postponed annotations

import json  # results files
import sys  # sibling import
from pathlib import Path  # paths

import numpy as np  # std

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regrade import P, RES, forward  # noqa: E402  this audit's ARCHITECTURE.md forward pass


def distance_irrelevance(logits: np.ndarray) -> float:
    """Def. 4.2 from the [p², p] number logits (rows a·p + b)."""
    a, b = np.divmod(np.arange(P * P), P)
    L = logits[np.arange(P * P), (a + b) % P].reshape(P, P)  # correct-answer logit L[a, b]
    i = np.arange(P)
    return float(np.mean([L[i, (i + d) % P].std() for d in range(P)]) / L.std())


def main() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vmi import targets, weights  # loading only
    from vmi.config import load as load_settings
    st = load_settings()
    stored = {r["name"]: r["distance_irrelevance"] for r in json.loads((RES / "controls" / "clock_pizza.json").read_text())["rows"]}
    rows = []
    for t in ("T0", "T1", "T2", "T-new", "T-paper"):
        rows.append({"name": t, "q": distance_irrelevance(forward(weights.from_model(targets.load_model(st, t).double()))["logits"]),
                     "stored": stored[t]})
    load = lambda p: {k: v.astype(np.float64) for k, v in np.load(p, allow_pickle=False).items()}  # noqa: E731
    for lv in ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble"):
        rows.append({"name": f"T0 {lv}", "kind": "compiler",
                     "q": distance_irrelevance(forward(load(st.runs_dir / "_compiler" / "T0" / f"{lv}.npz"))["logits"]),
                     "stored": stored[f"T0 {lv}"]})
    for p in sorted((RES / "runs").glob("*__T0__*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["stage"] in ("main", "controls") and rec["model"] == "claude-haiku-5-5":
            name = f"T0 {rec['level']} s{rec['seed']}"
            rows.append({"name": name, "kind": "agent", "level": rec["level"],
                         "q": distance_irrelevance(forward(load(st.runs_dir / rec["run_id"] / "work_out" / "weights.npz"))["logits"]),
                         "stored": stored.get(name)})
    agent_means = {lv: float(np.mean([r["q"] for r in rows if r.get("level") == lv]))
                   for lv in ("E1", "E2", "E3", "E4", "W-alg", "W-scramble")}
    return {"n": len(rows), "max_abs_diff": max(abs(r["q"] - r["stored"]) for r in rows if r["stored"] is not None),
            "targets_clock_side": all(r["q"] >= 0.4 for r in rows[:5]), "agent_means": agent_means, "rows": rows}


if __name__ == "__main__":
    res = main()
    (RES / "audit7").mkdir(exist_ok=True)
    (RES / "audit7" / "clock_pizza_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(f"{res['n']} networks; largest difference {res['max_abs_diff']:.1e}; targets on the Clock side: "
          f"{res['targets_clock_side']}; agent means {({k: round(v, 3) for k, v in res['agent_means'].items()})}")
