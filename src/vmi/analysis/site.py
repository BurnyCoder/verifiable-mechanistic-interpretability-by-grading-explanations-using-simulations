"""Data file for the interactive GitHub Pages site (docs/index.html reads docs/data.json).

Local explanation: collects every graded agent run (all stages and targets), the reference compiler's scores, the
no-agent reference points, the hypothesis verdicts and the key-results text into one JSON file; the page draws its
charts and tables from it, so the site, like README and paper, shows only numbers read from results/.
Global context: written by `vmi analyse` after the hypotheses and tables.
"""

from __future__ import annotations  # postponed annotations

import json  # results
import math  # NaN filtering

from vmi.analysis import tables  # key results, hypotheses
from vmi.config import REPO_ROOT  # paths

RES = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "data.json"
KEYS = ("hs", "kl_out", "accuracy", "matched_neuron_corr", "centred_logit_fvu", "calibrated_kl")


def _clean(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def build(max_round: int) -> dict:
    runs = []
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("stage") == "dev":
            continue
        runs.append({"run_id": r["run_id"], "stage": r["stage"], "target": r["target"], "level": r["level"],
                     "variant": r.get("variant", ""), "model": r["model"], "seed": r["seed"], "round": r.get("round"),
                     "failed": r.get("failed", False), "cost_usd": r.get("agent", {}).get("cost_usd"),
                     "seconds": r.get("agent", {}).get("seconds"), **{k: _clean(r.get(k)) for k in KEYS}})
    compiler = []
    for d in sorted((RES / "compiler").glob("*")):
        for p in sorted(d.glob("*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            compiler.append({"target": d.name, "level": p.stem, **{k: _clean(c.get(k)) for k in KEYS}})
    refs = []
    for t in ("T0",):
        b = RES / "baselines" / f"{t}.json"
        ri = RES / "controls" / f"random_init_{t}.json"
        if ri.exists():
            refs += [{"target": t, "kind": "random init", "hs": x["hs"], "kl_out": x["kl_out"]} for x in json.loads(ri.read_text())]
        if b.exists():
            d = json.loads(b.read_text())
            refs += [{"target": t, "kind": "independent retrain", "hs": x["hs"], "kl_out": x["kl_out"]} for x in d["retrains"]]
            refs += [{"target": t, "kind": "distilled student", "hs": x["hs"], "kl_out": x["kl_out"]} for x in d["students"]]
    hyp = RES / f"hypotheses_round{max_round}.json"
    H = json.loads(hyp.read_text())["hypotheses"] if hyp.exists() else {}
    verdicts = [{"id": k, "verdict": v.get("verdict") or "; ".join(f"{a}: {b}" for a, b in v.get("verdicts", {}).items()),
                 "evidence": str(v.get("label", v.get("N", "")))} for k, v in H.items()]
    data = {"round": max_round, "runs": runs, "compiler": compiler, "references": refs, "hypotheses": verdicts,
            "key_results_md": tables.key_results(max_round)}
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    return data
