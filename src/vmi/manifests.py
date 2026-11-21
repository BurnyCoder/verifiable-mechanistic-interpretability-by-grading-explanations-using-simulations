"""The preregistered agent conditions, stage by stage, and the run manifests for each round (N = 1, 3, 10).

Local explanation: `conditions(stage)` lists the agent conditions of a stage exactly as in reports/planned-runs.md;
`write_round(r)` writes one JSONL manifest per stage for round r with the round's run seeds (round 1: seed 0;
round 2: seeds 1–2; round 3: seeds 3–9). The queue skips runs whose result file exists, so manifests are resumable.
Global context: stages run in the preregistered order main → controls → (baselines, no agent) → ablations →
hyperparameter sensitivity → seed sensitivity within every round.
"""

from __future__ import annotations  # postponed annotations

import json  # JSONL manifests
from pathlib import Path  # output paths

from vmi.config import REPO_ROOT  # results folder

STAGES = ("main", "controls", "ablations", "hyper", "seed")  # agent stages in order (baselines have no agent runs)
ROUND_SEEDS = {1: [0], 2: [1, 2], 3: [3, 4, 5, 6, 7, 8, 9]}
# Amendment 2: the main simulator is Claude Haiku 5.5 in Claude Code (the only simulator above the stage-0 rule); the
# local Qwen models (custom harness, llama-server aliases) form the simulator-capability ablation
MAIN = {"model": "claude-haiku-5-5", "harness": "claudecode"}
LOCAL_MODELS = ("qwen3.6-35b-a3b", "qwen3.5-9b", "qwen3.5-4b")
LOCAL_ROUNDS = (1,)  # deviation 5: all local runs were at chance, so the user stopped them; rounds 2-3 are Claude-only


def conditions(stage: str, round_: int = 1) -> list[dict]:
    """Agent conditions of one stage (without seeds); local Qwen conditions exist only in LOCAL_ROUNDS."""
    if stage == "main":
        return [{"stage": "main", "target": "T0", "level": lv, **MAIN} for lv in ("E1", "E2", "E3", "E4")]
    if stage == "controls":
        return [{"stage": "controls", "target": "T0", "level": lv, **MAIN}
                for lv in ("E0", "W-freq", "W-alg", "W-role", "W-scramble")]
    if stage == "ablations":
        out = [{"stage": "ablations", "target": "T0", "level": lv, "model": m, "harness": "mini"} for m in LOCAL_MODELS
               for lv in ("E2", "E3", "E4", "W-freq") if round_ in LOCAL_ROUNDS]
        out += [{"stage": "ablations", "target": "T0", "level": "E3", "variant": v, **MAIN} for v in ("fmt-math", "fmt-code")]
        out.append({"stage": "ablations", "target": "T0", "level": "E3", "variant": "noselftest", "selftest": False, **MAIN})
        return out
    if stage == "hyper":  # Claude 5.5 rejects non-default sampling, so effort replaces temperature (amendment 2)
        return [{"stage": "hyper", "target": "T0", "level": lv, "variant": f"effort-{e}", "effort": e, **MAIN}
                for e in ("low", "high") for lv in ("E3", "W-freq")]
    if stage == "seed":
        out = [{"stage": "seed", "target": t, "level": lv, **MAIN} for t in ("T1", "T2") for lv in ("E2", "E3", "E4", "W-freq")]
        out += [{"stage": "seed", "target": "T-new", "level": lv, **MAIN} for lv in ("E3", "E4", "W-freq")]
        out += [{"stage": "seed", "target": "T-paper", "level": lv, **MAIN} for lv in ("E-paper", "E1", "E3", "E4", "W-freq")]
        return out
    raise ValueError(stage)


def write_round(r: int, out_dir: Path | None = None) -> list[Path]:
    """Write results/manifests/round{r}_{stage}.jsonl for every agent stage; return the paths."""
    out_dir = out_dir or REPO_ROOT / "results" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in STAGES:
        p = out_dir / f"round{r}_{stage}.jsonl"
        rows = [{**c, "seed": s, "round_": r} for s in ROUND_SEEDS[r] for c in conditions(stage, r)]
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8", newline="\n")
        paths.append(p)
    return paths
