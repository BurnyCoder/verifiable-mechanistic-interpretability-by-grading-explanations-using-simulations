"""Concrete example networks and their outputs, next to the ground truth (README highlights, paper results).

Local explanation: `panel_models` loads the target and one network per condition: the main simulator's first
replacement for E3, E4, E1, W-freq and W-alg, one independent retrain and one random initialisation.
`output_rows` runs them on every input and returns, for a few example inputs, each network's answer and confidence
next to the true answer (a+b) mod 113. The inputs are three fixed sums (including wrap-around) plus the first inputs
on which an explanation-built network answers wrongly, so errors are shown, not hidden.
Global context: used by `figures.fig_examples` / `figures.fig_outputs` and `tables.outputs_table`; nothing here is
typed by hand, every value comes from the saved weights in the runs folder and the published target runs.
Sources: input order a·p + b and read-out at the '=' position (`grader.activations.collect`, paper code / ARENA).
"""

from __future__ import annotations  # postponed annotations

import torch  # model outputs

from vmi.analysis.hypotheses import _select, load_runs  # same run selection as the tests
from vmi.config import P  # modulus

FIXED_INPUTS = [(5, 7), (60, 80), (112, 112)]  # an ordinary sum, a wrap-around sum, the largest input
PANELS = [("E3", "true"), ("E4", "true"), ("E1", "true"), ("W-freq", "control"), ("W-alg", "control")]


def panel_models(max_round: int, target: str = "T0") -> list[dict]:
    """[{name, kind, model, hs}] for the target and one network per condition (float64, CPU)."""
    from grokviz.model import GrokTransformer

    from vmi import targets, weights
    from vmi.config import load
    from vmi.grader.score import grade_models
    settings = load()
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    M = targets.load_model(settings, target).double()
    out = [{"name": f"target {target}", "kind": "target", "model": M, "hs": None}]
    for lv, kind in PANELS:
        rs = sorted(_select(runs, target=target, level=lv), key=lambda r: r["seed"])
        npz = rs and settings.runs_dir / rs[0]["run_id"] / "work_out" / "weights.npz"
        if npz and npz.exists():
            out.append({"name": f"agent, {lv}", "kind": kind, "model": weights.to_model(weights.load_npz(npz)).double(),
                        "hs": rs[0]["hs"]})
    retrain = "wd1_s2" if targets.resolve(target).run_id != "wd1_s2" else "wd1_s3"
    for name, model in ((f"retrain ({retrain})", targets.load_model(settings, retrain).double()),
                        ("random init", GrokTransformer(seed=1000).double())):
        out.append({"name": name, "kind": "reference", "model": model, "hs": grade_models(M, model, secondary=False)["hs"]})
    for p in out:
        p["model"] = p["model"].cpu()  # grading moves models to the GPU in place
    return out


@torch.no_grad()
def output_rows(panels: list[dict], n_errors: int = 2) -> tuple[list[tuple[int, int]], dict[str, list[tuple[int, float]]], dict[str, float]]:
    """Example inputs, each network's (answer, probability) on them, and each network's accuracy on all p² inputs."""
    from vmi.grader.activations import collect
    probs = {p["name"]: torch.softmax(collect(p["model"], "cpu").logits, -1) for p in panels}
    truth = torch.tensor([(a + b) % P for a in range(P) for b in range(P)])
    acc = {k: float((v.argmax(-1) == truth).double().mean()) for k, v in probs.items()}
    inputs = list(FIXED_INPUTS)
    for p in panels:  # first wrong answer of each explanation-built network, until n_errors are shown
        if p["name"].startswith("agent") and len(inputs) < len(FIXED_INPUTS) + n_errors:
            wrong = torch.nonzero(probs[p["name"]].argmax(-1) != truth).flatten()
            if len(wrong):
                i = int(wrong[0])
                if (i // P, i % P) not in inputs:
                    inputs.append((i // P, i % P))
    answers = {k: [(int(v[a * P + b].argmax()), float(v[a * P + b].max())) for a, b in inputs] for k, v in probs.items()}
    return inputs, answers, acc
