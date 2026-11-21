"""Pipeline entry point: `uv run vmi <command>` — explain, compile, run agents, grade, analyse.

Local explanation (commands):
  explain TARGET            measure TARGET's mechanism and write every explanation level to results/explanations/
  compile TARGET            reference-compile every level for TARGET and grade it (agent-free baseline)
  check                     isolation checks inside the sandbox (llama-server must be running for the last check)
  run TARGET LEVEL SEED     one agent run (prepare sandbox → OpenCode offline → audit → grade → results/runs/)
  queue MANIFEST            run every condition of a JSONL manifest that has no result yet (resumable)
  serve [--n-cpu-moe N]     start llama-server and keep it running
Global context: every number in the paper comes from the JSON files these commands write under results/.
Sources: argparse (https://docs.python.org/3/library/argparse.html); logging (https://docs.python.org/3/library/logging.html).
"""

from __future__ import annotations  # postponed annotations

import argparse  # CLI
import json  # results files
import logging  # timestamped logs
import sys  # stdout handler
import time  # timestamps
from pathlib import Path  # paths

from vmi.config import REPO_ROOT, Settings, load  # settings

RESULTS = REPO_ROOT / "results"
log = logging.getLogger("vmi")


def setup_logging(name: str) -> Path:
    """Log to the terminal and to logs/<name>_<timestamp>.log."""
    path = REPO_ROOT / "logs" / f"{name}_{time.strftime('%Y%m%dT%H%M%S')}.log"
    path.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)
    return path


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8", newline="\n")  # LF, as in git


def explain(settings: Settings, target: str) -> dict:
    """Analyse a target and write analysis.json plus every level's EXPLANATION.md, tables, spec and length."""
    from vmi import targets
    from vmi.explainer.analysis import analyse
    from vmi.explainer.ladder import LEVELS, make_spec
    from vmi.explainer.render import length_stats, render
    from vmi.grader.score import device

    dev = device()
    an = analyse(targets.load_model(settings, target, dev), targets.grid(target, dev))
    out = RESULTS / "explanations" / target
    _dump(out / "analysis.json", an)
    log.info("%s key frequencies %s, logit FVE %.3f, clusters %s", target, an["key_freqs"], an["logit_fit_fve"], an["neuron_clusters"])
    for level in LEVELS:
        spec = make_spec(an, level, target)
        text, extra = render(spec)
        d = out / level
        d.mkdir(parents=True, exist_ok=True)
        (d / "EXPLANATION.md").write_text(text, encoding="utf-8", newline="\n")
        for name, content in extra.items():
            (d / name).write_text(content, encoding="utf-8", newline="\n")
        _dump(d / "spec.json", spec)
        _dump(d / "length.json", length_stats(text, extra))
    if target == "T-paper":  # the published explanation, as written (numbers from the paper, not measured here)
        from vmi.explainer import paper
        d = out / "E-paper"
        d.mkdir(parents=True, exist_ok=True)
        (d / "EXPLANATION.md").write_text(paper.text(), encoding="utf-8", newline="\n")
        _dump(d / "spec.json", paper.spec())
        _dump(d / "length.json", length_stats(paper.text(), {}))
    from vmi.explainer.render import render_format
    for fmt in ("math", "code"):  # format ablation: E3 content in other notations
        spec = make_spec(an, "E3", target)
        text, extra = render_format(spec, fmt)
        d = out / f"E3-fmt-{fmt}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "EXPLANATION.md").write_text(text, encoding="utf-8", newline="\n")
        _dump(d / "spec.json", {**spec, "format": fmt})
        _dump(d / "length.json", length_stats(text, extra))
    return an


def load_explanation(target: str, level: str) -> tuple[dict, str, dict[str, str]]:
    """(spec, EXPLANATION.md text, extra files) for a target/level written by `explain`."""
    d = RESULTS / "explanations" / target / level
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    extra = {p.name: p.read_text(encoding="utf-8") for p in d.iterdir() if p.suffix == ".csv"}
    return spec, (d / "EXPLANATION.md").read_text(encoding="utf-8"), extra


def compile_and_grade(settings: Settings, target: str, levels: list[str] | None = None) -> dict:
    """Reference-compile each level's spec, grade against the target, write results/compiler/<target>/<level>.json."""
    import numpy as np

    from vmi import targets, weights
    from vmi.explainer.ladder import LEVELS
    from vmi.grader.score import grade_models
    from vmi.reference.compile import compile_spec

    model = targets.load_model(settings, target)
    out = {}
    if levels is None:
        levels = list(LEVELS) + (["E-paper"] if target == "T-paper" else [])
    for level in levels:
        spec, _, _ = load_explanation(target, level)
        arrays = compile_spec(spec)
        res = grade_models(model, weights.to_model(weights.validate(arrays)))
        res.update({"target": target, "level": level, "kind": "reference_compiler"})
        _dump(RESULTS / "compiler" / target / f"{level}.json", res)
        wdir = settings.runs_dir / "_compiler" / target  # compiled weights (large; published to HF, not git)
        wdir.mkdir(parents=True, exist_ok=True)
        np.savez(wdir / f"{level}.npz", **arrays)
        log.info("compiler %s %-10s acc %.4f KL %.3g HS %.3f", target, level, res["accuracy"], res["kl_out"], res["hs"])
        out[level] = res
    return out


def run_id(stage: str, target: str, level: str, seed: int, model: str, temperature: float, variant: str = "") -> str:
    """Stable identifier of one agent run (also its folder and result-file name)."""
    tag = model.replace(".", "").replace("-", "")
    return f"{stage}__{target}__{level}{('-' + variant) if variant else ''}__{tag}__t{temperature:g}__s{seed}"


def agent_run(settings: Settings, *, stage: str, target: str, level: str, seed: int, model: str | None = None,
              temperature: float | None = None, selftest: bool = True, variant: str = "", round_: int = 0) -> dict:
    """One full agent run: prepare → run offline in the sandbox → audit → grade → results/runs/<id>.json."""
    from vmi import targets
    from vmi.grader.score import grade_npz
    from vmi.simulator.audit import audit
    from vmi.simulator.runner import run_agent
    from vmi.simulator.sandbox import prepare_run

    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature
    rid = run_id(stage, target, level, seed, model, temperature, variant)
    result_file = RESULTS / "runs" / f"{rid}.json"
    if result_file.exists():
        log.info("skip %s (result exists)", rid)
        return json.loads(result_file.read_text(encoding="utf-8"))
    exp_level = f"{level}-{variant}" if variant.startswith("fmt") else level  # format ablation uses its own text
    spec, text, extra = load_explanation(target, exp_level)
    run_dir = settings.runs_dir / rid
    meta = {"run_id": rid, "stage": stage, "target": target, "level": level, "variant": variant, "round": round_}
    prepare_run(settings, run_dir, explanation=text, extra_files=extra, meta=meta, model=model,
                temperature=temperature, seed=seed, selftest=selftest)
    info = run_agent(settings, run_dir)
    built = run_dir / "work_out" / "build_weights.py"
    # an untouched starter writes all-zero weights; record it so such runs are not mistaken for agent constructions
    info["starter_unchanged"] = built.exists() and built.read_bytes() == (run_dir / "work" / "build_weights.py").read_bytes()
    au = audit(built)
    res = grade_npz(targets.load_model(settings, target), run_dir / "work_out" / "weights.npz")
    length = json.loads((RESULTS / "explanations" / target / exp_level / "length.json").read_text())
    res.update({**meta, "model": model, "temperature": temperature, "seed": seed, "selftest": selftest,
                "agent": info, "audit": au, "length": length, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _dump(result_file, res)
    log.info("run %s: failed=%s acc=%.4f KL=%.3g HS=%.3f audit=%s steps=%s %ss", rid, res["failed"], res["accuracy"],
             res["kl_out"], res["hs"], au["flagged"], info.get("steps"), info.get("seconds"))
    return res


def queue(settings: Settings, manifest: Path) -> None:
    """Run every condition in a JSONL manifest (one JSON object of agent_run kwargs per line) without a result yet."""
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            agent_run(settings, **json.loads(line))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vmi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("explain").add_argument("target")
    sub.add_parser("compile").add_argument("target")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("target")
    r.add_argument("level")
    r.add_argument("seed", type=int)
    r.add_argument("--stage", default="dev")
    q = sub.add_parser("queue")
    q.add_argument("manifest", type=Path)
    s = sub.add_parser("serve")
    s.add_argument("--n-cpu-moe", type=int, default=None)
    s.add_argument("--gguf", type=Path, default=None)
    s.add_argument("--alias", default=None)
    m = sub.add_parser("manifests")
    m.add_argument("round", type=int)
    c = sub.add_parser("controls")
    c.add_argument("round", type=int)
    b = sub.add_parser("baselines")
    b.add_argument("target")
    an = sub.add_parser("analyse")
    an.add_argument("round", type=int)
    a = ap.parse_args(argv)
    setup_logging(a.cmd)
    settings = load()
    if a.cmd == "explain":
        explain(settings, a.target)
    elif a.cmd == "compile":
        compile_and_grade(settings, a.target)
    elif a.cmd == "check":
        from vmi.simulator.runner import isolation_check
        lines = isolation_check(settings, settings.runs_dir / "_checks")
        _dump(RESULTS / "isolation" / f"check_{time.strftime('%Y%m%dT%H%M%S')}.json", lines)
        bad = [x for x in lines if x.startswith("FAIL")]
        log.info("isolation check: %d ok, %d FAIL %s", len(lines) - len(bad), len(bad), bad)
    elif a.cmd == "run":
        agent_run(settings, stage=a.stage, target=a.target, level=a.level, seed=a.seed)
    elif a.cmd == "queue":
        queue(settings, a.manifest)
    elif a.cmd == "serve":
        from vmi.simulator import llm_server
        proc = llm_server.start(settings, n_cpu_moe=a.n_cpu_moe, gguf=a.gguf, alias=a.alias)
        log.info("llama-server healthy (pid %d); Ctrl+C to stop", proc.pid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            llm_server.stop(proc)
    elif a.cmd == "manifests":
        from vmi.manifests import write_round
        for p in write_round(a.round):
            log.info("wrote %s", p)
    elif a.cmd == "controls":  # no-agent controls of the controls stage (planted, random init, wrong targets, trajectory)
        from vmi import nonagent
        if not (RESULTS / "controls" / "planted_calibration_T0.json").exists():
            nonagent.planted(settings)
        if not (RESULTS / "controls" / "random_init_T0.json").exists():
            nonagent.random_init(settings)
        nonagent.wrong_targets(settings, a.round)
        nonagent.trajectory(settings, a.round)
    elif a.cmd == "baselines":
        from vmi import nonagent
        nonagent.baselines(settings, a.target)
    elif a.cmd == "analyse":
        from vmi.analysis.hypotheses import write
        log.info("wrote %s", write(a.round))


if __name__ == "__main__":
    main()
