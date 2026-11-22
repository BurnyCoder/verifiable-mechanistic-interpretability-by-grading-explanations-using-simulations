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
import dataclasses  # per-command settings overrides (--harness)
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


def run_id(stage: str, target: str, level: str, seed: int, model: str, temperature: float, variant: str = "",
           harness: str = "opencode") -> str:
    """Stable identifier of one agent run (also its folder and result-file name); OpenCode runs carry no harness tag."""
    tag = model.replace(".", "").replace("-", "") + ("" if harness == "opencode" else f"-{harness}")
    temp = "tdef" if temperature is None else f"t{temperature:g}"  # tdef: the model's default sampling (Claude)
    return f"{stage}__{target}__{level}{('-' + variant) if variant else ''}__{tag}__{temp}__s{seed}"


def agent_run(settings: Settings, *, stage: str, target: str, level: str, seed: int, model: str | None = None,
              temperature: float | None = None, selftest: bool = True, variant: str = "", round_: int = 0,
              harness: str | None = None, effort: str | None = None) -> dict:
    """One full agent run: prepare → run offline in the sandbox → audit → grade → results/runs/<id>.json."""
    from vmi import targets
    from vmi.grader.score import grade_npz
    from vmi.simulator.audit import audit
    from vmi.simulator.runner import run_agent
    from vmi.simulator.sandbox import prepare_run

    if harness:  # per-condition harness from the manifest
        settings = dataclasses.replace(settings, harness=harness)
    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature
    if model.startswith("claude-"):  # Claude 5.5 models reject non-default sampling parameters
        temperature = None
    rid = run_id(stage, target, level, seed, model, temperature, variant, settings.harness)
    result_file = RESULTS / "runs" / f"{rid}.json"
    if result_file.exists():
        log.info("skip %s (result exists)", rid)
        return json.loads(result_file.read_text(encoding="utf-8"))
    exp_level = f"{level}-{variant}" if variant.startswith("fmt") else level  # format ablation uses its own text
    spec, text, extra = load_explanation(target, exp_level)
    run_dir = settings.runs_dir / rid
    meta = {"run_id": rid, "stage": stage, "target": target, "level": level, "variant": variant, "round": round_}
    prepare_run(settings, run_dir, explanation=text, extra_files=extra, meta=meta, model=model,
                temperature=temperature, seed=seed, selftest=selftest, effort=effort)
    mode = {"mini": "harness", "claudecode": "claude"}.get(settings.harness, "agent")
    if mode == "claude" and not model.startswith("claude-"):
        raise ValueError(f"the claudecode harness needs a Claude model id, got {model}")
    info = run_agent(settings, run_dir, mode=mode)
    if mode == "claude":  # API spend of this run, from the proxy's ledger
        from vmi.simulator.anthropic_proxy import spent
        from vmi.simulator.sandbox import CLAUDE_LEDGER
        info["cost_usd"] = round(spent(str(CLAUDE_LEDGER), rid), 4)
        info["total_claude_spend_usd"] = round(spent(str(CLAUDE_LEDGER)), 4)
    built = run_dir / "work_out" / "build_weights.py"
    # an untouched starter writes all-zero weights; record it so such runs are not mistaken for agent constructions
    info["starter_unchanged"] = built.exists() and built.read_bytes() == (run_dir / "work" / "build_weights.py").read_bytes()
    au = audit(built)
    res = grade_npz(targets.load_model(settings, target), run_dir / "work_out" / "weights.npz")
    length = json.loads((RESULTS / "explanations" / target / exp_level / "length.json").read_text())
    res.update({**meta, "model": model, "temperature": temperature, "seed": seed, "selftest": selftest,
                "harness": settings.harness, "effort": effort or ("medium" if mode == "claude" else None),
                "agent": info, "audit": au, "length": length, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _dump(result_file, res)
    log.info("run %s: failed=%s acc=%.4f KL=%.3g HS=%.3f audit=%s steps=%s %ss", rid, res["failed"], res["accuracy"],
             res["kl_out"], res["hs"], au["flagged"], info.get("steps"), info.get("seconds"))
    return res


LOCAL_SERVE = {  # alias → (GGUF file in VMI_MODELS_DIR, --n-cpu-moe); stage-0 settings (amendment 1, learnings.md)
    "qwen3.5-4b": ("Qwen3.5-4B-Q4_K_M.gguf", 0),  # fits on the GPU entirely
    "qwen3.5-9b": ("Qwen3.5-9B-Q4_K_M.gguf", 0),
    "qwen3.6-35b-a3b": (None, None),  # None → VMI_LLM_GGUF and VMI_LLM_N_CPU_MOE (experts in system RAM)
}


def local_queue(settings: Settings, manifest: Path) -> None:
    """Serve each local model in turn (smallest first), run its pending conditions of `manifest`, stop the server.

    Lets the capability ablation run unattended across model switches; a model with nothing pending is not served."""
    from vmi.simulator import llm_server
    conds = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    for alias, (gguf, n_cpu_moe) in LOCAL_SERVE.items():
        pending = [c for c in conds if c.get("model") == alias and not (RESULTS / "runs" / (run_id(
            c["stage"], c["target"], c["level"], c["seed"], c["model"], c.get("temperature", settings.llm_temperature),
            c.get("variant", ""), c.get("harness", settings.harness)) + ".json")).exists()]
        if not pending:
            continue
        log.info("serving %s for %d pending condition(s)", alias, len(pending))
        proc = llm_server.start(settings, gguf=settings.models_dir / gguf if gguf else None, alias=alias, n_cpu_moe=n_cpu_moe)
        try:
            queue(settings, manifest, only_model=alias)
        finally:
            llm_server.stop(proc)


def regrade(settings: Settings, keys: tuple[str, ...] = ("stitching_kl",)) -> int:
    """Recompute secondary metrics `keys` for every graded run from its saved weights (deviation 4: stitching fix).

    Grading is deterministic, so the primary metrics must reproduce; a mismatch raises instead of overwriting."""
    from vmi import targets
    from vmi.grader.score import grade_npz
    models, n = {}, 0
    for p in sorted((RESULTS / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
        if r.get("failed") or not npz.exists():
            continue
        m = models.setdefault(r["target"], targets.load_model(settings, r["target"]))
        g = grade_npz(m, npz)
        if abs(g["hs"] - r["hs"]) > 1e-9 or abs(g["kl_out"] - r["kl_out"]) > 1e-9 * max(1.0, r["kl_out"]):
            raise RuntimeError(f"{r['run_id']}: primary metrics did not reproduce ({g['hs']} vs {r['hs']})")
        r.update({k: g[k] for k in keys})
        _dump(p, r)
        n += 1
        log.info("regraded %s: %s", r["run_id"], {k: g[k] for k in keys})
    return n


def served_model(settings: Settings) -> str | None:
    """Alias of the model llama-server is serving now (None if no server answers)."""
    import httpx
    try:
        return httpx.get(f"http://127.0.0.1:{settings.llm_port}/props", timeout=5).json().get("model_alias")
    except (httpx.HTTPError, ValueError):
        return None


def queue(settings: Settings, manifest: Path, only_model: str | None = None) -> None:
    """Run every condition in a JSONL manifest (one JSON object of agent_run kwargs per line) without a result yet.
    Local-model conditions run only while llama-server serves that model; the others are skipped and picked up by
    a later call with the right server (the queue is resumable). `only_model` restricts the call to one simulator, so
    a local-model queue and a Claude queue can run side by side without both picking up the same condition."""
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cond = json.loads(line)
        if only_model and cond.get("model") != only_model:
            continue
        if cond.get("harness", settings.harness) != "claudecode":
            want, have = cond.get("model", settings.llm_model), served_model(settings)
            if want != have:
                log.info("skip %s/%s/%s: needs llama-server serving %s (serving %s)", cond["stage"], cond["level"],
                         cond.get("variant", ""), want, have)
                continue
        agent_run(settings, **cond)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vmi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("explain").add_argument("target")
    sub.add_parser("compile").add_argument("target")
    sub.add_parser("check").add_argument("--claude", action="store_true", help="Claude-mode plumbing check (one tiny request, well under $0.01)")
    r = sub.add_parser("run")
    r.add_argument("target")
    r.add_argument("level")
    r.add_argument("seed", type=int)
    r.add_argument("--stage", default="dev")
    r.add_argument("--harness", choices=["opencode", "mini", "claudecode"], default=None, help="override VMI_HARNESS")
    r.add_argument("--model", default=None, help="model alias (llama-server) or Claude model id (claudecode)")
    q = sub.add_parser("queue")
    q.add_argument("manifest", type=Path)
    q.add_argument("--only-model", default=None, help="run only this simulator's conditions")
    sub.add_parser("local", help="serve each local model in turn and run its pending conditions").add_argument("manifest", type=Path)
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
    tr = sub.add_parser("train", help="train fresh runs with the paper recipe (e.g. new_s11 = T-new, after the freeze)")
    tr.add_argument("run_ids", nargs="+")
    an = sub.add_parser("analyse")
    an.add_argument("round", type=int)
    sub.add_parser("regrade", help="recompute secondary metrics of graded runs from saved weights")
    sub.add_parser("diagnose", help="re-run agents' build scripts with single edits to find what made them fail")
    sub.add_parser("paper", help="build reports/paper.pdf from reports/paper/paper.md")
    sub.add_parser("sensitivity", help="grader and explainer sensitivity analyses (no agent)").add_argument("round", type=int)
    pub = sub.add_parser("publish", help="export artefacts (redacted, scanned) and upload them to the Hugging Face Hub")
    pub.add_argument("--dry-run", action="store_true", help="export and scan only")
    a = ap.parse_args(argv)
    setup_logging(a.cmd)
    settings = load()
    if a.cmd == "explain":
        explain(settings, a.target)
    elif a.cmd == "compile":
        compile_and_grade(settings, a.target)
    elif a.cmd == "check" and a.claude:
        from vmi.simulator.runner import claude_check
        info = claude_check(settings, settings.runs_dir / "_checks")
        _dump(RESULTS / "claude" / f"check_{time.strftime('%Y%m%dT%H%M%S')}.json", info)
        log.info("claude check: %s", json.dumps({k: v for k, v in info.items() if k != "proxy_log"}))
    elif a.cmd == "check":
        from vmi.simulator.runner import isolation_check
        lines = isolation_check(settings, settings.runs_dir / "_checks")
        _dump(RESULTS / "isolation" / f"check_{time.strftime('%Y%m%dT%H%M%S')}.json", lines)
        bad = [x for x in lines if x.startswith("FAIL")]
        log.info("isolation check: %d ok, %d FAIL %s", len(lines) - len(bad), len(bad), bad)
    elif a.cmd == "run":
        if a.harness:
            settings = dataclasses.replace(settings, harness=a.harness)
        agent_run(settings, stage=a.stage, target=a.target, level=a.level, seed=a.seed, model=a.model)
    elif a.cmd == "local":
        local_queue(settings, a.manifest)
    elif a.cmd == "queue":
        queue(settings, a.manifest, a.only_model)
    elif a.cmd == "serve":
        from vmi.simulator import llm_server
        proc = llm_server.start(settings, n_cpu_moe=a.n_cpu_moe, gguf=a.gguf, alias=a.alias)
        log.info("llama-server healthy (pid %d); Ctrl+C to stop", proc.pid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            llm_server.stop(proc)
    elif a.cmd == "train":
        from vmi.training import train_fresh
        train_fresh(settings, a.run_ids)
        log.info("trained %s into %s", a.run_ids, settings.data_dir / "trained")
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
    elif a.cmd == "publish":
        from vmi import publish
        folder = publish.export(settings)
        log.info("exported %s (scan clean)", folder)
        if not a.dry_run:
            log.info("collection %s", publish.upload(folder, f"Update after {time.strftime('%Y-%m-%d %H:%M')}"))
    elif a.cmd == "sensitivity":
        from vmi import sensitivity
        g = sensitivity.grader_sensitivity(settings, a.round)
        log.info("grader sensitivity: %s", {k: v["kendall_tau_vs_preregistered"] for k, v in g["summary"]["agent"].items()})
        e = sensitivity.explainer_sensitivity(settings)
        log.info("explainer sensitivity: %d compiler builds graded", len(e["rows"]))
    elif a.cmd == "paper":
        from vmi.analysis.report import build_paper
        log.info("wrote %s", build_paper())
    elif a.cmd == "regrade":
        log.info("regraded %d runs", regrade(settings))
    elif a.cmd == "diagnose":  # interventions on agents' build scripts → results/diagnoses/ (read by the README examples)
        from vmi.analysis.diagnose import diagnose_all
        for d in diagnose_all():
            log.info("diagnosed %s: %s", d["run_id"], [(v["name"], round(v["accuracy"], 4)) for v in d["variants"]])
    elif a.cmd == "analyse":  # hypotheses + figures + every table in README/reports/paper, from results/
        from vmi.analysis.report import regenerate
        log.info("report: %s", json.dumps(regenerate(a.round)))


if __name__ == "__main__":
    main()
