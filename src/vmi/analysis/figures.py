"""Static figures generated from results/ (no hand-typed numbers): written to reports/figures/*.png.

Local explanation: each `fig_*` function reads results files, draws one matplotlib figure and returns its path;
`make_all(round)` draws every figure whose inputs exist. Colours follow one rule everywhere: blue = true explanation,
orange = control (no or wrong explanation), aqua = no-agent reference points; marker shape separates agent runs
(filled circle) from the agent-free reference compiler (open diamond), so identity never rests on colour alone.
Two measures with different scales get two panels, never two y-axes.
Global context: `vmi report` calls `make_all`; README, paper, summary and site embed these PNGs.
Palette: the validated default categorical palette (slots 1–3 pass the all-pairs colour-vision-deficiency check) of
the dataviz guidance used for this project; matplotlib API: https://matplotlib.org/stable/api/
"""

from __future__ import annotations  # postponed annotations

import json  # results files
from pathlib import Path  # output paths

import matplotlib  # plotting backend selection

matplotlib.use("Agg")  # file output only (no display; works in background jobs)
import matplotlib.pyplot as plt  # noqa: E402  figures
import numpy as np  # noqa: E402  arrays

from vmi.analysis.hypotheses import LOCAL_MODELS, MAIN_MODEL, _select, load_runs  # noqa: E402  same selection as the tests
from vmi.config import REPO_ROOT  # noqa: E402  paths

RES = REPO_ROOT / "results"
OUT = REPO_ROOT / "reports" / "figures"
TRUE, CTRL, REF = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1–3: true explanation, control, no-agent reference
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1–4 for simulators (adjacent-pair safe; shapes added)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"  # text, secondary text, grid, chart surface
LADDER = ["E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble"]
IS_TRUE = {"E1", "E2", "E3", "E4", "E-paper"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def _references(target: str) -> dict[str, list[dict]]:
    """No-agent reference points for horizontal lines: random init, retrains, distilled students."""
    b = _json(RES / "baselines" / f"{target}.json") or {}
    return {k: v for k, v in (("random init", _json(RES / "controls" / f"random_init_{target}.json") or []),
                              ("retrains", b.get("retrains", [])), ("distilled students", b.get("students", []))) if v}


def fig_ladder(max_round: int, target: str = "T0") -> Path | None:
    """HS and output KL per explanation: every main-simulator run (dots) and the reference compiler (diamonds)."""
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    levels = [lv for lv in LADDER if _select(runs, target=target, level=lv) or (RES / "compiler" / target / f"{lv}.json").exists()]
    if not levels:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axes[0], "hs", "Hidden-state similarity HS ↑"), (axes[1], "kl_out", "Output KL (nats, log) ↓")):
        for i, lv in enumerate(levels):
            col = TRUE if lv in IS_TRUE else CTRL
            vals = [r[key] for r in _select(runs, target=target, level=lv)]
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [i] * len(vals),
                       vals, s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            c = _json(RES / "compiler" / target / f"{lv}.json")
            if c:
                ax.scatter([i + 0.25], [max(c[key], 1e-9)], s=40, marker="D", facecolor="none", edgecolor=col, linewidth=1.5, zorder=3)
        if key == "kl_out":
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.02, 1.02)
        refs = sorted(((float(np.mean([r[key] for r in rs])), f"{name} (n={len(rs)})") for name, rs in _references(target).items()))
        groups: list[list[tuple[float, str]]] = []
        for m, label in refs:  # lines closer than 12 px share one label, so labels never overprint or float between lines
            ax.axhline(m, color=REF, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            px = ax.transData.transform((0, max(m, 1e-12)))[1]
            if groups and px - ax.transData.transform((0, max(groups[-1][-1][0], 1e-12)))[1] < 12:
                groups[-1].append((m, label))
            else:
                groups.append([(m, label)])
        for g in groups:
            ax.annotate(" ≈ ".join(lbl for _, lbl in g), (len(levels) - 0.5, max(m for m, _ in g)), xytext=(2, 2),
                        textcoords="offset points", color=INK2, fontsize=7, ha="right", va="bottom")
        ax.set_xticks(range(len(levels)), levels, rotation=30)
        ax.set_title(title, loc="left")
    n = min((len(_select(runs, target=target, level=lv)) for lv in levels), default=0)
    h = [plt.Line2D([], [], marker="o", color=TRUE, linestyle="", label=f"{MAIN_MODEL} run, true explanation"),
         plt.Line2D([], [], marker="o", color=CTRL, linestyle="", label=f"{MAIN_MODEL} run, control explanation"),
         plt.Line2D([], [], marker="D", markerfacecolor="none", color=INK2, linestyle="", label="reference compiler (no agent)"),
         plt.Line2D([], [], color=REF, linestyle=(0, (4, 3)), linewidth=1, label="no-agent reference (mean)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(f"Target {target}: scores per explanation (round {max_round}, N = {n} run(s) per condition)", x=0.01, ha="left")
    return _save(fig, f"ladder_{target}")


def fig_kl_vs_hs(max_round: int, target: str = "T0") -> Path | None:
    """Behaviour vs mechanism: output KL (x, log) against HS (y) for runs, compiler and no-agent references."""
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lv in LADDER:
        col = TRUE if lv in IS_TRUE else CTRL
        rs = _select(runs, target=target, level=lv)
        for r in rs:
            ax.scatter(max(r["kl_out"], 1e-9), r["hs"], s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if rs:  # one direct label per condition, at its median point (no label on every point)
            ax.annotate(lv, (np.median([max(r["kl_out"], 1e-9) for r in rs]), np.median([r["hs"] for r in rs])),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ref_marker = {"random init": "v", "retrains": "s", "distilled students": "^"}  # shape tells reference types apart
    for name, rs in _references(target).items():
        ax.scatter([max(r["kl_out"], 1e-9) for r in rs], [r["hs"] for r in rs], s=36, marker=ref_marker[name], color=REF,
                   edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Output KL(M ‖ M̂), nats (log) ← better behaviour")
    ax.set_ylabel("Hidden-state similarity HS → better mechanism")
    h = [plt.Line2D([], [], marker="o", color=TRUE, linestyle="", label=f"{MAIN_MODEL} run, true explanation"),
         plt.Line2D([], [], marker="o", color=CTRL, linestyle="", label=f"{MAIN_MODEL} run, control explanation"),
         *[plt.Line2D([], [], marker=ref_marker[n], color=REF, linestyle="", label=f"{n} (no agent, n={len(rs)})")
           for n, rs in _references(target).items()]]
    ax.legend(handles=h, loc="center right", fontsize=7)
    ax.set_title(f"Target {target}: behaviour (KL) vs mechanism (HS), round {max_round}", loc="left")
    return _save(fig, f"kl_vs_hs_{target}")


def fig_planted(target: str = "T0") -> Path | None:
    """H4b calibration: score of the planted (compiled E3) model against itself trained further."""
    d = _json(RES / "controls" / f"planted_calibration_{target}.json")
    if not d:
        return None
    rows = d["rows"]
    x = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for ax, key, title in ((axes[0], "hs", "HS ↑"), (axes[1], "kl_out", "Output KL (nats) ↓")):
        ax.plot(x, [r[key] for r in rows], color=TRUE, marker="o", markersize=6)
        ax.set_xticks(list(x), [f"{r['step']:,}" for r in rows])
        ax.set_xlabel("further training steps (snapshots)")
        ax.set_title(title, loc="left")
    fig.suptitle(f"Planted-mechanism calibration ({target} compiled E3 model as ground truth)", x=0.01, y=1.04, ha="left")
    return _save(fig, f"planted_{target}")


def fig_trajectory(target: str = "T0") -> Path | None:
    """H8: the best E3/E4 replacement graded against every 10th training checkpoint, above the restricted loss."""
    d = _json(RES / "controls" / f"trajectory_{target}.json")
    if not d:
        return None
    rows = d["rows"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    axes[0].plot(steps, [r["hs"] for r in rows], color=TRUE)
    axes[0].set_title(f"HS of replacement {d['replacement'].split('__')[2]} vs checkpoint ↑", loc="left")
    axes[1].plot(steps, [r["restricted_loss"] for r in rows], color=TRUE)
    axes[1].set_yscale("log")
    axes[1].set_title("Restricted loss of the checkpoint (key frequencies only; log) ↓", loc="left")
    axes[1].set_xlabel("training step of the target checkpoint")
    fig.suptitle(f"Training trajectory ({target}); Spearman(HS, −log restricted loss) = "
                 f"{d['spearman_hs_vs_neglog_restricted_loss']:.2f}", x=0.01, ha="left")
    return _save(fig, f"trajectory_{target}")


def fig_length(max_round: int, target: str = "T0") -> Path | None:
    """HS against explanation length (characters, log): compiler per level and main-simulator runs."""
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    any_point = False
    for lv in LADDER:
        L = _json(RES / "explanations" / target / lv / "length.json")
        if not L:
            continue
        col = TRUE if lv in IS_TRUE else CTRL
        c = _json(RES / "compiler" / target / f"{lv}.json")
        if c:
            ax.scatter(L["chars"], c["hs"], s=40, marker="D", facecolor="none", edgecolor=col, linewidth=1.5, zorder=3)
            any_point = True
        rs = _select(runs, target=target, level=lv)
        if rs:
            ax.scatter([L["chars"]] * len(rs), [r["hs"] for r in rs], s=40, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            any_point = True
        ax.annotate(lv, (L["chars"], c["hs"] if c else rs[0]["hs"] if rs else 0), xytext=(7, 5), textcoords="offset points",
                    fontsize=7, color=INK2)
    if not any_point:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("explanation length (characters, log)")
    ax.set_ylabel("HS ↑")
    h = [plt.Line2D([], [], marker="o", color=INK2, linestyle="", label=f"{MAIN_MODEL} run"),
         plt.Line2D([], [], marker="D", markerfacecolor="none", color=INK2, linestyle="", label="reference compiler"),
         plt.Line2D([], [], marker="s", color=TRUE, linestyle="", label="true explanation"),
         plt.Line2D([], [], marker="s", color=CTRL, linestyle="", label="control explanation")]
    ax.legend(handles=h, loc="lower right", fontsize=7)
    ax.set_title(f"Target {target}: score against explanation length (round {max_round})", loc="left")
    return _save(fig, f"length_{target}")


def fig_simulators(max_round: int, target: str = "T0") -> Path | None:
    """Capability ablation (H5/H6): HS per explanation for each simulator, with the reference compiler."""
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    levels = ["E2", "E3", "E4", "W-freq"]
    models = [m for m in LOCAL_MODELS + [MAIN_MODEL] if any(_select(runs, target=target, level=lv, model=m) for lv in levels)]
    if len(models) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    markers = ["o", "s", "^", "v"]
    for j, m in enumerate(models):
        off = (j - (len(models) - 1) / 2) * 0.12
        for i, lv in enumerate(levels):
            vals = [r["hs"] for r in _select(runs, target=target, level=lv, model=m)]
            ax.scatter([i + off] * len(vals), vals, s=40, marker=markers[j], color=SERIES[j], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for i, lv in enumerate(levels):
        c = _json(RES / "compiler" / target / f"{lv}.json")
        if c:
            ax.scatter([i + 0.35], [c["hs"]], s=40, marker="D", facecolor="none", edgecolor=INK2, linewidth=1.5)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("HS ↑")
    h = [plt.Line2D([], [], marker=markers[j], color=SERIES[j], linestyle="", label=m) for j, m in enumerate(models)]
    h.append(plt.Line2D([], [], marker="D", markerfacecolor="none", color=INK2, linestyle="", label="reference compiler"))
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    ax.set_title(f"Target {target}: simulator capability ablation (round {max_round})", loc="left")
    return _save(fig, f"simulators_{target}")


def make_all(max_round: int) -> list[Path]:
    """Every figure whose inputs exist (missing inputs are skipped, never faked)."""
    figs = [fig_ladder(max_round), fig_kl_vs_hs(max_round), fig_planted(), fig_trajectory(), fig_length(max_round),
            fig_simulators(max_round)]
    return [p for p in figs if p]
