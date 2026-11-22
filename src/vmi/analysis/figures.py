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
    # Audit 4: the snapshot's own task accuracy is drawn too — its dip and recovery is what makes KL rise and fall, so
    # without it the KL panel reads as a grader failure. Snapshots are categorical (0, 100, 1k, 10k steps): markers only,
    # no connecting lines, so the uneven spacing is not drawn as a trajectory.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for ax, key, title in ((axes[0], "hs", "HS ↑"), (axes[1], "kl_out", "Output KL (nats) ↓"),
                           (axes[2], "snapshot_accuracy", "Snapshot task accuracy")):
        ax.plot(x, [r[key] for r in rows], color=TRUE, marker="o", markersize=6, linestyle="none")
        ax.set_xticks(list(x), [f"{r['step']:,}" for r in rows])
        ax.set_xlabel("further training steps (snapshot)")
        ax.set_title(title, loc="left")
        if key in ("hs", "snapshot_accuracy"):
            ax.set_ylim(-0.02, 1.02)  # full [0, 1] range, as in every other figure (no truncated axis)
        if key == "snapshot_accuracy":  # the dip is small on a full axis, so print the values instead of zooming
            for xi, r in zip(x, rows):
                ax.annotate(f"{r[key]:.3f}", (xi, r[key]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7)
    fig.suptitle(f"Planted-mechanism calibration ({target} compiled E3 model as ground truth)", x=0.01, y=1.04, ha="left")
    return _save(fig, f"planted_{target}")


def fig_trajectory(target: str = "T0") -> Path | None:
    """H8: every E3/E4 replacement graded against each training checkpoint (one thin line each), with the post-hoc
    negative controls (W-freq, E1, E0 replacements) behind them, above the restricted loss."""
    d = _json(RES / "controls" / f"trajectory_{target}.json")
    if not d or "replacements" not in d:
        return None
    steps = d["steps"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    for c in d.get("negative_controls", []):  # post hoc: replacements without T0's mechanism, drawn behind
        axes[0].plot(steps, c["hs"], color=CTRL, lw=0.8, alpha=0.6)
    for r in d["replacements"]:
        axes[0].plot(steps, r["hs"], color=TRUE, lw=0.8, alpha=0.7)
    # Audit 4 positive control (post hoc): final T0 itself against its own checkpoints, black dashed. Any network that
    # resembles the end point rises the same way, which is how H8's correlation should be read.
    pc = _json(RES / "audit4" / "h8_positive_control.json")
    if pc and pc.get("steps") == steps:
        axes[0].plot(steps, pc["hs"]["final T0 (weights dump)"], color="black", lw=1.2, ls="--")
    axes[0].set_ylim(0, 1)
    nc = len(d.get("negative_controls", []))
    axes[0].set_title(f"HS vs checkpoint ↑: {len(d['replacements'])} E3/E4 replacements (blue)"
                      + (f", {nc} W-freq/E1/E0 controls (orange, post hoc)" if nc else "")
                      + (", final T0 itself (black dashed, post hoc)" if pc else ""), loc="left", fontsize=8)
    axes[1].plot(steps, d["restricted_loss"], color=TRUE)
    axes[1].set_yscale("log")
    axes[1].set_title("Restricted loss of the checkpoint (key frequencies only; log) ↓", loc="left")
    axes[1].set_xlabel(f"training step of the target checkpoint ({len(steps)} checkpoints)")
    ctrl_rho = [c["rho"] for c in d.get("negative_controls", [])]
    fig.suptitle(f"Training trajectory ({target}); Spearman(HS, −log restricted loss): E3/E4 min "
                 f"{d['spearman_min']:.2f}, median {d['spearman_hs_vs_neglog_restricted_loss']:.2f}"
                 + (f"; controls max {max(ctrl_rho):+.2f}" if ctrl_rho else "")
                 + (f"; final T0 itself {pc['rho_vs_neglog_restricted_loss']['final T0 (weights dump)']:.2f}" if pc else ""),
                 x=0.01, ha="left", fontsize=9)
    return _save(fig, f"trajectory_{target}")


def fig_length(max_round: int, target: str = "T0") -> Path | None:
    """HS against explanation length (characters, log): compiler per level and main-simulator runs."""
    runs = [r for r in load_runs(max_round) if not r.get("failed")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    any_point = False
    placed = []  # label positions already used, so neighbouring labels do not overlap
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
        y = c["hs"] if c else rs[0]["hs"] if rs else 0
        near = any(abs(np.log(L["chars"] / x)) < 0.3 and abs(y - yy) < 0.1 for x, yy in placed)  # a label close by
        left = lv in ("E1", "E2", "E3", "E4")  # true explanations labelled on the left, controls on the right
        ax.annotate(lv, (L["chars"], y), xytext=(-7, 5) if left else (7, -13) if near else (7, 5), textcoords="offset points",
                    fontsize=7, color=INK2, ha="right" if left else "left")
        placed.append((L["chars"], y))
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
    n_of = {m: max(len(_select(runs, target=target, level=lv, model=m)) for lv in levels) for m in models}  # runs per condition
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
    h = [plt.Line2D([], [], marker=markers[j], color=SERIES[j], linestyle="", label=f"{m} (N = {n_of[m]} per condition)") for j, m in enumerate(models)]
    h.append(plt.Line2D([], [], marker="D", markerfacecolor="none", color=INK2, linestyle="", label="reference compiler"))
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    ax.set_title(f"Target {target}: simulator capability ablation (rounds 1–{max_round}; local runs stopped after round 1)", loc="left")
    return _save(fig, f"simulators_{target}")


KIND_COLOUR = {"target": INK2, "true": TRUE, "control": CTRL, "reference": REF}


def _title(p: dict) -> str:
    return p["name"] + (" (ground truth)" if p["kind"] == "target" else f" — HS {p['hs']:.2f}")


def fig_examples(max_round: int, target: str = "T0", panels: list[dict] | None = None) -> Path | None:
    """Output examples: which frequencies each network actually built, against the target's (ground truth).

    One panel per network (`examples.panel_models`), bars = per-frequency norm of W_L = W_U·W_out (grokviz
    `wl_fourier_norms`, cos and sin combined; paper Fig. 3), each panel scaled to its own maximum; shaded columns
    mark the target's key frequencies. HS against the target is printed in each panel title."""
    import torch
    from grokviz import fourier as gf
    from grokviz import metrics as gm

    from vmi.analysis.examples import panel_models
    panels = panels or panel_models(max_round, target)
    F, _ = gf.fourier_basis(113)
    key = set(json.loads((RES / "explanations" / target / "analysis.json").read_text())["key_freqs"])
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.0 * len(panels)), sharex=True)
    for ax, p in zip(axes, panels):
        with torch.no_grad():
            norms = gf.frequency_norms(gm.wl_fourier_norms(p["model"], F)).numpy()
        for f in key:
            ax.axvspan(f - 0.5, f + 0.5, color=GRID, zorder=0)
        ax.bar(np.arange(1, len(norms) + 1), norms / max(norms.max(), 1e-12), width=0.8, color=KIND_COLOUR[p["kind"]], zorder=2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.grid(axis="x", visible=False)
        ax.set_title(_title(p), loc="left", fontsize=8, pad=2)
    axes[-1].set_xlabel("frequency k (shaded: the target's key frequencies)")
    fig.supylabel("W_L Fourier norm / max", fontsize=8)
    fig.suptitle(f"Which frequencies each network built (target {target}, round {max_round}, first run per condition)",
                 x=0.01, ha="left", fontsize=10, y=1.0)
    fig.tight_layout()
    return _save(fig, f"examples_{target}")


def fig_outputs(max_round: int, target: str = "T0", panels: list[dict] | None = None) -> Path | None:
    """Outputs vs. internals over all 113×113 inputs: P(correct answer) and the neuron best matching a target neuron.

    Left column: probability each network gives to the true answer (a+b) mod 113 for every input (row a, column b);
    the ground truth is 1 everywhere. Right column: activation of the network's MLP neuron with the largest |Pearson r|
    to the target's highest-variance neuron (each panel scaled to its own maximum; r printed). Working networks look
    the same on the left and differ on the right; that difference is what HS measures and output KL does not."""
    import torch

    from vmi.analysis.examples import panel_models
    from vmi.grader.activations import collect
    panels = panels or panel_models(max_round, target)
    truth = torch.tensor([(a + b) % 113 for a in range(113) for b in range(113)])
    acts = {p["name"]: collect(p["model"], "cpu") for p in panels}
    ref = acts[panels[0]["name"]].mlp_post
    j0 = int(ref.var(0).argmax())  # the target neuron with the largest activation variance over inputs
    x0 = (ref[:, j0] - ref[:, j0].mean()) / (ref[:, j0].std() + 1e-12)
    fig, axes = plt.subplots(2, len(panels), figsize=(1.45 * len(panels), 3.9))  # one column per network
    for col, p in zip(axes.T, panels):
        a = acts[p["name"]]
        prob = torch.softmax(a.logits, -1).gather(1, truth[:, None]).squeeze(1).reshape(113, 113).numpy()
        X = (a.mlp_post - a.mlp_post.mean(0)) / (a.mlp_post.std(0) + 1e-12)
        r = (X * x0[:, None]).mean(0)  # Pearson r of every neuron with the target neuron j0
        j = int(r.abs().argmax())
        act = a.mlp_post[:, j].reshape(113, 113).numpy()
        im = col[0].imshow(prob, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        col[1].imshow(act / max(act.max(), 1e-12), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        head = p["name"].replace("agent, ", "agent ").replace(" (", "\n(")
        col[0].set_title(head + ("\nground truth" if p["kind"] == "target" else f"\nHS {p['hs']:.2f}"),
                         fontsize=7, pad=3, color=KIND_COLOUR[p["kind"]])
        col[1].set_xlabel(f"neuron {j0}\n(reference)" if p["kind"] == "target" else f"neuron {j}\nr = {float(r[j]):+.2f}",
                          fontsize=7)
        for ax in col:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axes[0][0].set_ylabel("P(correct answer)\nrows a, columns b", fontsize=7)
    axes[1][0].set_ylabel("best-matching neuron\n(scaled to its max)", fontsize=7)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"Outputs vs. internals on all 113×113 inputs (target {target}, round {max_round}): top, the probability "
                 "of the true answer (ground truth: 1 everywhere); bottom, the neuron best matching the target's neuron",
                 x=0.01, ha="left", fontsize=8)
    return _save(fig, f"outputs_{target}")


def fig_architecture() -> Path:
    """Pipeline diagram (no data): explainer → weights-blind offline agent → grader, with the agent-free compiler path."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def box(x, y, w, h, text, edge=INK2, dashed=False, size=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", linewidth=1.3,
                                    edgecolor=edge, facecolor=SURFACE, linestyle=(0, (4, 3)) if dashed else "solid"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3)

    def arrow(a, b, text="", color=INK2, dx=0.0, dy=1.2):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color))
        if text:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha="center", fontsize=7, color=INK2)

    # top row: target → explainer → explanation → [sandbox: agent → weights] → grader
    box(0.5, 31, 12, 10, "Target model M\n(trained weights)")
    box(16, 31, 14, 10, "Explainer script\n(weights access:\nFourier analysis)")
    box(33.5, 31, 13, 10, "EXPLANATION.md\nE0–E4 (true)\nW-* (wrong)")
    ax.add_patch(FancyBboxPatch((50, 21.5), 29, 24, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.3,
                                edgecolor=CTRL, facecolor="none", linestyle=(0, (4, 3))))
    ax.text(64.5, 43.6, "offline sandbox: no network, no path to M", ha="center", fontsize=7, color=CTRL)
    box(51.5, 31, 11, 10, "LLM agent\n(Claude Code or\nbash harness)", edge=TRUE)
    box(65.5, 31, 12, 10, "build_weights.py\n↓\nweights.npz = M̂", edge=TRUE)
    box(65.5, 23, 12, 5.5, "TASK.md, forward.py,\nselftest.py", size=6.5)
    box(83, 31, 16, 10, "Grader\nKL_out on 12,769 inputs\nHS at 3 sites (CV ridge)")
    # below the sandbox: the model endpoint; bottom row: the agent-free reference compiler
    box(50, 12, 29, 6, "LLM endpoint outside the sandbox: API proxy\n(key, budget ledger) or llama-server", size=6.5)
    box(33.5, 1, 13, 8, "spec.json\n(explanation's\nparameters)", dashed=True)
    box(50, 2, 29, 6, "reference compiler (no agent): spec → weights", edge=REF)
    arrow((13, 36), (15.5, 36))
    arrow((30.5, 36), (33, 36))
    arrow((47, 36), (51, 36))
    arrow((63, 36), (65, 36))
    arrow((78, 36), (82.5, 36), "M̂")
    arrow((56.5, 18.5), (56.5, 30.5), "unix socket", dx=3.8, dy=-3)
    arrow((23, 30.5), (40, 9.5))
    arrow((47, 5), (49.5, 5))
    arrow((79.5, 5), (91, 30.5), color=REF)
    ax.plot([6.5, 6.5, 91, 91], [41.5, 50, 50, 44], color=INK2, linewidth=1.2)  # M goes to the grader only, around the sandbox
    ax.add_patch(FancyArrowPatch((91, 44), (91, 41.5), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=INK2))
    ax.text(48.75, 51, "M (to the grader only)", ha="center", fontsize=7, color=INK2)
    return _save(fig, "architecture")


def make_all(max_round: int) -> list[Path]:
    """Every figure whose inputs exist (missing inputs are skipped, never faked)."""
    figs = [fig_architecture(), fig_ladder(max_round), fig_kl_vs_hs(max_round), fig_planted(), fig_trajectory(),
            fig_length(max_round), fig_simulators(max_round)]
    from vmi.analysis.examples import panel_models
    panels = panel_models(max_round)  # load the example networks once for both example figures
    figs += [fig_examples(max_round, panels=panels), fig_outputs(max_round, panels=panels)]
    return [p for p in figs if p]
