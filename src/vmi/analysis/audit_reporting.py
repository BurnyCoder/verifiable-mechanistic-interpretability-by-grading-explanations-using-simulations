"""Generate dated hypothesis evidence from the independent saved-grade reconstruction."""

from statistics import median

from .scientific_audit import ROOT


def interval(xs, digits=3):
    return "unavailable" if xs is None else "[" + ", ".join(f"{x:.{digits}f}" for x in xs) + "]"


def render(d):
    """Keep effect estimates, denominators, missing arms and historical measurements visible in the report."""
    i, fold, cov = d["inventory"], d["fold_overlap"], d["coverage"]
    lines = [
        f"The manifests contain {i['manifest_rows']} slots. We matched {i['reported']} unique non-development grades, "
        f"excluded {i['dev']} development records and found {len(i['unfilled_manifest_rows'])} unfilled local slots. "
        f"The {len(i['infra_retry_ids'])} archived infrastructure failure shares its ID with a successful retry. "
        f"The included cohort has {i['with_failure_grade']} failure grades. Recombining the saved directional scores "
        f"at each site into HS gave a maximum absolute difference of {i['maximum_hs_reconstruction_error']:.3g}.",
        "",
        "| Round | Included run IDs | H1 HS on T0 | H1 decreasing KL on T0 | E0 minus E1 (HS, KL) |",
        "|---|---:|---|---|---|",
    ]
    for n, r in d["rounds"].items():
        h = r["targets"]["T0"]["H1"]

        def summarize(x):
            return (
                f"{'supported' if x['supported'] else 'not supported'}, J={x['J']}, "
                f"p={x['upper_tail']}/{x['assignments']} ({x['p']:.6g}), ρ={x['rho']:.3f}, CI {interval(x['ci'])}"
            )

        lines.append(
            f"| {n} | {r['N']} | {summarize(h['hs'])} | {summarize(h['decreasing_kl'])} | "
            f"{h['E0_minus_E1']['hs']:+.4f}, {h['E0_minus_E1']['kl_out']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive values of ρ indicate the predicted trend for both endpoints. The KL analysis ranks negative KL. "
            "E0 is a directional check outside the E1–E4 trend. A single observation per level yields no within-level uncertainty estimate.",
            "",
            "### Replications on each available ladder",
            "",
            "| Round | Target | Measured E levels | H1 HS p / ρ | H1 decreasing KL p / ρ | H3 W-freq agreement, KL, true KL range | H3 interval / upper reading |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for n, r in d["rounds"].items():
        for target, t in r["targets"].items():
            h, k = t["H1"], t["H3"]
            pairs = [f"{h[m]['p']:.5g} / {h[m]['rho']:+.3f}" for m in ("hs", "decreasing_kl")]
            verdict = " / ".join("passes" if k[m] else "fails" for m in ("interval_reading", "upper_bound_reading"))
            if not k["full_design"]:
                verdict += " on observed levels only, E2 missing"
            lines.append(
                f"| {n} | {target} | {', '.join(h['levels'])} | {' | '.join(pairs)} | "
                f"{k['agreement']:.5f}, {k['wrong_frequency_kl']:.6f}, {interval(k['true_kl_range'], 6)} | {verdict} |"
            )
    lines.extend(
        [
            "",
            "The primary H3 prediction requires agreement ≥0.99, KL inside the range of the three true condition "
            "means, and HS below each. Its falsifier names only the upper KL boundary and the HS comparison. We "
            "retain the agreement check for both tabulated readings. Missing levels restrict both to observed arms. "
            "The JSON also gives each W-freq minus true HS difference and its interval.",
            "",
            "### H2 discrimination",
            "",
            "| Round | Target | True minus control | Δ mean HS | Exact bootstrap percentile interval | Rank-biserial r | Raw p | Holm p | Registered verdict |",
            "|---|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for n, r in d["rounds"].items():
        for target, t in r["targets"].items():
            for name, c in t["H2"].items():
                lines.append(
                    f"| {n} | {target} | {name.replace('>', ' − ')} | {c['mean_difference']:+.4f} | {interval(c['ci'])} | "
                    f"{c['rank_biserial']:+.3f} | {c['p']:.4g} | {c['holm_p']:.4g} | {c['verdict']} |"
                )
    lines.extend(
        [
            "",
            "At one replicate §6 prescribes “cannot tell” for every H2, H4 and H7 contrast. Enumerated probabilities "
            "in this report show what the ranks do. T0 has four comparisons. Each seed target ran only the "
            "wrong-frequency control, so the remaining H2 comparisons are missing. At three versus three the minimum "
            "one-sided untied p is 1/20. Section 5 requires p strictly below 0.05. The later “cannot tell” label for a "
            "test with inadequate resolution is post hoc. Difference intervals above enumerate bootstrap support. The "
            "production report uses the registered 10,000 resamples and can differ by one or more discrete quantiles.",
            "",
            "### Baselines, calibration, specificity and trajectory",
            "",
            "| Round | Hypothesis | Result from saved measurements | Verdict and evidence limit |",
            "|---|---|---|---|",
        ]
    )
    for n, r in d["rounds"].items():
        for name, rows in r["H4"].items():
            arms = ", ".join(
                f"{lv} r={c['rank_biserial']:+.2f}, p={c['p']:.5g}, Holm p={c['holm_p']:.5g}" for lv, c in rows.items()
            )
            verdict = ", ".join(f"{lv} {c['verdict']}" for lv, c in rows.items())
            lines.append(
                f"| {n} | H4, {name} | {arms} | {verdict}. "
                + (
                    "Fixed same-recipe reference seeds."
                    if name == "retrains"
                    else "Two-sided secondary comparison with temperature-one probability students."
                )
                + " |"
            )
        pc = r["H4b"]["rows"]
        lines.append(
            f"| {n} | H4b | Step-zero HS {pc[0]['hs']:.12f}. KL sequence "
            + ", ".join(f"{v['kl_out']:.6g}" for v in pc)
            + " | Failed. HS falls across snapshots. KL peaks and then falls. One planted series cannot validate calibration under other perturbations. |"
        )
        loc = r["H5_H6"]
        gaps = ", ".join(f"{lv} {g['agent_minus_compiler']:+.4f}" for lv, g in loc["T0_compiler"].items())
        lines.append(
            f"| {n} | H5 | T0 agent minus compiler HS, {gaps} | Not evaluable for capability. "
            f"Only {loc['completed_cells']}/{loc['required_cells']} local cells completed. |"
        )
        lines.append(
            f"| {n} | H6 | {loc['completed_cells']}/{loc['required_cells']} local cells, all observed local HS zero | "
            "Not evaluable. The stopped cells leave missing comparisons and tied observed rankings. |"
        )
        h7, h8 = r["H7"], r["H8"]
        margins = list(h7["margins"].values())
        lines.append(
            f"| {n} | H7 | {len(margins)} margins over {len(h7['wrong_targets'])} fixed comparators, "
            f"{min(margins):+.4f} to {max(margins):+.4f}, signed-rank p={h7['p']:.6g} | {h7['verdict']}. "
            "Specificity conditional on this low-overlap target panel. |"
        )
        rho = list(h8["replacements"].values())
        lines.append(
            f"| {n} | H8 | {len(rho)} builds × {h8['checkpoints']} checkpoints, median ρ={median(rho):.4f}, "
            f"minimum {min(rho):.4f} | {h8['verdict']}. Successive checkpoints share a training path. |"
        )
    hist = d["rounds"]["1"]["H8"]["historical"]
    later = d["rounds"]["2"]["H8"]["historical"]
    prompts = d["stimuli"]
    example = prompts["T0_E3_amplitude_example"]
    role = d["W_role_compiler"]
    lines.extend(
        [
            "",
            f"The original round-one analysis used the selected E3 seed-zero replacement over {hist['checkpoints']} "
            f"sampled checkpoints (ρ={hist['rho']:.6f}). The first-round H8 row above is a retrospective reconstruction "
            "from the later complete trajectories restricted to seed-zero builds. It was unavailable at that decision. "
            "[Original Git record](https://github.com/BurnyCoder/verifiable-mechanistic-interpretability-by-grading-explanations-using-simulations/blob/26ff26200c0c4b7d07742d7e3cdc9e253d1defeb/results/controls/trajectory_T0.json).",
            "",
            f"The initial round-two summary also sampled {later['checkpoints']} checkpoints for one selected E4 build "
            f"(ρ={later['rho']:.6f}). [Superseded round-two measurements](../../results/controls/trajectory_T0_superseded.json).",
            "",
            "### Metric discrepancies and uncertainty",
            "",
            f"- The character counts agree with the archived text plus any neuron table for all {prompts['stimuli']} stimuli. "
            f"T0 W-alg contains {prompts['character_ratios']['T0']['W-alg']:.1%} as many characters as E3. "
            f"The compiler can access more precision than the agent text. At T0 E3 frequency {example['frequency']}, "
            f"its logit amplitude is {example['compiler']}, displayed as {example['rendered']}. "
            "The effect of this precision difference on grades remains unknown. [Full specification](../../results/explanations/T0/E3/spec.json) · "
            "[Rendered text](../../results/explanations/T0/E3/EXPLANATION.md) · [Compiler](../../src/vmi/reference/compile.py).",
            f"- The wrong-role compiler achieved {role['argmax']:.4%} top-1 argmax accuracy. The fraction with its true "
            f"logit within 1e-12 of the maximum was {role['near_tie']:.4%}. The fields have distinct definitions. "
            "[Primary grade](../../results/compiler/T0/W-role.json) · [Diagnostic](../../results/controls/clock_pizza.json).",
            "- Scale-adjusted logit FVU ranges were "
            + ", ".join(f"{k} {min(v):.5f}–{max(v):.5f}" for k, v in d["scale_adjusted_fvu"].items())
            + ". All observed E3 errors were lower than W-alg. E4 and W-scramble overlapped. This post-hoc comparison used three builds per arm. "
            "[FVU records](../../results/controls/logit_fvu_T0.json).",
            f"- {fold['swap_in_training']:,} of {fold['ordered_off_diagonal']:,} off-diagonal ordered inputs ({fold['fraction']:.1%}) "
            "have their reverse in a different fold. Both operands also recur throughout the grid. Alignment generalization to entirely unseen unordered pairs remains unmeasured. "
            "[Fold code](../../src/vmi/grader/metrics.py).",
            f"- For {cov['repetitions']:,} independent experiments with three iid standard-normal observations per arm, the exact bootstrap percentile range "
            f"covered the population mean in {cov['mean']:.2%} and the difference of means in {cov['difference']:.2%}. "
            f"Monte Carlo standard errors were {cov['monte_carlo_se'][0]:.2%} and {cov['monte_carlo_se'][1]:.2%}. "
            "The single-arm range covers a continuous population median with probability 3/4. For independent draws from a continuous distribution F with finite mean μ, its coverage "
            "is 1 − F(μ)³ − [1 − F(μ)]³. Skewness can put the mean far from the median. [Statistical code](../../src/vmi/analysis/audit_statistics.py) · "
            "[Hesterberg](https://arxiv.org/pdf/1411.5279).",
        ]
    )
    return "\n".join(lines)


def update(data):
    from .tables import inject

    inject(ROOT / "reports/reviews/scientific-audit-2026-11-26.md", "reconstructed-evidence", render(data))
