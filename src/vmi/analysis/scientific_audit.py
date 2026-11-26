"""Reconstruct the dated scientific audit directly from records, without importing the grader, models or earlier audits.

The protocol is the tagged preregistration, 801edf013db7239f8994e6780c5873e1890f5103, §§4–8.
Measurement records are inputs. Agreement of downstream arithmetic cannot authenticate the original forward pass.
"""

import json
from statistics import mean
from pathlib import Path

import numpy as np

from . import audit_statistics as stats

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
AUDIT = RES / "scientific-audit"
LEVELS = ("E1", "E2", "E3", "E4")
TARGETS = ("T0", "T1", "T2", "T-new", "T-paper")
COMPARISONS = (("E3", "W-freq"), ("E3", "W-alg"), ("E3", "W-role"), ("E4", "W-scramble"))


def read(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def key(r):
    """Manifest identity independent of the filename encoder. Retry attempts share a condition and seed."""
    return (
        r.get("round_", r.get("round")),
        r["stage"],
        r["target"],
        r["level"],
        r.get("variant", ""),
        r["model"],
        r["harness"],
        r["seed"],
    )


def inventory():
    runs = [read(f.relative_to(RES)) for f in sorted((RES / "runs").glob("*.json"))]
    included = [r for r in runs if r["stage"] != "dev" and r["round"] in (1, 2)]
    planned = [
        json.loads(line)
        for f in sorted((RES / "manifests").glob("round[12]_*.jsonl"))
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    ids, design = [key(r) for r in included], [key(r) for r in planned]
    assert len(ids) == len(set(ids)) and len(design) == len(set(design)) and set(ids) <= set(design)
    assert all(r["run_id"] == f.stem for f, r in zip(sorted((RES / "runs").glob("*.json")), runs))
    missing = sorted(set(design) - set(ids))
    assert all(k[1] == "ablations" and k[5].startswith("qwen") for k in missing)
    residuals = []
    for r in included:
        sites = r["sites"]
        assert set(sites) == {"resid_mid", "mlp_post", "resid_post"}
        hs = []
        for site in sites.values():
            a, b = site["completeness"], site["soundness"]
            assert 0 <= a <= 1 and 0 <= b <= 1
            hs.append(2 * a * b / (a + b) if a + b else 0.0)
            assert abs(hs[-1] - site["harmonic"]) < 1e-12
        residuals.append(abs(mean(hs) - r["hs"]))
    retries = [f.stem for f in sorted((RES / "runs_infra_failed").glob("*.json"))]
    assert set(retries) <= {r["run_id"] for r in included}
    return included, {
        "reported": len(included),
        "dev": sum(r["stage"] == "dev" for r in runs),
        "manifest_rows": len(planned),
        "with_failure_grade": sum(r["failed"] for r in included),
        "infra_retry_ids": retries,
        "unfilled_manifest_rows": missing,
        "maximum_hs_reconstruction_error": max(residuals),
    }


def replication(runs, target):
    """Return every measured endpoint, explicit missing arms and both readings of the H3 wording."""
    sel = [
        r
        for r in runs
        if r["target"] == target
        and r["model"] == "claude-haiku-5-5"
        and not r["variant"]
        and r.get("temperature") is None
    ]
    groups = {lv: [r for r in sel if r["level"] == lv] for lv in (*LEVELS, "E0", *(w for _, w in COMPARISONS))}
    vals = lambda lv, metric: [row[metric] for row in groups[lv]]
    levels = [lv for lv in LEVELS if groups[lv]]
    true = [lv for lv in LEVELS[1:] if groups[lv]]
    n = min(len(groups[lv]) for lv in levels)
    h1 = {
        "levels": levels,
        "missing_levels": sorted(set(LEVELS) - set(levels)),
        "hs": stats.trend([vals(lv, "hs") for lv in levels]),
        "decreasing_kl": stats.trend([[-v for v in vals(lv, "kl_out")] for lv in levels]),
    }
    if groups["E0"] and groups["E1"]:
        h1["E0_minus_E1"] = {m: float(np.mean(vals("E0", m)) - np.mean(vals("E1", m))) for m in ("hs", "kl_out")}
    h2 = {}
    for a, b in COMPARISONS:
        if groups[a] and groups[b]:
            x, y = vals(a, "hs"), vals(b, "hs")
            h2[a + ">" + b] = {**stats.difference(x, y), **stats.ranksum(x, y)}
    adj = stats.holm({k: t["p"] for k, t in h2.items()})
    for k, t in h2.items():
        t.update(
            holm_p=adj[k],
            verdict="cannot tell at this N"
            if n == 1
            else "supported"
            if adj[k] < 0.05 and t["mean_difference"] > 0
            else "not supported",
        )
    kl = [float(np.mean(vals(lv, "kl_out"))) for lv in true]
    wf = float(np.mean(vals("W-freq", "kl_out")))
    agreement = float(np.mean(vals("W-freq", "argmax_agreement")))
    diffs = {lv: stats.difference(vals("W-freq", "hs"), vals(lv, "hs")) for lv in true}
    passes_other = agreement >= 0.99 and all(d["mean_difference"] < 0 for d in diffs.values())
    return {
        "N": n,
        "run_ids": sorted(r["run_id"] for r in sel),
        "H1": h1,
        "H2": h2,
        "H3": {
            "true_levels": true,
            "full_design": len(true) == 3,
            "agreement": agreement,
            "wrong_frequency_kl": wf,
            "true_kl_range": [min(kl), max(kl)],
            "hs_differences": diffs,
            "interval_reading": passes_other and min(kl) <= wf <= max(kl),
            "upper_bound_reading": passes_other and wf <= max(kl),
        },
    }


def other_hypotheses(runs):
    """H4–H8 use the measured baseline and control records, linked by run ID across files."""
    ids = {r["run_id"] for r in runs}
    main = [r for r in runs if r["target"] == "T0" and r["stage"] == "main"]
    agents = {lv: [r["hs"] for r in main if r["level"] == lv] for lv in ("E3", "E4")}
    n = len(agents["E3"])
    baseline = read("baselines/T0.json")
    h4 = {}
    for name in ("retrains", "students"):
        controls = [r["hs"] for r in baseline[name]]
        tests = {lv: stats.ranksum(x, controls, two_sided=name == "students") for lv, x in agents.items()}
        adj = stats.holm({lv: t["p"] for lv, t in tests.items()})
        for lv, t in tests.items():
            t["holm_p"] = adj[lv]
            t["verdict"] = "cannot tell at this N" if n == 1 else "supported" if adj[lv] < 0.05 else "not supported"
        h4[name] = tests
    planted = read("controls/planted_calibration_T0.json")["rows"]
    assert [r["step"] for r in planted] == [0, 100, 1000, 10000]
    h4b = {
        "rows": planted,
        "identity_exact": planted[0]["hs"] == 1 and planted[0]["kl_out"] == 0,
        "hs_decreases": all(a["hs"] > b["hs"] for a, b in zip(planted, planted[1:])),
        "kl_increases": all(a["kl_out"] < b["kl_out"] for a, b in zip(planted, planted[1:])),
    }
    local = [r for r in runs if r["model"].startswith("qwen")]
    gaps = {}
    for lv in ("E2", "E3", "E4", "W-freq"):
        grades = [
            r["hs"] for r in runs if r["stage"] in ("main", "controls") and r["target"] == "T0" and r["level"] == lv
        ]
        c = read(f"compiler/T0/{lv}.json")["hs"]
        gaps[lv] = {
            "agent_minus_compiler": float(np.mean(grades) - c),
            "compiler_below_interval": min(grades) > c if len(grades) == 3 else None,
        }
    h56 = {
        "T0_compiler": gaps,
        "local_cells": [[r["model"], r["level"], r["accuracy"], r["hs"]] for r in local],
        "required_cells": 12,
        "completed_cells": len({(r["model"], r["level"]) for r in local}),
        "H5_verdict": "not evaluable",
        "H6_verdict": "not evaluable",
    }
    wrong = read("controls/wrong_targets_T0.json")["rows"]
    hs = {r["run_id"]: r["hs"] for r in main}
    margins = {r["run_id"]: hs[r["run_id"]] - max(r["hs_wrong"].values()) for r in wrong if r["run_id"] in ids}
    p = stats.signed_rank(list(margins.values()))
    h7 = {
        "margins": margins,
        "p": p,
        "wrong_targets": sorted(wrong[0]["hs_wrong"]),
        "verdict": "cannot tell at this N"
        if n == 1
        else "supported"
        if p < 0.05 and np.median(list(margins.values())) > 0
        else "not supported",
    }
    tr = read("controls/trajectory_T0.json")
    y = -np.log(tr["restricted_loss"])
    h8 = {
        kind: {r["run_id"]: stats.rank_correlation(r["hs"], y) for r in tr[kind] if r["run_id"] in ids}
        for kind in ("replacements", "negative_controls")
    }
    h8["checkpoints"] = len(y)
    h8["verdict"] = "supported (descriptive)" if min(h8["replacements"].values()) > 0 else "not supported"
    historic = read(
        "controls/trajectory_T0_round1_historical.json" if n == 1 else "controls/trajectory_T0_superseded.json"
    )
    h8["historical"] = {
        "replacement": historic["replacement"],
        "checkpoints": len(historic["rows"]),
        "rho": stats.rank_correlation(
            [r["hs"] for r in historic["rows"]], [-np.log(r["restricted_loss"]) for r in historic["rows"]]
        ),
    }
    return {"H4": h4, "H4b": h4b, "H5_H6": h56, "H7": h7, "H8": h8}


def stimulus_checks():
    """Compare archived text lengths, rendered precision and the target frequencies available in controls."""
    lengths, embed, count = {}, {}, 0
    for target in sorted((RES / "explanations").iterdir()):
        measured = {}
        for length_file in sorted(target.glob("*/length.json")):
            folder = length_file.parent
            prose = (folder / "EXPLANATION.md").read_text(encoding="utf-8")
            table = folder / "neurons.csv"
            size = len(prose) + (len(table.read_text(encoding="utf-8")) if table.exists() else 0)
            assert size == read(length_file.relative_to(RES))["chars"]
            measured[folder.name] = size
            count += 1
        lengths[target.name] = {w: measured[w] / measured[t] for t, w in COMPARISONS}
        truth = set(read(target.relative_to(RES) / "analysis.json")["key_freqs"])
        embed[target.name] = {
            w: sorted(truth & set(read(target.relative_to(RES) / w / "spec.json")["embed_freqs"]))
            for _, w in COMPARISONS
        }
    spec = read("explanations/T0/E3/spec.json")["logit_amplitudes"]
    key = next(iter(spec))
    full = spec[key]
    displayed = f"{full:.3g}"
    assert f"| {key} | {displayed} |" in (RES / "explanations/T0/E3/EXPLANATION.md").read_text(encoding="utf-8")
    return {
        "stimuli": count,
        "character_ratios": lengths,
        "true_embedding_frequencies_in_controls": embed,
        "T0_E3_amplitude_example": {"frequency": key, "compiler": full, "rendered": displayed},
    }


def reconstruct():
    """Compute the dated results, including the split's exact overlap and the matched control ranges."""
    runs, counts = inventory()
    rounds = {}
    for number in (1, 2):
        subset = [r for r in runs if r["round"] <= number]
        rounds[str(number)] = {
            "N": len(subset),
            "run_ids": sorted(r["run_id"] for r in subset),
            "targets": {t: replication(subset, t) for t in TARGETS},
            **other_hypotheses(subset),
        }
    p = 113
    idx = np.arange(p * p)
    fold = np.empty_like(idx)
    for i, block in enumerate(np.array_split(np.random.default_rng(0).permutation(idx), 5)):
        fold[block] = i
    swapped = (idx % p) * p + idx // p
    different = int(np.sum(fold != fold[swapped]))
    eligible = int(np.sum(idx != swapped))
    logs = read("controls/logit_fvu_T0.json")["rows"]
    fv = {
        lv: [r["calibrated_logit_fvu"] for r in logs if r["kind"] == "agent" and r["level"] == lv]
        for lv in ("E3", "W-alg", "E4", "W-scramble")
    }
    wrole = next(r for r in read("controls/clock_pizza.json")["rows"] if r["name"] == "T0 W-role")
    students = read("baselines/T0.json")["students_logit"]
    assert all(s["converged"] == (s["centred_logit_fvu"] < 0.05) for s in students)
    return {
        "frozen_protocol": "801edf013db7239f8994e6780c5873e1890f5103",
        "inventory": counts,
        "rounds": rounds,
        "fold_overlap": {
            "ordered_off_diagonal": eligible,
            "swap_in_training": different,
            "fraction": different / eligible,
        },
        "coverage": stats.bootstrap_coverage(),
        "stimuli": stimulus_checks(),
        "scale_adjusted_fvu": fv,
        "E3_Walg_complete_separation": max(fv["E3"]) < min(fv["W-alg"]),
        "W_role_compiler": {
            "argmax": read("compiler/T0/W-role.json")["accuracy"],
            "near_tie": wrole["within_tolerance_of_max"],
        },
        "logit_students": [{k: s[k] for k in ("seed", "hs", "centred_logit_fvu", "converged")} for s in students],
    }


def main():
    data = reconstruct()
    write(AUDIT / "reconstruction.json", data)
    return data


if __name__ == "__main__":
    main()
