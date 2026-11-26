"""Regression checks against frozen run IDs and recorded outcomes. All inputs are numbers or text."""

import json

import pytest

from vmi.analysis import hypotheses, predictions, report
from vmi.config import REPO_ROOT


@pytest.mark.parametrize(("round_number", "n", "replacements"), [(1, 37, 2), (2, 101, 6)])
def test_round_membership_and_historical_trajectory(round_number, n, replacements):
    """A later control file must never add replacement IDs to an earlier round."""
    data = hypotheses.evaluate(round_number)
    assert data["n_runs"] == n
    h = data["hypotheses"]
    assert len(h["H7"]["rows"]) == len(h["H8"]["replacements"]) == replacements
    assert len(h["H8"]["steps"]) == 401
    if round_number == 1:
        assert set(h["H2"]["verdicts_prereg"].values()) == {"cannot tell at this N"}
        assert h["H8"]["historical"]["checkpoints"] == 41


def test_replication_endpoints_and_published_checkpoint():
    rows = predictions.replication_rows(2)
    paper = next(r for r in rows if r["target"].startswith("T-paper"))
    assert "0.666" in paper["h1_kl"]
    assert not paper["h3_interval"] and paper["h3_upper"]
    assert "E2 missing" in paper["h3"]
    assert next(r for r in rows if r["target"] == "T2")["h3_interval"]


def test_saved_blocks_do_not_execute_networks(monkeypatch):
    """The interface must also work in a checkout with no external run folders."""
    from vmi.analysis import examples

    def forbidden(*args, **kwargs):
        pytest.fail("A saved-only report reached a network or forward-pass helper")

    monkeypatch.setattr(examples, "panel_models", forbidden)
    monkeypatch.setattr(examples, "output_rows", forbidden)
    generated = report.blocks(2, saved_only=True)
    assert "12,769" in generated["outputs-T0"]
    assert "0.0065" in generated["clock-pizza-T0"]


def test_published_top1_matches_primary_record():
    base = REPO_ROOT / "results"
    diagnostics = json.loads((base / "controls/clock_pizza.json").read_text())
    row = next(r for r in diagnostics["rows"] if r["name"] == "T0 W-role")
    grade = json.loads((base / "compiler/T0/W-role.json").read_text())
    assert row["accuracy"] == grade["accuracy"]
    assert row["within_tolerance_of_max"] > row["accuracy"]


@pytest.mark.parametrize("number", [1, 2])
def test_independent_estimates_agree_with_reporting(number):
    """Compare two implementations, using run identities and numeric estimates before display rounding."""
    import numpy as np
    from vmi.analysis import stats

    audit = json.loads((REPO_ROOT / "results/scientific-audit/reconstruction.json").read_text(encoding="utf-8"))
    independent = audit["rounds"][str(number)]
    runs = hypotheses.load_runs(number)
    normal = hypotheses.evaluate(number)["hypotheses"]
    assert set(independent["run_ids"]) == {r["run_id"] for r in runs}
    for target, rep in independent["targets"].items():
        relevant = [
            r
            for r in runs
            if r["target"] == target
            and r["model"] == hypotheses.MAIN_MODEL
            and not r["variant"]
            and r.get("temperature") is None
        ]
        assert {r["run_id"] for r in relevant} == set(rep["run_ids"])
        h1 = rep["H1"]
        for endpoint, field, sign in (("hs", "hs", 1), ("decreasing_kl", "kl_out", -1)):
            groups = [np.array([sign * r[field] for r in relevant if r["level"] == lv]) for lv in h1["levels"]]
            jt = stats.jonckheere(groups)
            corr = stats.spearman_ci(
                np.repeat(np.arange(len(groups)), list(map(len, groups))), np.concatenate(groups), groups
            )
            assert (jt["J"], jt["p"]) == pytest.approx((h1[endpoint]["J"], h1[endpoint]["p"]), abs=1e-14)
            assert corr["rho"] == pytest.approx(h1[endpoint]["rho"], abs=1e-13)
            if number == 2:
                np.testing.assert_allclose(corr["ci"], h1[endpoint]["ci"], rtol=0, atol=1e-12)
        if target != "T0":
            display = next(x for x in predictions.replication_rows(number) if x["target"].split()[0] == target)
            for kind, key in (("interval", "interval_reading"), ("upper-bound reading", "upper_bound_reading")):
                assert f"{kind} {'passes' if rep['H3'][key] else 'fails'}" in display["h3"]
    main = independent["targets"]["T0"]
    for name, c in main["H2"].items():
        assert normal["H2"]["verdicts_prereg"][name] == c["verdict"]
        assert normal["H2"]["effects"][name]["diff"] == pytest.approx(c["mean_difference"])
        if number == 2:
            assert normal["H2"]["holm_p"][name] == pytest.approx(c["holm_p"])
            assert normal["H2"]["tests"][name]["r_rb"] == pytest.approx(c["rank_biserial"])
    assert normal["H3"]["wfreq"]["kl"] == pytest.approx(main["H3"]["wrong_frequency_kl"])
    assert (normal["H3"]["verdict"] == "supported") == main["H3"]["interval_reading"]
    for lv in ("E3", "E4"):
        if number == 2:
            for family, actual in (
                ("retrains", normal["H4"]["holm_p"][lv]),
                ("students", normal["H4"]["students"][lv]["holm_p"]),
            ):
                assert actual == pytest.approx(independent["H4"][family][lv]["holm_p"])
    assert normal["H4b"]["identity_exact"] == independent["H4b"]["identity_exact"]
    assert not independent["H4b"]["kl_increases"]
    assert all(normal[k]["verdict"].startswith("not evaluable") for k in ("H5", "H6"))
    assert normal["H7"]["wilcoxon"]["p"] == pytest.approx(independent["H7"]["p"])
    assert [r["d"] for r in normal["H7"]["rows"]] == pytest.approx(
        [independent["H7"]["margins"][r["run_id"]] for r in normal["H7"]["rows"]]
    )
    assert normal["H8"]["spearman_min"] == pytest.approx(min(independent["H8"]["replacements"].values()))


def test_accuracy_definitions_and_discrepant_claims():
    base = REPO_ROOT / "results"
    audit = json.loads((base / "scientific-audit/reconstruction.json").read_text())
    assert audit["inventory"]["maximum_hs_reconstruction_error"] < 1.2e-16
    assert audit["E3_Walg_complete_separation"] and audit["fold_overlap"]["swap_in_training"] == 10236
    for row in json.loads((base / "controls/clock_pizza.json").read_text(encoding="utf-8"))["rows"]:
        origin = json.loads((REPO_ROOT / row["accuracy_source"]).read_text(encoding="utf-8"))
        assert row["accuracy"] == origin["target_accuracy" if row["kind"] == "target" else "accuracy"]
