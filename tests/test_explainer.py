"""Tests for the explanation ladder and rendering on a synthetic analysis dict (no target needed).

Global context: checks the preregistered properties of the explanations — false controls really are false in the stated
way (disjoint frequencies, scrambled neurons), length-matched controls use the same template, only E4/W-scramble ship
a table, E0/E1 contain no model-specific numbers — and that the reference compiler builds working models from them.
"""

import numpy as np
import pytest

from vmi.explainer.ladder import LEVELS, make_spec
from vmi.explainer.render import length_stats, render
from vmi.reference.compile import compile_spec

K = [4, 19, 33, 48]


def _fake_analysis() -> dict:
    rng = np.random.default_rng(0)
    neurons = [{"neuron": i, "freq": int(K[i % 4]), "bias": 0.1, "a_amp": 1.0, "a_phase": float(rng.uniform(0, 360)),
                "b_amp": 1.0, "b_phase": float(rng.uniform(0, 360)), "sum_amp": 0.2, "sum_phase": 10.0, "diff_amp": 0.1,
                "diff_phase": 20.0, "pre_fit_fve": 0.9, "out_amp": 0.5, "out_phase": 30.0, "out_fit_fve": 0.95} for i in range(512)]
    return {"key_freqs": K, "logit_amplitudes": {k: 12.0 for k in K}, "logit_offset": 0.0, "logit_fit_fve": 0.95,
            "loss_all_inputs": 3e-7, "mean_correct_logprob": -3e-7, "logit_std": 9.0,
            "embedding_norm_by_freq": {k: 3.0 for k in K}, "neuron_clusters": {k: 100 for k in K}, "neurons_unclustered": 112,
            "neuron_fve_threshold": 0.85,
            # head pattern as in trained models (paper Table 2; dev target): two heads with opposite coefficients at one
            # frequency, two others at single frequencies
            "heads": [{"head": h, "freq": f, "alpha": a, "beta": b, "pattern_fve": 0.95, "attn_to_eq": 0.002,
                       "C_fve": 0.97, "ov_norm_by_freq": {k: 1.0 for k in K}}
                      for h, (f, a, b) in enumerate([(K[1], 0.13, 0.22), (K[2], -0.125, -0.27), (K[1], -0.13, -0.22),
                                                     (K[0], -0.02, -0.28)])],
            "readoff": [{"freq": k, "component": c, "coef_unit_dir": 40.0, "fve": 0.95} for k in K for c in ("cos", "sin")],
            "wl_residual_sq_frac": 0.005, "neurons": neurons}


@pytest.fixture(scope="module")
def specs():
    an = _fake_analysis()
    return {lv: make_spec(an, lv, "fake") for lv in LEVELS}


def test_wrong_frequency_controls_are_disjoint(specs):
    assert set(specs["W-freq"]["embed_freqs"]).isdisjoint(K) and len(specs["W-freq"]["embed_freqs"]) == len(K)
    assert specs["W-role"]["embed_freqs"] == K and set(specs["W-role"]["mlp_freqs"]).isdisjoint(K)
    assert specs["W-alg"]["algorithm"] == "pizza" and specs["W-alg"]["embed_freqs"] == K


def test_scramble_changes_neuron_table_only(specs):
    e4, ws = specs["E4"]["neurons"], specs["W-scramble"]["neurons"]
    assert any(a["freq"] != b["freq"] for a, b in zip(e4, ws))
    assert all(a["a_amp"] == b["a_amp"] for a, b in zip(e4, ws))


def test_only_tables_for_e4_levels_and_no_numbers_in_e0_e1(specs):
    for lv in LEVELS:
        text, extra = render(specs[lv])
        assert ("neurons.csv" in extra) == (lv in ("E4", "W-scramble"))
        if lv in ("E0", "E1"):
            assert not any(str(k) in text.split("Algorithm")[-1] for k in K if k > 9)  # no key frequency leaks


def test_true_and_wrong_algorithms_state_the_neuron_function_equally_precisely(specs):
    # stage-0 finding: "neurons form products" was vaguer than the Pizza text's ReLU(cos(w a + φ) + cos(w b + φ))
    for lv in ("E1", "E2", "E3", "E4", "W-freq", "W-role", "W-scramble", "W-alg"):
        text = render(specs[lv])[0]
        assert "cos(w_k a + φ) + cos(w_k b + φ)" in text, lv
    from vmi.explainer.render import CLOCK_ALGO_PAPER
    assert "form products" in CLOCK_ALGO_PAPER and "+ φ" not in CLOCK_ALGO_PAPER  # E-paper keeps the paper's wording


def test_length_matched_controls(specs):
    e3 = length_stats(*render(specs["E3"]))["chars"]
    for lv in ("W-freq", "W-role"):
        assert abs(length_stats(*render(specs[lv]))["chars"] - e3) / e3 < 0.1


@pytest.mark.parametrize("level", ["E1", "E2", "E3", "W-freq", "W-alg"])
def test_compiler_builds_working_models(specs, level):
    import importlib.util
    from pathlib import Path
    spec_ = importlib.util.spec_from_file_location("f", Path(__file__).resolve().parents[1] / "sandbox_template" / "forward.py")
    f = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(f)
    W = compile_spec(specs[level])
    acc = (f.forward(W, f.all_inputs())["logits"][:, -1, :113].argmax(-1) == f.labels()).mean()
    assert acc > 0.99
