"""The agent's numpy reference forward pass must equal the grader's torch model exactly (up to float precision).

Global context: if forward.py disagreed with grokviz's GrokTransformer, an agent could pass its self-test and still be
graded on a different network. This test pins the two implementations together.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from grokviz.model import GrokTransformer
from grokviz.task import all_pairs

from vmi import weights

TEMPLATE = Path(__file__).resolve().parents[1] / "sandbox_template"


def _load_forward():
    spec = importlib.util.spec_from_file_location("sandbox_forward", TEMPLATE / "forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numpy_forward_matches_torch():
    fwd = _load_forward()
    model = GrokTransformer(seed=3).eval()
    with torch.no_grad():
        model.blocks[0].mlp.b_in.normal_(0, 0.2)
        logits_t, cache = model.run_with_cache(all_pairs(113))
    w = weights.from_model(model)
    acts = fwd.forward(w, fwd.all_inputs())
    assert np.allclose(acts["logits"], logits_t.double().numpy(), atol=1e-4)
    assert np.allclose(acts["resid_mid"], cache["blocks.0.resid_mid"].double().numpy(), atol=1e-4)
    assert np.allclose(acts["mlp_post"], cache["blocks.0.mlp.post"].double().numpy(), atol=1e-4)
    assert np.allclose(acts["pattern"], cache["blocks.0.attn.pattern"].double().numpy(), atol=1e-6)
    assert (fwd.all_inputs() == all_pairs(113).numpy()).all()


def test_selftest_reports_shapes_and_accuracy(tmp_path):
    w = weights.from_model(GrokTransformer(seed=4))
    np.savez(tmp_path / "weights.npz", **w)
    out = subprocess.run([sys.executable, str(TEMPLATE / "selftest.py"), str(tmp_path / "weights.npz")],
                         capture_output=True, text=True, cwd=TEMPLATE)
    assert out.returncode == 0 and "shapes OK" in out.stdout
    w["W_U"] = w["W_U"][:, :113]
    np.savez(tmp_path / "bad.npz", **w)
    out = subprocess.run([sys.executable, str(TEMPLATE / "selftest.py"), str(tmp_path / "bad.npz")],
                         capture_output=True, text=True, cwd=TEMPLATE)
    assert out.returncode == 1 and "W_U" in out.stdout


def test_starter_build_script_writes_valid_zero_weights(tmp_path):
    out = subprocess.run([sys.executable, str(TEMPLATE / "build_weights.py")], capture_output=True, text=True,
                         cwd=tmp_path)
    assert out.returncode == 0
    w = weights.load_npz(tmp_path / "weights.npz")  # raises if a name or shape is wrong
    assert all(not v.any() for v in w.values())


def test_prepare_run_fills_budget_and_strips_selftest_in_ablation(tmp_path):
    from vmi.config import load
    from vmi.simulator.sandbox import prepare_run

    settings = load()
    full = prepare_run(settings, tmp_path / "full", explanation="x", extra_files={}, meta={})
    task = (full / "work" / "TASK.md").read_text(encoding="utf-8")
    assert "{" not in task and f"{settings.run_timeout_s // 60} minutes" in task
    assert (full / "work" / "build_weights.py").exists() and (full / "work" / "selftest.py").exists()
    ablated = prepare_run(settings, tmp_path / "noself", explanation="x", extra_files={}, meta={}, selftest=False)
    task = (ablated / "work" / "TASK.md").read_text(encoding="utf-8").lower()
    assert "selftest" not in task and "self-test" not in task
    assert not (ablated / "work" / "selftest.py").exists()
    assert "selftest" not in (ablated / "prompt.txt").read_text(encoding="utf-8")
