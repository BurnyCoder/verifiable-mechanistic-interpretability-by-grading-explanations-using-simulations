"""Training jobs: fresh runs with grokviz's deterministic trainer, plus distillation and continued training.

Local explanation:
- `train_fresh` trains runs with the paper's official recipe through `grokviz.train.train_runs` unchanged
  (vectorised over runs, deterministic, resumable every 2,000 steps) into `data/trained/<run_id>` — used for the
  reproduction check of T0 and for the held-out target T-new.
- `train_from` is one small full-batch AdamW loop with a pluggable loss and initial weights: label mode (task labels,
  must reproduce a published run before use), distillation mode (KL to a target's logits on all inputs) and
  continued training of a compiled model (planted-mechanism calibration). Snapshots are saved as weight files.
Global context: these models are baselines and controls (preregistration §7); the simulator never trains anything.
Sources: grokviz train_runs (MIT; paper recipe: AdamW lr 1e-3, betas (0.9, 0.98), wd 1, 10-step warm-up, float64 CE,
https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64/transformers.py);
knowledge distillation with a KL loss on logits — Hinton et al. 2015 (https://arxiv.org/abs/1503.02531);
PyTorch determinism notes (https://pytorch.org/docs/stable/notes/randomness.html).
"""

from __future__ import annotations  # postponed annotations

import os  # cuBLAS determinism variable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by torch.use_deterministic_algorithms on CUDA

from pathlib import Path  # noqa: E402 — after the environment variable on purpose

import numpy as np  # noqa: E402
import torch  # noqa: E402
from grokviz.config import Config  # noqa: E402
from grokviz.losses import cross_entropy_f64  # noqa: E402
from grokviz.task import all_pairs, make_labels, paper_split  # noqa: E402
from grokviz.train import RunSpec, train_runs  # noqa: E402

from vmi import weights  # noqa: E402
from vmi.config import FRAC_TRAIN, P, Settings  # noqa: E402


def grokviz_config(settings: Settings) -> Config:
    """grokviz Config with the paper's recipe (same values as grokviz's documented defaults at commit d52a898)."""
    trained = settings.data_dir / "trained"
    return Config(device="auto", runs_dir=trained, data_dir=settings.data_dir, site_dir=trained / "_site",
                  logs_dir=trained / "_logs", reports_dir=trained / "_reports", p=P, frac_train=FRAC_TRAIN, d_model=128,
                  d_mlp=512, n_heads=4, d_head=32, n_ctx=3, lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98,
                  warmup_steps=10, num_steps=40000, ckpt_every=100, optim_every=1000, norms_every=10, test_every=10,
                  wd1_seeds=[], wd0_seeds=[], rand_seeds=[], split0_init_seeds=[], paper_hf_repo=settings.paper_repo,
                  paper_hf_file="full_run_data.pth", paper_hf_revision=settings.paper_revision,
                  paper_sha256=settings.paper_sha256, paper_pdf_url="", power_pdf_url="",
                  hf_runs_repo=settings.hf_runs_repo, hf_site_data_repo="", hf_collection_title="", site_weight_snapshots=[])


def train_fresh(settings: Settings, run_ids: list[str], device: str = "cuda") -> None:
    """Train fresh runs ("wd1_sX" or "new_sX": init seed = split seed = X, weight decay 1) with grokviz's trainer."""
    cfg = grokviz_config(settings)
    specs = [RunSpec(r, r.rsplit("_s", 1)[0], int(r.rsplit("_s", 1)[1]), int(r.rsplit("_s", 1)[1]), 1.0) for r in run_ids]
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    train_runs(cfg, specs, torch.device(device), cfg.runs_dir)


def train_from(init: dict[str, np.ndarray], *, mode: str, steps: int, snapshots: list[int], out_dir: Path,
               teacher_logits: torch.Tensor | None = None, split_seed: int = 0, device: str = "cuda") -> dict[int, dict]:
    """Full-batch AdamW (paper recipe) from given weights; returns {step: weight arrays} at the snapshot steps.

    mode "labels": cross-entropy on the paper-style training split (seed `split_seed`), 114 output classes as in the
    paper's code. mode "distill": KL(teacher ‖ student) on the 113 number logits at '=' for all 113² inputs.
    """
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = weights.to_model(init, device).train()
    for p_ in model.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))  # 10-step linear warm-up
    tokens = all_pairs(P).to(device)
    if mode == "labels":
        split = paper_split(P, FRAC_TRAIN, split_seed)
        x, y = tokens[split.train_idx.to(device)], make_labels(P)[split.train_idx].to(device)
    else:
        assert teacher_logits is not None
        x = tokens
        t_logp = torch.log_softmax(teacher_logits.to(device).double(), -1)
    out_dir.mkdir(parents=True, exist_ok=True)
    snaps: dict[int, dict] = {}
    for step in range(steps + 1):
        if step in snapshots:
            arrays = weights.from_model(model)
            np.savez(out_dir / f"step_{step:06d}.npz", **arrays)
            snaps[step] = arrays
        if step == steps:
            break
        logits = model(x, last_only=True)[:, -1]
        if mode == "labels":
            loss = cross_entropy_f64(logits, y)
        else:
            s_logp = torch.log_softmax(logits[:, :P].double(), -1)
            loss = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return snaps
