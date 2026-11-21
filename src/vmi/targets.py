"""Target models: the preregistered registry (T0, T1, T2, dev, T-paper, T-new) and loaders.

Local explanation: targets are final checkpoints (step 40,000) of runs trained with the paper's recipe. Runs from
`BurnyCoder/grokking-modular-addition-runs` are read with grokviz's `RunStore`; the paper's own checkpoint is downloaded
at a pinned revision, sha256-checked and loaded with `torch.load(weights_only=True)` (tensors and plain containers only).
Global context: the explainer reads targets (it may see weights); the simulator never does — only the grader and
explainer import this module, and the sandbox has no path to `data_dir`.
Sources: grokviz RunStore (MIT, https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer);
paper checkpoint layout (paper notebook Grokking_Analysis.ipynb; state_dicts[400] = epoch 40,000, saved every 100);
torch.load weights_only (https://pytorch.org/docs/stable/generated/torch.load.html).
"""

from __future__ import annotations  # postponed annotations

import hashlib  # sha256 of the downloaded paper checkpoint
import json  # published key_freqs.json
from dataclasses import dataclass  # target descriptor
from pathlib import Path  # paths

import torch  # checkpoint loading
from grokviz.metrics import Grid  # all p² inputs, labels, split masks, Fourier basis
from grokviz.model import GrokTransformer, ModelShape  # paper-faithful network
from grokviz.runstore import RunStore  # lazy access to stacked checkpoints

from vmi.config import FRAC_TRAIN, P, Settings  # constants and paths

PAPER_CKPT_INDEX = 400  # paper run saved every 100 epochs → index 400 is epoch 40,000 (the analysed model)


@dataclass(frozen=True)
class Target:
    """A target model: where it comes from and which data split it was trained on."""

    name: str  # registry name, e.g. "T0"
    run_id: str  # run folder name ("wd1_s1") or "paper"
    split_seed: int  # seed of the paper-style 30 % train split
    fn_name: str = "add"  # task: "add" or "rand" (random-label control)
    label_seed: int | None = None  # seed of the random label table (rand runs only)


# Preregistered roles (reports/preregistration.md §2). Every other run is addressable by its run_id.
REGISTRY: dict[str, str] = {"T0": "wd1_s1", "T1": "wd1_s4", "T2": "wd1_s8", "dev": "wd1_s10", "T-paper": "paper", "T-new": "new_s11"}


def resolve(name_or_run: str) -> Target:
    """Map a registry name or a run id to a Target with its split seed (read from the run's naming convention)."""
    run_id = REGISTRY.get(name_or_run, name_or_run)
    if run_id == "paper":
        return Target(name_or_run, "paper", split_seed=0)  # paper code: random.seed(0) before the shuffle
    if run_id.startswith("split0_init"):
        return Target(name_or_run, run_id, split_seed=0)  # paper split with a different initialisation
    if run_id.startswith("rand_s"):
        s = int(run_id.split("_s")[1])
        return Target(name_or_run, run_id, split_seed=s, fn_name="rand", label_seed=s)
    s = int(run_id.rsplit("_s", 1)[1])  # wd1_sX, wd0_sX, new_sX: init seed = split seed = X
    return Target(name_or_run, run_id, split_seed=s)


def run_dir(settings: Settings, run_id: str, prefer_trained: bool = False) -> Path:
    """Folder of a complete run: downloaded runs live in data/grokking-runs, runs trained here in data/trained."""
    bases = [settings.runs_store, settings.data_dir / "trained"]
    for base in (bases[::-1] if prefer_trained else bases):
        if (base / run_id / "meta.json").exists() and (base / run_id / "checkpoints.safetensors").exists():
            return base / run_id
    raise FileNotFoundError(f"run {run_id} not found under {settings.runs_store} or {settings.data_dir / 'trained'}")


def paper_checkpoint(settings: Settings) -> Path:
    """Download (once) the paper's full_run_data.pth at the pinned revision and verify its sha256."""
    from huggingface_hub import hf_hub_download  # imported lazily: only needed for the paper target

    path = Path(hf_hub_download(settings.paper_repo, "full_run_data.pth", revision=settings.paper_revision,
                                local_dir=settings.data_dir / "paper"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != settings.paper_sha256:
        raise ValueError(f"paper checkpoint sha256 {digest} != pinned {settings.paper_sha256}")
    return path


def load_model(settings: Settings, name_or_run: str, device: str | torch.device = "cpu", ckpt: int = -1) -> GrokTransformer:
    """Load a target's checkpoint (default: final) into GrokTransformer (strict state-dict load)."""
    t = resolve(name_or_run)
    if t.run_id == "paper":
        data = torch.load(paper_checkpoint(settings), map_location="cpu", weights_only=True)
        sd = data["state_dicts"][PAPER_CKPT_INDEX if ckpt == -1 else ckpt]
        model = GrokTransformer(ModelShape())
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval()
    return RunStore(run_dir(settings, t.run_id)).load_model(ckpt, device)


def grid(name_or_run: str, device: str | torch.device = "cpu") -> Grid:
    """All 113² inputs with the target's labels and train/test split (float64 Fourier basis on `device`)."""
    t = resolve(name_or_run)
    return Grid.build(P, FRAC_TRAIN, t.split_seed, device, fn_name=t.fn_name, label_seed=t.label_seed)


def published_key_freqs(settings: Settings, name_or_run: str) -> list[int] | None:
    """Key frequencies recorded with the downloaded run (analysis/key_freqs.json), or None if absent."""
    t = resolve(name_or_run)
    if t.run_id == "paper":
        return [14, 35, 41, 42, 52]  # Nanda et al. 2023 §4.1
    f = run_dir(settings, t.run_id) / "analysis" / "key_freqs.json"
    return json.loads(f.read_text())["key_freqs"] if f.exists() else None
