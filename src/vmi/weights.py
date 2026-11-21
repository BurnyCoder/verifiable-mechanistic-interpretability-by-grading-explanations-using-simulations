"""The weight file format shared by the agent, the reference compiler and the grader.

Local explanation: a replacement model is a `weights.npz` with exactly 11 named float arrays in the paper's layout
(`SCHEMA`). `validate` checks names, shapes and finiteness; `to_model` loads them into grokviz's `GrokTransformer`
(strict state-dict load), and `from_model` does the reverse for targets and baselines.
Global context: the sandbox's ARCHITECTURE.md documents the same schema to the agent, so the grader can load any
agent's output without conversion; all symmetry handling happens in the grader's metrics, not here.
Sources: paper code layouts (https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64/transformers.py);
numpy .npz format (https://numpy.org/doc/stable/reference/generated/numpy.savez.html).
"""

from __future__ import annotations  # postponed annotations

from pathlib import Path  # file paths

import numpy as np  # arrays
import torch  # state dicts
from grokviz.model import GrokTransformer, ModelShape  # paper-faithful model with hooks

from vmi.config import D_HEAD, D_MLP, D_MODEL, D_VOCAB, N_CTX, N_HEADS  # architecture constants

# name in weights.npz → (state-dict key in GrokTransformer, required shape)
SCHEMA: dict[str, tuple[str, tuple[int, ...]]] = {
    "W_E": ("embed.W_E", (D_MODEL, D_VOCAB)),  # column t = embedding of token t (t = 113 is '=')
    "W_pos": ("pos_embed.W_pos", (N_CTX, D_MODEL)),  # row i added at position i
    "W_Q": ("blocks.0.attn.W_Q", (N_HEADS, D_HEAD, D_MODEL)),  # q_h = W_Q[h] @ x
    "W_K": ("blocks.0.attn.W_K", (N_HEADS, D_HEAD, D_MODEL)),  # k_h = W_K[h] @ x
    "W_V": ("blocks.0.attn.W_V", (N_HEADS, D_HEAD, D_MODEL)),  # v_h = W_V[h] @ x
    "W_O": ("blocks.0.attn.W_O", (D_MODEL, N_HEADS * D_HEAD)),  # out = W_O @ concat_h(z_h)
    "W_in": ("blocks.0.mlp.W_in", (D_MLP, D_MODEL)),  # pre = W_in @ x + b_in
    "b_in": ("blocks.0.mlp.b_in", (D_MLP,)),
    "W_out": ("blocks.0.mlp.W_out", (D_MODEL, D_MLP)),  # mlp_out = W_out @ relu(pre) + b_out
    "b_out": ("blocks.0.mlp.b_out", (D_MODEL,)),
    "W_U": ("unembed.W_U", (D_MODEL, D_VOCAB)),  # logits = x @ W_U
}


class WeightsError(ValueError):
    """Raised when a weight file is missing arrays, has wrong shapes or non-finite values."""


def validate(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return float64 copies of the 11 required arrays, or raise WeightsError explaining the first problem."""
    out = {}
    for name, (_, shape) in SCHEMA.items():
        if name not in arrays:
            raise WeightsError(f"missing array {name}")
        a = np.asarray(arrays[name], dtype=np.float64)  # accept any real dtype; analyse in float64
        if a.shape != shape:
            raise WeightsError(f"{name} has shape {a.shape}, expected {shape}")
        if not np.all(np.isfinite(a)):
            raise WeightsError(f"{name} contains NaN or Inf")
        out[name] = a
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a weights.npz (allow_pickle=False: never execute pickled objects from an agent's file)."""
    with np.load(path, allow_pickle=False) as f:
        return validate({k: f[k] for k in f.files})


def to_model(arrays: dict[str, np.ndarray], device: str | torch.device = "cpu") -> GrokTransformer:
    """Build a GrokTransformer holding these weights (strict load; the causal-mask buffer comes from the class)."""
    model = GrokTransformer(ModelShape())  # random init, immediately overwritten below
    sd = model.state_dict()  # includes the 'mask' buffer, which is fixed and not part of the schema
    for name, (key, _) in SCHEMA.items():
        sd[key] = torch.as_tensor(arrays[name], dtype=torch.float32)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def from_model(model: GrokTransformer) -> dict[str, np.ndarray]:
    """Extract the 11 schema arrays (float64 numpy) from a model, e.g. a target or a trained baseline."""
    sd = model.state_dict()
    return {name: sd[key].detach().double().cpu().numpy() for name, (key, _) in SCHEMA.items()}


def zeros() -> dict[str, np.ndarray]:
    """The all-zero model: uniform output distribution; the score assigned to failed runs (preregistration §4)."""
    return {name: np.zeros(shape) for name, (_, shape) in SCHEMA.items()}
