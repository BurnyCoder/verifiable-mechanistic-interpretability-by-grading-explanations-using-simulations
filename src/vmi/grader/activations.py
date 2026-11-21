"""Run a model once on all 113² inputs and keep what the grader compares, at the '=' (last) position.

Local explanation: `collect` returns float64 tensors: the 113 number logits, the three hidden-state sites of the
primary metric (resid_mid, mlp_post, resid_post), neuron pre-activations and the attention pattern from '='.
Global context: every metric in `metrics.py` takes two of these `Acts` objects (target and replacement), so each
model is run exactly once per grading.
Sources: grokviz GrokTransformer hook names (blocks.0.resid_mid, blocks.0.mlp.post, blocks.0.resid_post,
blocks.0.attn.pattern); paper read-out convention logits[:, -1, :-1] (paper code / ARENA).
"""

from __future__ import annotations  # postponed annotations

from dataclasses import dataclass  # container

import torch  # tensors
from grokviz.model import GrokTransformer  # model with named hooks
from grokviz.task import all_pairs  # the p² token triples, row a·p + b

from vmi.config import P  # modulus


@dataclass
class Acts:
    """Activations of one model on every input, at the '=' position (float64)."""

    logits: torch.Tensor  # [p², 113] number logits ('=' class dropped)
    resid_mid: torch.Tensor  # [p², 128] residual stream after attention
    mlp_post: torch.Tensor  # [p², 512] neuron activations after ReLU
    resid_post: torch.Tensor  # [p², 128] residual stream after the MLP (input to the unembedding)
    mlp_pre: torch.Tensor  # [p², 512] neuron pre-activations
    pattern: torch.Tensor  # [p², 4, 3] attention from '=' to (a, b, '=') per head

    def site(self, name: str) -> torch.Tensor:
        """Hidden-state site by name (the three primary sites are 'resid_mid', 'mlp_post', 'resid_post')."""
        return getattr(self, name)


SITES = ("resid_mid", "mlp_post", "resid_post")  # preregistered primary hidden-state sites


@torch.no_grad()
def collect(model: GrokTransformer, device: str | torch.device | None = None) -> Acts:
    """Forward all 113² inputs in one batch and return the '='-position activations in float64."""
    dev = device or next(model.parameters()).device
    tokens = all_pairs(P).to(dev)  # [12769, 3]
    logits, cache = model.to(dev).run_with_cache(tokens)
    return Acts(
        logits=logits[:, -1, :P].double(),
        resid_mid=cache["blocks.0.resid_mid"][:, -1].double(),
        mlp_post=cache["blocks.0.mlp.post"][:, -1].double(),
        resid_post=cache["blocks.0.resid_post"][:, -1].double(),
        mlp_pre=cache["blocks.0.mlp.pre"][:, -1].double(),
        pattern=cache["blocks.0.attn.pattern"][:, :, -1, :].double(),
    )
