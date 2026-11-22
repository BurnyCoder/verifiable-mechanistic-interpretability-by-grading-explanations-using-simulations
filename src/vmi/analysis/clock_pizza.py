"""Clock-versus-Pizza diagnostics of Zhong et al. (2023) for the targets and for literal Clock/Pizza builds (post hoc).

Local explanation:
- `distance_irrelevance(L)`: Zhong et al. Def. 4.2 (§4.1.2). L is the correct-logit matrix, L[a, b] = logit of the
  answer (a + b) mod p for input (a, b). q = mean over differences d of std_a L[a, a + d], divided by std of L over all
  p² inputs. Clock's logits do not depend on a − b (q near 1); Pizza's carry the factor |cos(w_k(a − b)/2)|, so within
  one difference the logits barely vary (q near 0).
- `gradient_symmetricity(model)`: Zhong et al. Def. 4.1 (§4.1.1). For random (a, b, c), the cosine similarity of the
  gradients of logit c with respect to the input embeddings of a and of b, averaged; Pizza's constant attention makes
  the two gradients nearly equal (their Pizza Model A: 0.994; Fig. 7's phase lines are fitted to the label > 0.98), while Clock's multiplication
  makes them asymmetric (§2.2). Zhong et al.'s reference Pizza (Model A) has constant attention, but Pizza also arises at
  intermediate attention rates (Fig. 7) and once with full attention (App. H); our hand-built replacements compute their (near-constant) attention from the
  embeddings, so the gradient through the attention scores points off the data manifold and in opposite directions
  for a and b, and the full gradient gives strongly negative values. `fixed_attention=True` (a variant, not Zhong et
  al.'s definition) holds the attention pattern fixed when differentiating; observed after the first run (literal
  Pizza build: −0.87 full, +1.00 with the pattern fixed), so the reading rule below uses distance irrelevance only,
  which Zhong et al. call the decisive factor (§4.1.3).
Global context: audit 3 (2026-11-22). The W-alg control assumes that T0 is a Clock model, and the "negative result"
that HS does not separate W-alg from E3 rests on that assumption, which no earlier analysis tested. Reading rule, taken
from the source (audit 5: this module and its first results were committed together in e9b0b0d/1586324, so git cannot
confirm that the rule preceded the values; its safeguard is that it is Zhong et al.'s own range): Zhong et al. call
distance irrelevance 0–0.4 typical of Pizza and 0.4–1 typical of
Clock (§4.1.2), and their Fig. 7 phase-transition lines, drawn in attention-rate × width space, are logistic-regression
fits to two labels per run, gradient symmetricity > 98 % and distance irrelevance < 0.6 (one fit per label, not a joint
rule, and not lines on either metric; audit 6). The W-alg negative result stands if T0 is on the Clock side (q ≥ 0.4) and the literal
Pizza build (reference compiler, W-alg) is on the Pizza side (q < 0.4); otherwise the paper must say that W-alg may
not be a wrong explanation of T0. Zhong et al. used p = 59 and inputs [a, b]; here p = 113 with an "=" token and
causal attention, so the compiler's literal builds of the true explanation (E3) and of the Pizza explanation (W-alg)
are reported as in-setting reference points. Audit 4: the literal E3 build is itself on the Pizza side (q ≈ 0.18), so
it is a build of the rendered true explanation, not a Clock reference; no Clock-side build of an explanation exists.
Post hoc; no preregistered verdict uses this file.
Sources: Zhong et al. 2023, The Clock and the Pizza, Defs. 4.1–4.2, §4.1.2 ranges, Fig. 7
(https://arxiv.org/abs/2306.17844); gradients by torch.autograd.grad
(https://pytorch.org/docs/stable/generated/torch.autograd.grad.html).
"""

from __future__ import annotations  # postponed annotations

import json  # results file
from pathlib import Path  # paths

import numpy as np  # std, random triples
import torch  # forward passes and autograd
import torch.nn.functional as F  # cosine similarity

from vmi import targets, weights  # target checkpoints, weights.npz → model
from vmi.analysis.hypotheses import MAIN_MODEL, load_runs  # reported agent runs
from vmi.config import REPO_ROOT, Settings  # paths

RES = REPO_ROOT / "results"
P = 113  # modulus; token P is "="
N_GRAD = 100  # Zhong et al. footnote 3: gradient symmetricity over a random subset of 100 triples (a, b, c)
PIZZA_Q, CLOCK_Q = 0.4, 0.4  # §4.1.2: q in 0–0.4 typical of Pizza, 0.4–1 typical of Clock (reading rule above)
LEVELS = ("E0", "E1", "E2", "E3", "E4", "W-freq", "W-alg", "W-role", "W-scramble")  # every T0 explanation


def _all_inputs() -> torch.Tensor:
    """Every input [a, b, =] for a, b in 0..p−1, row a·p + b."""
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    return torch.stack([a.flatten(), b.flatten(), torch.full((P * P,), P)], 1)


def correct_logit_matrix(model) -> np.ndarray:
    """L[a, b] = logit of the correct answer (a + b) mod p at the '=' position (float64)."""
    toks = _all_inputs()
    with torch.no_grad():
        logits = model(toks)[:, -1, :P].double()  # number classes only
    ans = (toks[:, 0] + toks[:, 1]) % P
    return logits[torch.arange(len(toks)), ans].reshape(P, P).numpy()


def distance_irrelevance(L: np.ndarray) -> float:
    """Zhong et al. Def. 4.2: mean over d of std_a L[a, a + d] (fixed difference), over std of all of L."""
    p, i = L.shape[0], np.arange(L.shape[0])
    within = np.mean([L[i, (i + d) % p].std() for d in range(p)])  # entries with b − a = d, i.e. one fixed a − b
    return float(within / L.std())


def gradient_symmetricity(model, n: int = N_GRAD, seed: int = 0, fixed_attention: bool = False) -> float:
    """Zhong et al. Def. 4.1: mean cosine similarity of ∂Q_abc/∂E_a and ∂Q_abc/∂E_b over n random triples (a, b, c);
    with fixed_attention the attention pattern is treated as a constant (variant, see the module docstring)."""
    abc = np.random.default_rng(seed).integers(0, P, size=(n, 3))
    toks = torch.as_tensor(np.c_[abc[:, :2], np.full(n, P)])
    leaf = {}

    def as_leaf(x):  # replace the token embeddings by a leaf tensor so autograd returns ∂logit/∂embedding
        leaf["x"] = x.detach().clone().requires_grad_(True)
        return leaf["x"]
    hooks = {"hook_embed": as_leaf}
    if fixed_attention:  # stop the gradient through the softmax pattern (values still depend on the embeddings)
        hooks["blocks.0.attn.pattern"] = lambda x: x.detach()
    logits = model(toks, hooks=hooks)[:, -1, :]
    q = logits[torch.arange(n), torch.as_tensor(abc[:, 2])].sum()  # samples are independent, so one backward suffices
    (g,) = torch.autograd.grad(q, leaf["x"])
    return float(F.cosine_similarity(g[:, 0], g[:, 1], dim=1).mean())  # positions 0 (a) and 1 (b)


def diagnose(model) -> dict:
    """Both diagnostics for one model, evaluated in float64."""
    model = model.double().eval()
    q = distance_irrelevance(correct_logit_matrix(model))
    return {"distance_irrelevance": q, "gradient_symmetricity": gradient_symmetricity(model),
            "gradient_symmetricity_fixed_attention": gradient_symmetricity(model, fixed_attention=True),
            "accuracy": float((correct_logit_matrix(model) >= _max_logit(model) - 1e-12).mean()),
            "side": "Clock" if q >= CLOCK_Q else "Pizza"}


def _max_logit(model) -> np.ndarray:
    """Largest number-class logit per input, shaped like the correct-logit matrix (for task accuracy)."""
    with torch.no_grad():
        return model(_all_inputs())[:, -1, :P].double().max(-1).values.reshape(P, P).numpy()


def analyse(settings: Settings) -> Path:
    """Diagnose every target, the T0 reference-compiler build of every explanation (E3 = literal build of the rendered
    true explanation, W-alg = literal Pizza) and every reported T0 agent replacement of the main and controls stages; write
    results/controls/clock_pizza.json."""
    rows = []
    for t in ("T0", "T1", "T2", "T-new", "T-paper"):
        rows.append({"kind": "target", "name": t, **diagnose(targets.load_model(settings, t))})
    for lv in LEVELS:
        npz = settings.runs_dir / "_compiler" / "T0" / f"{lv}.npz"
        if npz.exists():
            rows.append({"kind": "compiler", "name": f"T0 {lv}", **diagnose(weights.to_model(weights.load_npz(npz)))})
    for r in load_runs(2):
        if r["target"] == "T0" and r["model"] == MAIN_MODEL and r.get("variant", "") == "" and r["level"] in LEVELS:
            npz = settings.runs_dir / r["run_id"] / "work_out" / "weights.npz"
            if npz.exists():
                rows.append({"kind": "agent", "name": f"T0 {r['level']} s{r['seed']}", "run_id": r["run_id"],
                             **diagnose(weights.to_model(weights.load_npz(npz)))})
    out = {"definition": __doc__.split("Global context:")[0].strip(), "reading_rule":
           f"Clock if distance irrelevance ≥ {CLOCK_Q}, Pizza if < {PIZZA_Q} (Zhong et al. §4.1.2); W-alg negative "
           "result stands if T0 is Clock and the literal Pizza build (compiler W-alg) is Pizza", "rows": rows}
    path = RES / "controls" / "clock_pizza.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return path
