"""Static audit of an agent's build_weights.py for forbidden training or optimisation (TASK.md rule).

Local explanation: flags scripts that import autodiff/optimisation libraries or contain loop-and-update patterns
typical of gradient descent (a learning rate, gradients, a loss being reduced inside a loop). A flag is not a verdict:
flagged runs are reviewed by hand, and runs confirmed as training are excluded from the primary analysis
(preregistration §2) and reported.
Global context: the rule exists so the agent constructs weights from the explanation instead of fitting the task;
closed-form linear algebra (np.linalg.lstsq / solve / pinv) is explicitly allowed.
Known gap (audit 7, deviation 14): these patterns do not catch derivative-free searches — a loop that tries a few
values of a constant and keeps the best by a fit or by task accuracy. Reading every reported final script found 7 of
96 such loops (scripts/audit7/build_script_searches.py, which detects them post hoc); the frozen audit is unchanged so
that rounds stay comparable, and extending it is on the README's to-do list for round 3.
"""

from __future__ import annotations  # postponed annotations

import re  # pattern search
from pathlib import Path  # script path

FORBIDDEN_IMPORTS = re.compile(r"^\s*(import|from)\s+(torch|jax|tensorflow|sklearn|scipy\.optimize|autograd|optax)\b", re.M)
SUSPICIOUS = {
    "learning_rate": re.compile(r"\b(lr|learning_rate|step_size)\s*=", re.I),
    "gradient": re.compile(r"\b(grad|gradient|backward|d_loss|dloss)\w*", re.I),
    "update_in_loop": re.compile(r"for\s+\w+\s+in\s+range\(\s*\d{3,}", re.I),
    "optimizer": re.compile(r"\b(adam|sgd|optimi[sz]er|minimi[sz]e)\b", re.I),
}


def audit(script: Path) -> dict:
    """Return {'flagged': bool, 'reasons': [...]} for build_weights.py (missing file → not flagged, run fails anyway)."""
    if not script.exists():
        return {"flagged": False, "reasons": ["no build_weights.py"]}
    src = script.read_text(encoding="utf-8", errors="replace")
    reasons = [f"forbidden import: {m.group(2)}" for m in FORBIDDEN_IMPORTS.finditer(src)]
    hits = [name for name, pat in SUSPICIOUS.items() if pat.search(src)]
    if "gradient" in hits and ("update_in_loop" in hits or "learning_rate" in hits):  # gradient + loop/lr = likely training
        reasons.append("gradient-descent pattern: " + ", ".join(hits))
    elif "optimizer" in hits and "update_in_loop" in hits:
        reasons.append("optimiser pattern: " + ", ".join(hits))
    return {"flagged": bool(reasons), "reasons": reasons, "suspicious_terms": hits}
