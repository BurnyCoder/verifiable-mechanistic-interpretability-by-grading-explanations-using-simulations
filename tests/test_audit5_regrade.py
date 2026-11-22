"""The audit-5 re-grader's independent forward pass agrees with the project's grader on a random network.

Local explanation: `scripts/audit5/regrade_independent.py` computes logits and hidden states from
`sandbox_template/ARCHITECTURE.md` alone. This test runs it and the project's `vmi.grader.activations.collect` on the
same randomly initialised GrokTransformer and requires the logits and the three hidden-state sites to agree in float64,
and the independent HS of a network against itself to be 1 (up to the ridge term). It guards the audit script against
silent drift (input order, head concatenation, causal mask), which would otherwise only show as unexplained
differences in `results/audit5/regrade_independent.json`.
Global context: audit 5 (deviation 12).
Sources: forward-pass specification `sandbox_template/ARCHITECTURE.md`; importlib loading of a script by path
(https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly).
"""

import importlib.util  # load the audit script by path (scripts/ is not a package)

import numpy as np  # comparisons

from vmi.config import REPO_ROOT  # repository root
from vmi.grader.activations import collect  # the project's forward pass


def _audit_module():
    spec = importlib.util.spec_from_file_location("regrade_independent", REPO_ROOT / "scripts" / "audit5" / "regrade_independent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_independent_forward_matches_project_forward_on_a_random_network():
    from grokviz.model import GrokTransformer  # paper-faithful architecture
    m = _audit_module()
    model = GrokTransformer(seed=7).double()
    mine = m.forward(m.from_state_dict(model.state_dict()))
    ref = collect(model, "cpu")
    for site in ("logits", "resid_mid", "mlp_post", "resid_post"):
        np.testing.assert_allclose(mine[site], getattr(ref, site).cpu().numpy(), rtol=1e-9, atol=1e-9)
    h, sites = m.hs(mine, mine)
    assert h > 0.999 and all(v > 0.999 for v in sites.values())
