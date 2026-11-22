"""The preregistered target rules (§2), re-applied to the published key frequencies, give the registered targets.

Local explanation: `results/targets_selection.json` (written by `scripts/audit5/checks.py` from each run's published
`analysis/key_freqs.json`) lists the key frequencies of all eleven `wd1` seeds. This test re-applies §2's rules without
the script's code — T0: first seed whose set differs from the paper's {14, 35, 41, 42, 52}; T1 / T2: fewest / most key
frequencies among the remaining non-dev seeds, ties to the lower seed — and checks them against `targets.REGISTRY`.
Global context: guards the paper's §3.1 sentence on target selection, which is now generated from this file (audit 5).
Sources: preregistration §2; registry in `src/vmi/targets.py`.
"""

import json  # results file

import pytest  # skip when the file has not been generated

from vmi.config import REPO_ROOT  # repository root
from vmi.targets import REGISTRY  # preregistered roles

SEL = REPO_ROOT / "results" / "targets_selection.json"


@pytest.mark.skipif(not SEL.exists(), reason="run scripts/audit5/checks.py first")
def test_published_key_frequencies_reproduce_the_registered_targets():
    s = json.loads(SEL.read_text())
    kf = s["key_freqs"]
    seeds = [f"wd1_s{i}" for i in range(10)]  # wd1_s10 is the dev target, fixed by §2
    t0 = next(r for r in seeds if sorted(kf[r]) != [14, 35, 41, 42, 52])
    rest = [r for r in seeds if r != t0]
    t1 = sorted(rest, key=lambda r: (len(kf[r]), int(r.split("_s")[1])))[0]
    t2 = sorted(rest, key=lambda r: (-len(kf[r]), int(r.split("_s")[1])))[0]
    assert (t0, t1, t2) == (REGISTRY["T0"], REGISTRY["T1"], REGISTRY["T2"])
    assert (s["rule_T0"], s["rule_T1"], s["rule_T2"]) == (t0, t1, t2)
    assert REGISTRY["dev"] == s["dev"] == "wd1_s10"
