"""Independent re-evaluation of every preregistered hypothesis (preregistration §5) from raw results files.

Written from the preregistration text, without importing the project's analysis code, to check
results/hypotheses_round2.json. Exact permutation JT (all multinomial assignments), scipy exact Mann-Whitney / Wilcoxon.
Audit 3 (2026-11-22). Usage: uv run python scripts/audit3/independent_hypotheses.py .   (~1.5 min)
"""
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(sys.argv[1]) / "results"
runs = [json.loads(Path(p).read_text()) for p in glob.glob(str(R / "runs" / "*.json"))]
runs = [r for r in runs if r["stage"] != "dev" and 1 <= int(r["round"]) <= 2]


def sel(target="T0", level=None, model="claude-haiku-5-5", variant=""):
    return [r for r in runs if r["target"] == target and r["model"] == model and r.get("variant", "") == variant
            and (level is None or r["level"] == level) and (r.get("temperature") is None)]


def vals(rs, k="hs"):
    return np.array([r[k] for r in rs], float)


def jt_exact(groups):
    """One-sided JT (increasing), exact over every assignment of the pooled values to groups of the given sizes."""
    x = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    C = (x[:, None] < x[None, :]).astype(float) + 0.5 * (x[:, None] == x[None, :])
    n = len(x)
    labels = []
    def rec(items, k, lab):
        if k == len(sizes):
            labels.append(lab.copy()); return
        for comb in itertools.combinations(items, sizes[k]):
            for i in comb: lab[i] = k
            rec([i for i in items if i not in comb], k + 1, lab)
    rec(list(range(n)), 0, np.zeros(n, int))
    L = np.array(labels)
    obs_lab = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    def J(lab):
        return (C * (lab[..., :, None] < lab[..., None, :])).sum(axis=(-1, -2))
    Js = np.concatenate([J(L[i:i + 20000]) for i in range(0, len(L), 20000)])
    obs = J(obs_lab)
    return float(obs), float((Js >= obs - 1e-12).mean()), len(L)


def spearman_boot(groups, n=10000, seed=1):
    lv = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    rho = stats.spearmanr(lv, np.concatenate(groups)).statistic
    rng = np.random.default_rng(seed)
    b = [stats.spearmanr(lv, np.concatenate([rng.choice(g, len(g)) for g in groups])).statistic for _ in range(n)]
    b = np.array([v for v in b if np.isfinite(v)])
    return float(rho), (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


out = {}
lv4 = ["E1", "E2", "E3", "E4"]
g_hs = [vals(sel(level=l)) for l in lv4]
g_kl = [-vals(sel(level=l), "kl_out") for l in lv4]
print("N per level", [len(g) for g in g_hs])
for name, g in (("HS", g_hs), ("-KL", g_kl)):
    J, p, nass = jt_exact(g)
    rho, ci = spearman_boot(g)
    ok = p < 0.05 and rho > 0 and ci[0] > 0
    print(f"H1 {name}: J {J} exact p {p:.3g} ({nass} assignments)  rho {rho:.3f} CI {ci[0]:.3f}..{ci[1]:.3f} -> {'supported' if ok else 'not supported'}")
e0 = vals(sel(level="E0"))
print(f"H1 E0 clause: mean E0 {e0.mean():.3f} vs E1 {g_hs[0].mean():.3f} -> E0 below E1? {e0.mean() < g_hs[0].mean()}; "
      f"MW E1>E0 one-sided p {stats.mannwhitneyu(g_hs[0], e0, alternative='greater', method='exact').pvalue:.3f}")

# H2
ps = {}
for a, b in (("E3", "W-freq"), ("E3", "W-alg"), ("E3", "W-role"), ("E4", "W-scramble")):
    x, y = vals(sel(level=a)), vals(sel(level=b))
    ps[f"{a}>{b}"] = (stats.mannwhitneyu(x, y, alternative="greater", method="exact").pvalue, x.mean() - y.mean(), len(x), len(y))
m = len(ps); run = 0
for i, (k, (p, d, nx, ny)) in enumerate(sorted(ps.items(), key=lambda kv: kv[1][0])):
    run = max(run, min(1, (m - i) * p))
    minp = 1 / __import__("math").comb(nx + ny, nx)
    v = "cannot tell" if minp >= 0.05 else ("supported" if run < 0.05 and d > 0 else "not supported")
    print(f"H2 {k}: p {p:.3f} Holm {run:.3f} diff {d:+.3f} min p {minp:.3f} -> {v}")

# H3
wf = sel(level="W-freq")
tm = {l: (vals(sel(level=l), "kl_out").mean(), vals(sel(level=l)).mean()) for l in ("E2", "E3", "E4")}
agree, kl, hs = vals(wf, "argmax_agreement").mean(), vals(wf, "kl_out").mean(), vals(wf).mean()
ok3 = agree >= 0.99 and kl <= max(v[0] for v in tm.values()) and all(hs < v[1] for v in tm.values())
print(f"H3: agree {agree:.4f} KL {kl:.4g} (true KL {[f'{v[0]:.3g}' for v in tm.values()]}) HS {hs:.3f} -> {'supported' if ok3 else 'not supported'};"
      f" KL within [min,max]: {min(v[0] for v in tm.values()) <= kl <= max(v[0] for v in tm.values())}")

# H4
base = json.loads((R / "baselines" / "T0.json").read_text())
retr = np.array([x["hs"] for x in base["retrains"]]); stud = np.array([x["hs"] for x in base["students"]])
p4 = {l: stats.mannwhitneyu(vals(sel(level=l)), retr, alternative="greater", method="exact").pvalue for l in ("E3", "E4")}
srt = sorted(p4.items(), key=lambda kv: kv[1]); run = 0; adj = {}
for i, (k, p) in enumerate(srt):
    run = max(run, min(1, (2 - i) * p)); adj[k] = run
print(f"H4: n retrains {len(retr)} (max {retr.max():.3f}), p {p4}, Holm {adj} -> {['supported' if adj[l] < 0.05 else 'not supported' for l in adj]}")
for l in ("E3", "E4"):
    print(f"   vs students two-sided p {stats.mannwhitneyu(vals(sel(level=l)), stud, alternative='two-sided', method='exact').pvalue:.3f} (min p 0.1)")

# H4b
rows = json.loads((R / "controls" / "planted_calibration_T0.json").read_text())["rows"]
hsr, klr = [r["hs"] for r in rows], [r["kl_out"] for r in rows]
ok = abs(hsr[0] - 1) < 1e-6 and klr[0] < 1e-9 and all(a > b for a, b in zip(hsr, hsr[1:])) and all(a < b for a, b in zip(klr, klr[1:]))
print(f"H4b: HS {hsr} KL {klr} -> {'supported' if ok else 'not supported'}")

# H5 (T0 half, descriptive)
for l in ("E2", "E3", "E4", "W-freq"):
    v = vals(sel(level=l)); c = json.loads((R / "compiler" / "T0" / f"{l}.json").read_text())["hs"]
    rng = np.random.default_rng(1); bm = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    lo = np.percentile(bm, 2.5)
    print(f"H5 T0 {l}: agent {v.mean():.3f} compiler {c:.3f} gap {v.mean() - c:+.3f} beyond CI {v.mean() > c and lo > c}")

# H7
wr = json.loads((R / "controls" / "wrong_targets_T0.json").read_text())["rows"]
d = np.array([r["hs_T0"] - max(r["hs_wrong"].values()) for r in wr])
print(f"H7: n {len(d)} wrong targets {len(wr[0]['hs_wrong'])} d {np.round(d, 3)} recomputed-d matches stored "
      f"{np.allclose(d, [r['d'] for r in wr])}; Wilcoxon p {stats.wilcoxon(d, alternative='greater').pvalue:.4f} (min 1/64)")
# check that hs_T0 equals the run record's hs
hs_by_id = {r["run_id"]: r["hs"] for r in runs}
print("   hs_T0 equals run records:", all(abs(r["hs_T0"] - hs_by_id[r["run_id"]]) < 1e-9 for r in wr),
      "| replacements are the main E3/E4 runs:", sorted(r["run_id"][:9] for r in wr))

# H8
tr = json.loads((R / "controls" / "trajectory_T0.json").read_text())
rhos = [x["rho"] for x in tr["replacements"]]
print(f"H8: {len(rhos)} replacements, steps {len(tr['steps'])}, rho {np.round(rhos, 3)} -> {'supported' if min(rhos) > 0 else 'not supported'}")
# recompute one rho from stored per-replacement series if present
x0 = tr["replacements"][0]
if "hs" in x0:
    print("   rho recomputed:", stats.spearmanr(x0["hs"], -np.log(tr["restricted_loss"])).statistic, "stored", x0["rho"])

# Replications with all available true levels of E1-E4 (preregistered H1 test over the ladder levels that were run)
for t in ("T1", "T2", "T-new", "T-paper"):
    levels = [l for l in lv4 if sel(target=t, level=l)]
    for lvset in ([l for l in levels if l != "E1"], levels):
        if len(lvset) < 2: continue
        g = [vals(sel(target=t, level=l)) for l in lvset]
        J, p, nass = jt_exact(g); rho, ci = spearman_boot(g)
        v = "cannot tell" if 1 / nass >= 0.05 else ("supported" if p < 0.05 and rho > 0 and ci[0] > 0 else "not supported")
        print(f"Rep H1 {t} {'<'.join(lvset)}: exact p {p:.3f} rho {rho:.2f} CI {ci[0]:.2f}..{ci[1]:.2f} -> {v}")
