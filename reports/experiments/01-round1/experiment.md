# Experiment 01 — Round 1 (N = 1 per condition)

*Status: running. Every number below is generated from `results/` by `vmi analyse 1`. With N = 1 there is no
within-condition spread: verdicts rest on directions, on the trend across conditions and on the deterministic
no-agent arms, and every verdict is labelled "at N=1 — not yet N=3 or N=10". Protocol: preregistration with
amendments 1–2, frozen at tag `protocol-freeze`; post-freeze deviations are logged at the end of
[the preregistration](../../preregistration.md).*

## Setting

- **Simulator:** Claude Haiku 5.5 in Claude Code (offline sandbox, default sampling, effort `medium`, 60-minute cap).
- **Main target:** T0 (`wd1_s1`). Seed-sensitivity targets: T1, T2, T-new (held out), T-paper (published checkpoint).
- **Stages, in order:** main method (E1–E4) → controls (E0, W-freq, W-alg, W-role, W-scramble; random init, wrong
  targets, planted calibration, training trajectory) → baselines (independent retrains, distilled students, reference
  compiler, Fourier-restricted M, weights dump) → ablations (local Qwen simulators; explanation format; no self-test)
  → hyperparameter sensitivity (effort low/high) → seed sensitivity (other targets).

## Main method and agent controls on T0

<!-- BEGIN scoreboard-T0 -->
| Explanation | role | chars | N | agent accuracy | agent KL_out ↓ | agent HS ↑ | agent matched-neuron corr ↑ | compiler KL_out ↓ | compiler HS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| E0 | control: no mechanism | 279 | 1 | 1.000 | 0.009 | 0.004 | 0.020 | 1.64e-04 | 0.051 |
| E1 | method | 1556 | 1 | 1.000 | 1.05e-04 | 0.048 | 0.028 | 1.64e-04 | 0.051 |
| E2 | method | 1788 | 1 | 1.000 | 1.20e-05 | 0.899 | 0.666 | 1.30e-05 | 0.895 |
| E3 | method | 3712 | 1 | 1.000 | 0.023 | 0.939 | 0.828 | 4.25e-04 | 0.927 |
| E4 | method | 51329 | 1 | 0.979 | 0.087 | 0.912 | 0.888 | 0.010 | 0.952 |
| W-freq | control: wrong frequencies | 3724 | 1 | 1.000 | 9.31e-04 | 0.055 | 0.030 | 0.001 | 0.095 |
| W-alg | control: wrong algorithm (Pizza) | 2224 | 1 | 1.000 | 0.015 | 0.889 | 0.666 | 0.040 | 0.921 |
| W-role | control: wrong component roles | 3777 | 1 | 1.000 | 1.09e-04 | 0.168 | 0.015 | 4.727 | 0.500 |
| W-scramble | control: scrambled neurons | 51590 | 1 | 1.000 | 0.003 | 0.918 | 0.621 | 2.02e-05 | 0.859 |
<!-- END scoreboard-T0 -->

![Scores per explanation on T0](../../figures/ladder_T0.png)

**Reading.** The true ladder rises from E1 (no model-specific numbers) to E2–E4 on HS; E0 and E1 build working
adders whose internals share almost nothing with T0, i.e. the simulator's priors give the right behaviour but not
this model's mechanism. Among the controls, wrong frequencies (W-freq) and wrong component roles (W-role) give
correct behaviour with low HS. The Pizza explanation (W-alg) and the scrambled neuron table (W-scramble) are **not**
separated from the true explanations on primary HS: both keep T0's key frequencies, and HS (linear regression over
whole sites) sees frequency subspaces, not the per-neuron assignment or the Clock-vs-Pizza nonlinearity. The secondary
matched-neuron correlation does separate W-scramble from E4. Output KL does not rank the ladder: E2 has the smallest KL
and E4 the largest. For E3 most of the gap is the logit scale (fitting one logit temperature, the secondary
"calibrated KL" in `results/runs/`, lowers it by an order of magnitude); for E4 it is not (its replacement answers some
inputs wrongly). *Post hoc:* the secondary centred-logit FVU, which compares whole logit vectors instead of the
near-one-hot output distributions, puts E3, E4, W-alg and W-scramble below 1 and E0–E2, W-freq and W-role above 1, i.e.
it behaves like HS rather than like KL.

## Behaviour vs mechanism (H3)

![KL vs HS on T0](../../figures/kl_vs_hs_T0.png)

Models that match T0's behaviour almost perfectly — independent retrains, logit-distilled students, the W-freq
replacement — sit at the bottom of the HS axis. Output KL alone cannot tell them from the true replacements; HS can.

## No-agent controls

<!-- BEGIN controls-T0 -->
| Planted model trained further (steps) | HS ↑ | KL_out | snapshot task accuracy |
|---|---|---|---|
| 0 | 1.000 | 0.000 | 1.0000 |
| 100 | 0.998 | 0.371 | 0.9774 |
| 1000 | 0.957 | 0.040 | 0.9958 |
| 10000 | 0.888 | 4.25e-04 | 1.0000 |

| Replacement | HS vs T0 | best wrong target (HS) | d = HS(T0) − max HS(wrong) |
|---|---|---|---|
| E3 | 0.939 | T2 (0.274) | 0.665 |
| E4 | 0.912 | T2 (0.226) | 0.685 |
<!-- END controls-T0 -->

![Planted-mechanism calibration](../../figures/planted_T0.png)

![Training trajectory](../../figures/trajectory_T0.png)

**Reading.** On the planted mechanism (a model whose mechanism is known because it was compiled from E3), HS starts
at 1 and decreases monotonically as the model is trained away from it; output KL is not monotone (training first
disturbs, then re-sharpens the outputs), so H4b's KL half fails while its HS half holds. Replacements built from T0's
explanation fit T0 far better than any other target or no-mechanism model (H7 direction), and their HS against T0's
checkpoints rises through circuit formation together with the fall of the restricted loss (H8).

## Baselines (no agent)

<!-- BEGIN baselines-T0 -->
| Reference point | what it has | n | task accuracy | KL_out ↓ | HS ↑ |
|---|---|---|---|---|---|
| Random initialisation | nothing (floor) | 5 | 0.009 ± 0.001 | 4.766 ± 0.006 | 0.236 ± 0.006 |
| Independent retrains | same recipe, other seeds | 9 | 1.000 ± 0.000 | 6.90e-06 ± 3.96e-06 | 0.119 ± 0.097 |
| Logit-distilled students | M's full output logits, no explanation | 3 | 1.000 ± 0.000 | 2.38e-06 ± 8.25e-07 | 0.233 ± 0.058 |
| Fourier-restricted M | M's weights, key frequencies only in W_E/W_U | 1 | 1.000 | 0.002 | 0.992 |
| Weights dump | all 226,816 weights | 1 | 1.000 | 0.000 | 1.000 |
<!-- END baselines-T0 -->

**Reading.** The training recipe alone (independent retrains) and full behavioural access (logit-distilled students)
both reproduce T0's outputs, yet their HS stays at or below the random-initialisation floor. Explanations E2–E4 lift
HS far above both; keeping only T0's key-frequency Fourier components in the embedding and unembedding barely
changes HS, so the key frequencies carry nearly all of what HS measures.

## Reference compiler on every target

T0:

<!-- BEGIN compiler-T0 -->
| Explanation | chars | task accuracy | KL_out ↓ | HS ↑ | resid_mid | mlp_post | resid_post | matched neuron corr |
|---|---|---|---|---|---|---|---|---|
| E0 | 279 | 1.0000 | 1.64e-04 | 0.051 | 0.000 | 0.154 | 0.000 | 0.015 |
| E1 | 1556 | 1.0000 | 1.64e-04 | 0.051 | 0.000 | 0.154 | 0.000 | 0.015 |
| E2 | 1788 | 1.0000 | 1.30e-05 | 0.895 | 0.884 | 0.945 | 0.856 | 0.675 |
| E3 | 3712 | 1.0000 | 4.25e-04 | 0.927 | 0.913 | 0.973 | 0.895 | 0.715 |
| E4 | 51329 | 0.9998 | 0.010 | 0.952 | 0.913 | 0.992 | 0.951 | 0.962 |
| W-freq | 3724 | 1.0000 | 0.001 | 0.095 | 0.074 | 0.200 | 0.009 | 0.024 |
| W-alg | 2224 | 1.0000 | 0.040 | 0.921 | 0.884 | 0.973 | 0.906 | 0.715 |
| W-role | 3777 | 0.0065 | 4.727 | 0.500 | 0.913 | 0.000 | 0.587 | 0.000 |
| W-scramble | 51590 | 1.0000 | 2.02e-05 | 0.859 | 0.913 | 0.972 | 0.691 | 0.670 |
<!-- END compiler-T0 -->

T1:

<!-- BEGIN compiler-T1 -->
| Explanation | chars | task accuracy | KL_out ↓ | HS ↑ | resid_mid | mlp_post | resid_post | matched neuron corr |
|---|---|---|---|---|---|---|---|---|
| E0 | 279 | 1.0000 | 1.17e-04 | 0.059 | 0.009 | 0.169 | 0.000 | 0.054 |
| E1 | 1556 | 1.0000 | 1.17e-04 | 0.059 | 0.009 | 0.169 | 0.000 | 0.054 |
| E2 | 1782 | 1.0000 | 6.18e-06 | 0.950 | 0.939 | 0.976 | 0.935 | 0.834 |
| E3 | 3536 | 1.0000 | 1.25e-06 | 0.984 | 0.987 | 0.976 | 0.989 | 0.749 |
| E4 | 49805 | 1.0000 | 2.19e-07 | 0.992 | 0.987 | 0.997 | 0.993 | 0.958 |
| W-freq | 3536 | 1.0000 | 0.025 | 0.069 | 0.005 | 0.203 | 0.000 | 0.059 |
| W-alg | 2152 | 1.0000 | 0.061 | 0.936 | 0.939 | 0.976 | 0.893 | 0.749 |
| W-role | 3585 | 0.0131 | 4.727 | 0.571 | 0.987 | 0.000 | 0.725 | 0.000 |
| W-scramble | 50456 | 1.0000 | 2.51e-07 | 0.992 | 0.987 | 0.996 | 0.993 | 0.782 |
<!-- END compiler-T1 -->

T2:

<!-- BEGIN compiler-T2 -->
| Explanation | chars | task accuracy | KL_out ↓ | HS ↑ | resid_mid | mlp_post | resid_post | matched neuron corr |
|---|---|---|---|---|---|---|---|---|
| E0 | 279 | 1.0000 | 5.33e-04 | 0.304 | 0.290 | 0.396 | 0.228 | 0.173 |
| E1 | 1556 | 1.0000 | 5.33e-04 | 0.304 | 0.290 | 0.396 | 0.228 | 0.173 |
| E2 | 1794 | 1.0000 | 1.62e-05 | 0.833 | 0.959 | 0.978 | 0.563 | 0.869 |
| E3 | 3826 | 1.0000 | 4.57e-06 | 0.783 | 0.986 | 0.977 | 0.387 | 0.830 |
| E4 | 51812 | 1.0000 | 6.98e-05 | 0.910 | 0.986 | 0.988 | 0.757 | 0.965 |
| W-freq | 3808 | 1.0000 | 1.52e-04 | 0.158 | 0.173 | 0.272 | 0.028 | 0.054 |
| W-alg | 2270 | 1.0000 | 0.009 | 0.948 | 0.959 | 0.977 | 0.909 | 0.830 |
| W-role | 3887 | 0.0063 | 4.727 | 0.541 | 0.986 | 0.000 | 0.638 | 0.000 |
| W-scramble | 52289 | 1.0000 | 2.83e-05 | 0.963 | 0.986 | 0.987 | 0.916 | 0.820 |
<!-- END compiler-T2 -->

T-new:

<!-- BEGIN compiler-T-new -->
| Explanation | chars | task accuracy | KL_out ↓ | HS ↑ | resid_mid | mlp_post | resid_post | matched neuron corr |
|---|---|---|---|---|---|---|---|---|
| E0 | 279 | 1.0000 | 1.60e-04 | 0.032 | 0.000 | 0.096 | 0.000 | 0.010 |
| E1 | 1556 | 1.0000 | 1.60e-04 | 0.032 | 0.000 | 0.096 | 0.000 | 0.010 |
| E2 | 1786 | 1.0000 | 7.69e-06 | 0.955 | 0.953 | 0.980 | 0.930 | 0.774 |
| E3 | 3635 | 1.0000 | 8.34e-06 | 0.970 | 0.987 | 0.980 | 0.941 | 0.751 |
| E4 | 51778 | 1.0000 | 2.54e-05 | 0.983 | 0.987 | 0.992 | 0.968 | 0.963 |
| W-freq | 3635 | 1.0000 | 1.87e-04 | 0.089 | 0.107 | 0.158 | 0.002 | 0.061 |
| W-alg | 2191 | 1.0000 | 0.040 | 0.946 | 0.953 | 0.980 | 0.904 | 0.751 |
| W-role | 3683 | 0.0242 | 4.728 | 0.538 | 0.987 | 0.000 | 0.628 | 0.000 |
| W-scramble | 52030 | 1.0000 | 5.54e-05 | 0.976 | 0.987 | 0.993 | 0.946 | 0.737 |
<!-- END compiler-T-new -->

T-paper:

<!-- BEGIN compiler-T-paper -->
| Explanation | chars | task accuracy | KL_out ↓ | HS ↑ | resid_mid | mlp_post | resid_post | matched neuron corr |
|---|---|---|---|---|---|---|---|---|
| E0 | 279 | 1.0000 | 2.01e-04 | 0.045 | 0.003 | 0.133 | 0.000 | 0.012 |
| E1 | 1556 | 1.0000 | 2.01e-04 | 0.045 | 0.003 | 0.133 | 0.000 | 0.012 |
| E2 | 1790 | 1.0000 | 8.00e-06 | 0.944 | 0.949 | 0.971 | 0.911 | 0.751 |
| E3 | 3731 | 1.0000 | 6.19e-06 | 0.867 | 0.990 | 0.973 | 0.637 | 0.774 |
| E4 | 51728 | 1.0000 | 7.58e-05 | 0.983 | 0.990 | 0.986 | 0.973 | 0.966 |
| W-freq | 3725 | 1.0000 | 6.67e-05 | 0.121 | 0.129 | 0.212 | 0.022 | 0.040 |
| W-alg | 2229 | 1.0000 | 0.033 | 0.926 | 0.949 | 0.974 | 0.855 | 0.774 |
| W-role | 3783 | 0.0085 | 4.727 | 0.550 | 0.990 | 0.000 | 0.658 | 0.000 |
| W-scramble | 52150 | 1.0000 | 3.11e-05 | 0.986 | 0.990 | 0.989 | 0.980 | 0.747 |
| E-paper | 3102 | 1.0000 | 1.60e-06 | 0.969 | 0.990 | 0.974 | 0.943 | 0.727 |
<!-- END compiler-T-paper -->

**Reading.** The compiler is not monotone in explanation detail on every target: on T2 and T-paper its E3 build
scores below E2, so the extra E3 quantities as rendered (amplitudes, attention fits) can mislead a literal
construction. This is a property of the explanations and the compiler, measured without any agent.

## All round-1 agent runs (incl. ablations, hyperparameter and seed sensitivity)

<!-- BEGIN runs -->
| Stage | Target | Explanation | Simulator | T | Variant | N | failed | accuracy | KL_out ↓ (mean ± sd) | HS ↑ (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ablations | T0 | E2 | qwen3.5-4b | 0.6 | – | 1 | 0 | 0.009 | 4.73 | 0.000 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-code | 1 | 0 | 1.000 | 0.00212 | 0.918 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | fmt-math | 1 | 0 | 1.000 | 0.000716 | 0.911 |
| ablations | T0 | E3 | claude-haiku-5-5 | default | noselftest | 1 | 0 | 1.000 | 0.00134 | 0.964 |
| controls | T0 | E0 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.00908 | 0.004 |
| controls | T0 | W-freq | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000931 | 0.055 |
| controls | T0 | W-alg | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.0146 | 0.889 |
| controls | T0 | W-role | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000109 | 0.168 |
| controls | T0 | W-scramble | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.00283 | 0.918 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-high | 1 | 0 | 1.000 | 0.00152 | 0.923 |
| hyper | T0 | E3 | claude-haiku-5-5 | default | effort-low | 1 | 0 | 0.562 | 1.97 | 0.922 |
| hyper | T0 | W-freq | claude-haiku-5-5 | default | effort-low | 1 | 0 | 0.770 | 1.03 | 0.114 |
| main | T0 | E1 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.000105 | 0.048 |
| main | T0 | E2 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 1.2e-05 | 0.899 |
| main | T0 | E3 | claude-haiku-5-5 | default | – | 1 | 0 | 1.000 | 0.0229 | 0.939 |
| main | T0 | E4 | claude-haiku-5-5 | default | – | 1 | 0 | 0.979 | 0.0872 | 0.912 |
<!-- END runs -->

## Hypotheses at N = 1

<!-- BEGIN hypotheses -->
| Hypothesis | Verdict | Evidence size |
|---|---|---|
| H1 | not supported | at N=1 — not yet N=3 or N=10 |
| H2 | E3>W-freq: cannot tell at this N; E3>W-alg: cannot tell at this N; E3>W-role: cannot tell at this N; E4>W-scramble: not supported | at N=1 — not yet N=3 or N=10 |
| H3 | supported | at N=1 — not yet N=3 or N=10 |
| H4 | E3: cannot tell at this N; E4: cannot tell at this N | at N=1 — not yet N=3 or N=10 |
| H4b | not supported | deterministic |
| H5 | not evaluated yet | – |
| H6 | not evaluated yet | – |
| H7 | cannot tell at this N | 2 E3/E4 replacements (at N=1 — not yet N=3 or N=10) |
| H8 | supported | one replacement, all checkpoints |
<!-- END hypotheses -->

Full statistics: `results/hypotheses_round1.json`.

## Deviations from the plan in this round

See the post-freeze deviations at the end of the preregistration: the analysis code was brought in line with
amendment 2 before any analysis; the local Qwen ablation runs proceed in parallel with later Haiku stages; one
infrastructure failure (a unix socket path over the length limit) was archived and re-run.
