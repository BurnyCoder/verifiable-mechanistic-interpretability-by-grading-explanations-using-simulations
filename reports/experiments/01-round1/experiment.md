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
<!-- END baselines-T0 -->

**Reading.** The training recipe alone (independent retrains) and full behavioural access (logit-distilled students)
both reproduce T0's outputs, yet their HS stays at or below the random-initialisation floor. Explanations E2–E4 lift
HS far above both; keeping only T0's key-frequency Fourier components in the embedding and unembedding barely
changes HS, so the key frequencies carry nearly all of what HS measures.

## Reference compiler on every target

T0:

<!-- BEGIN compiler-T0 -->
<!-- END compiler-T0 -->

T1:

<!-- BEGIN compiler-T1 -->
<!-- END compiler-T1 -->

T2:

<!-- BEGIN compiler-T2 -->
<!-- END compiler-T2 -->

T-new:

<!-- BEGIN compiler-T-new -->
<!-- END compiler-T-new -->

T-paper:

<!-- BEGIN compiler-T-paper -->
<!-- END compiler-T-paper -->

**Reading.** The compiler is not monotone in explanation detail on every target: on T2 and T-paper its E3 build
scores below E2, so the extra E3 quantities as rendered (amplitudes, attention fits) can mislead a literal
construction. This is a property of the explanations and the compiler, measured without any agent.

## All round-1 agent runs (incl. ablations, hyperparameter and seed sensitivity)

<!-- BEGIN runs -->
<!-- END runs -->

## Hypotheses at N = 1

<!-- BEGIN hypotheses -->
<!-- END hypotheses -->

Full statistics: `results/hypotheses_round1.json`.

## Deviations from the plan in this round

See the post-freeze deviations at the end of the preregistration: the analysis code was brought in line with
amendment 2 before any analysis; the local Qwen ablation runs proceed in parallel with later Haiku stages; one
infrastructure failure (a unix socket path over the length limit) was archived and re-run.
