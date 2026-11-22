# Round-2 adversarial review (N = 3)

*2026-11-22, against the code, results and paper after round 2 (three runs per condition). Reviewer stance as in
[round 1](round1-review.md): assume the headline claims are wrong and look for what would make them so. Numbers are in
the generated tables of [02-round2/experiment.md](../experiments/02-round2/experiment.md).*

## Soundness

1. **Analysis corrections after seeing N = 3 results (deviation 7).** Two code paths did not implement §5–§6 as
   written: H1 reported only its HS part, and H2 skipped the power rule at N ≥ 2. Risk: post-hoc analysis choices.
   Both corrections follow from the preregistered text, neither creates a "supported" verdict, and the change is
   disclosed with its direction. *Action:* keep; `test_verdict_rules_follow_preregistration_sections_5_and_6` now
   checks both rules against hand-computed cases.
2. **H1's HS support is driven by the E1 → E2 step.** A Jonckheere trend test rewards any monotone ordering; here
   E1 sits far below the rest, every E3 run is above every E2 run, and E3 vs E4 is mixed. "HS rises with every added
   level of detail" would overclaim. *Action:* text says "the big step is E1 → E2; E3 → E4 is mixed".
3. **H2 is undecidable at N = 3, but its intervals already show the blind spots.** W-alg and W-scramble tie with the
   true explanations (bootstrap CIs include 0). More runs would likely make E3 > W-freq and E3 > W-role significant (their
   CIs are far from 0); they cannot rescue the two blind-spot arms unless the means move. *Action:* reported as a
   negative result.
4. **Replicates changed two round-1 readings.** The low-effort accuracy collapse and the E0 memorisation each held in
   one run of three. This is what replication is for; the text was corrected. *Residual risk:* other single-run
   statements in the round-1 report remain N = 1 statements and are labelled so.
5. **Memorisation is post hoc and target-specific.** It rests on T0's E0/E1 runs plus a target swap; E1 3/3 is strong,
   E0 1/3 is weak. On T-paper the algorithm-only runs vary widely. *Action:* claims limited to "algorithm-only
   explanations, main target"; T-paper excluded from validity claims.
6. **Specificity (H7) uses wrong targets from one training family.** All wrong targets are modular-addition networks
   (other seeds, no-weight-decay and random networks). That is the relevant hard case for this task, but says nothing
   about other architectures. *Action:* stated as scope.
7. **Example selection.** The README's examples (highlights, outputs, agent transcripts) could be cherry-picked. They
   are chosen by fixed rules in code (first replicate of a condition, or an extreme of a stored metric) and
   regenerated from results; the rules are printed next to the examples. *Residual risk:* the rule set itself was
   written after the round-1 results were known.
8. **Secondary stitching metric** gives identical values for replacements that span the same subspace; it adds no
   discrimination below the frequency level (post hoc, reported in the paper's discussion).

## Presentation

9. **Every number generated.** New tables (predictions, replications, highlights, outputs, ablations, agent
   examples) and figures are produced by `vmi analyse`; prose readings were re-read against the N = 3 numbers and four
   stale N = 1 statements were corrected (README, paper, summary).
10. **No graph crimes found** in the two new figures: one colour scale per row with a colour bar, panels labelled with
    HS and the neuron correlation, ground truth shown in the first column.

## Verdict (self-assessment at N = 3)

The frequency-level validity claims now pass preregistered tests; the negative results (output KL, the two blind
spots, the compiler not being a ceiling, memorisation on published models) are stable across replicates. Not yet at
the preregistered N = 10, and the capability axis (H5, H6) is untested.
