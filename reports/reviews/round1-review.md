# Round-1 adversarial review (NeurIPS-style self-review)

*2026-11-21, written during round 1 (N = 1 per condition) against the code, results and paper draft at that time.
Reviewer stance: assume the headline claims are wrong and look for what would make them so. Each point ends with the
action taken or planned. Numbers are in the generated tables of
[experiments/01-round1/experiment.md](../experiments/01-round1/experiment.md).*

## Summary of the submission

A weights-blind LLM agent compiles a textual explanation of a one-layer modular-addition transformer into replacement
weights; the replacement is scored by output KL and by cross-validated hidden-state similarity (HS). Claims: output KL
is not a valid explanation score here; HS is valid for frequency-level content, with two blind spots.

## Soundness

1. **Sample size.** Every agent condition has one run. Per-run variance is unknown, so "E2 lifts HS to near the
   compiler" or "E4 < E3" may be noise. The preregistered tests mostly return "cannot tell at this N", and the text
   says so. *Action:* rounds 2–3 (N = 3, 10) as preregistered; no claim in README/paper is stated without its N.

2. **HS has a non-zero, network-dependent floor.** Random initialisations score higher HS against T0 than
   independent trained retrains do. Plausible mechanism: at resid_mid a random network's state is roughly a random
   linear function of the one-hot inputs, which can linearly recover part of any input-linear state of T0, while a
   retrain concentrates its state on *other* frequencies. So "HS above the random floor" is not the right bar for
   "captures the mechanism", and absolute HS values are not comparable across sites or architectures. *Action:*
   compare against the full set of no-agent references (random init, retrains, students) and the compiler, as the
   tables do; state the floor explicitly in the paper's discussion; check the per-site floors in the sensitivity
   analysis (single-site variants).

3. **HS is a frequency-subspace detector.** W-alg (Pizza) and W-scramble score like true explanations, for the agent
   and for the compiler. This is a grader property, not an agent failure. H2's E3 > W-alg and E4 > W-scramble arms
   are therefore expected to fail at N = 10 too. That is an honest negative result, but it limits the headline claim
   to frequency-level explanations. *Action:* stated in abstract, discussion and limitations; matched-neuron
   correlation reported as the candidate second metric (post hoc for this purpose; it was preregistered as secondary).

4. **Output KL's failure may be partly a logit-scale artefact of the self-test.** The self-test checks accuracy, not
   calibration, so agents have no incentive to match the target's logit scale. A calibrated KL (one fitted
   temperature) removes most of E3's KL but not E4's. The claim "KL is not a valid score" should be phrased as "raw
   output KL, as preregistered", with calibrated KL and centred-logit FVU reported. *Action:* paper and report text
   say so; centred-logit FVU marked post hoc.

5. **Planted calibration design.** The planted model is trained further from a fresh optimiser state; the KL spike at
   100 steps reflects early large Adam updates rather than "distance from the planted mechanism". H4b as preregistered
   requires both metrics to be monotone and is therefore "not supported" because of KL, while HS is monotone. *Action:*
   report both halves separately; do not re-define H4b post hoc.

6. **Agent seeds are not seeds for Claude.** Claude 5.5 sampling cannot be seeded, so "seed s" for Haiku labels a
   replicate; runs are not bit-reproducible. *Action:* wording fixed in paper and experiments.md; transcripts, wire
   logs and weights are published for every run so results can be audited, and the compiler gives a deterministic
   reference.

7. **Explanations are rendered by our own script.** The ladder could be unintentionally tuned to the compiler or to
   the grader. Mitigations in place: all development on the dev target; T-new trained after the freeze; explainer
   thresholds varied in the sensitivity analysis. The compiler is non-monotone (E3 < E2) on T2 and T-paper, which
   shows the rendering is not tuned to make the ladder look good. *Action:* keep; report.

8. **Post-freeze changes.** Four deviations were logged (analysis code aligned with amendment 2; stage ordering; an
   infrastructure failure re-run; a NaN secondary metric repaired). The analysis fix was made after per-run scores of
   two stages had been printed. The changes are mechanical (main-model identifier; verdict wording for underpowered
   tests) and cannot move a verdict from "not supported" to "supported". *Action:* disclosed with commit hashes.

9. **Infrastructure-failure rule introduced after a failure.** "A run that never reached the model is re-run" was
   formulated when the first such run happened. Risk: selective re-running. The rule is mechanical (no model response
   at all) and the failed record is archived. *Action:* keep the archive; apply the rule to every future run.

10. **Contamination.** Claude may know Nanda et al. E0/E1 show that priors yield a working adder with near-zero HS on
    T0, so priors do not supply T0's mechanism. T-paper results (seed stage) must be read with the contamination flag.
    *Update after the seed stage:* the priors are specific: with no frequencies given, the replacements carry the
    published checkpoint's frequencies, and the same T0 E0/E1 weights score high when graded against the published
    checkpoint (target swap, post hoc). So T-paper scores are confounded by memory and are excluded from every
    validity claim; the unpublished targets carry the conclusions.

11. **Grader sensitivity not yet reported** at the time of writing. *Action:* `vmi sensitivity 1` (ridge × 10^±2, 3/10
    folds, fold seeds, single sites, orthogonal alignment; explainer thresholds) — tables in the round-1 report.
    *Done:* the ordering of conditions is stable across ridge, folds and fold seeds; both blind spots (W-alg,
    W-scramble) remain under orthogonal alignment, so they are not an artefact of allowing general linear maps.

## Presentation

12. **No graph crimes found** in the static figures after fixes: full HS axes (the planted-calibration HS axis was
    truncated in the first draft and is now 0–1), log scales labelled, no dual axes, identity not by colour alone.
13. **Every number generated.** README, paper, report and site tables come from `vmi analyse`; prose avoids numbers
    except configuration constants. *Residual risk:* prose readings can go stale when results change between rounds.
    *Action:* re-read every "Reading" paragraph after each round.

## Significance and novelty

14. The combination (SGD-trained target, weights-blind full-weight simulation, KL + symmetry-aware HS, explanation
    controls) is not found in the literature search; each part has precedent (iWSP, Gao 2026, Tracr, Cammarata et al.).
    The main value is the validation methodology and the negative results about KL and HS blind spots.

## Verdict (self-assessment at N = 1)

Promising methodology with clear, honestly reported negative results; not yet evidence at the strength the
preregistration targets. Accept-worthy only after N = 10 and a second metric for sub-frequency structure.

## Addendum at the end of round 1 (2026-11-21, after the seed stage and the local runs)

15. **Results-driven stop of the local simulators (deviation 5).** The user stopped the Qwen runs after five runs at
    chance. A stop on results can bias conclusions when it removes conditions that might have gone the other way.
    Here the removed conditions only feed H5 and H6, which are now reported as "not evaluable"; no Haiku condition and
    no other hypothesis depends on them. Residual risk: the 35B model, untested in round 1, might have worked. Its
    seven stage-0 runs on the dev target reached at most 2.4 % task accuracy, which makes that unlikely but not
    excluded. *Action:* disclosed in the preregistration, paper (method, results, limitations) and README.
16. **Memorisation is now shown by a target swap,** not only by frequency sets: the same T0 E0/E1 weights score far
    higher against the published checkpoint than against T0. Two replacements, N = 1; post hoc. *Action:* T-paper
    excluded from every validity claim.
17. **The stitching metric does not fill HS's blind spots.** After the NaN repair it produces values, but a linear
    stitching map depends only on the subspace the replacement spans, so it has HS's invariance (W-alg and
    W-scramble score like true explanations). *Action:* generated stitching table in the paper's discussion; the
    second metric for sub-frequency structure remains open (matched-neuron correlation is the only candidate so far).
18. **Prose readings re-read** after the final regeneration: Qwen text, H5/H6 text and the stitching sentence were
    stale and are fixed.
