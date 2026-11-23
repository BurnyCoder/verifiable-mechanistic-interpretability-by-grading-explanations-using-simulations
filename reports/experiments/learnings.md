# Learnings

Running log of what was learned from the literature, from errors and from development, and why each fix was chosen.
Newest entries at the bottom. Per-experiment learnings live in each experiment's folder.

## From the literature and SOTA review (2026-11-21)

- **Same accuracy ≠ same mechanism is documented in exactly this task family.** Clock vs Pizza networks
  ([Zhong et al. 2023](https://arxiv.org/abs/2306.17844); same accuracy, different logit patterns) and Tracr's compressed models (layer cosine ≈ 0.8 at almost perfect
  accuracy; [Lindner et al. 2023](https://arxiv.org/abs/2301.05062)) motivated a hidden-state metric next to output KL.
- **The alignment class must match the model's symmetries.** Without LayerNorm an invertible linear change of the
  residual basis (compensated in the reading and writing weights) leaves the function unchanged, so CKA/Procrustes
  (orthogonal invariance) would penalise functionally identical replacements; under Sutter et al.'s assumptions, unrestricted maps make any
  algorithm match any network ([Sutter et al. 2025](https://arxiv.org/abs/2507.08802)). Hence cross-validated *linear* regression in
  both directions as the primary hidden-state score. Its invariance is partial: R² is invariant to invertible linear
  maps of the predictor only ([Kornblith et al. 2019, Table 1](https://arxiv.org/abs/1905.00414)), so completeness
  R²(M̂→M) is invariant to the replacement's basis, soundness R²(M→M̂) only to orthogonal maps and uniform scaling; under
  any exact invertible linear equivalence both are 1 up to the ridge term, so HS = 1 up to it. Tests: `test_invariant_to_residual_basis_change` (HS = 1,
  CKA < 1 under a random invertible change of basis) and
  `test_soundness_depends_on_the_replacement_basis_and_whitened_soundness_does_not`.
- **Agents can recall published circuits** ([Haklay et al. 2026](https://arxiv.org/abs/2603.20101)); the paper's
  frequencies {14, 35, 41, 42, 52} are public. Hence targets analysed in no paper (weights and key frequencies uploaded to
  Hugging Face on 2026-10-21, when it became public is not recorded; after Claude Haiku 5.5's June 2026 training-data cutoff; no internet in the
  sandbox) and a held-out target trained after the freeze.
- **Simulators can ignore explanations** ([Poché et al. 2026](https://arxiv.org/abs/2609.08585)) → E0 (no explanation)
  and E1 (no model-specific numbers) controls.

## Engineering (2026-11-21)

- **Ollama vs llama-server.** Ollama's OpenAI endpoint cannot set the context size ([docs](https://docs.ollama.com/api/openai-compatibility)) and defaults to 4k context ([docs](https://docs.ollama.com/context-length)) on
  < 24 GiB GPUs, which breaks agent tool calls; it also exposes `/api/pull`. llama-server with a fixed GGUF and
  `--n-cpu-moe` was chosen instead.
- **numpy was invisible inside the sandbox** because it lives in `~/.local` and `/home` is hidden; there is no sudo
  and no `python3-venv`. Fix: `scripts/setup_wsl.sh` copies numpy into `~/vmi-sim/pylib`, bind-mounted read-only.
- **No libseccomp** for the AF_VSOCK filter → the 9-instruction classic-BPF program is assembled by hand
  (`scripts/make_seccomp.py`); the isolation check confirms `socket(AF_VSOCK)` fails inside the sandbox.
- **Git Bash mangles `/mnt/c/...` arguments** passed to `wsl.exe`; set `MSYS_NO_PATHCONV=1` (the Python runner passes
  arguments directly and is unaffected).
- **The first throughput measurement was a page-in artefact.** With all MoE expert weights memory-mapped from the
  D: drive, the first request after loading Qwen3.6-35B-A3B ran at 3.5–3.9 prompt tokens/s (the OS was paging the
  17.7 GB file in, with ~1–8 GB of free RAM next to the user's other applications). The same configuration warm
  (second request, 32k context) ran at 150 prompt tokens/s and 21.6 generated tokens/s, above the preregistered
  8 tokens/s threshold, so the main simulator was kept. Lesson: measure warm, and keep the server running for a whole
  batch of agent runs. At 64k context the non-expert weights plus KV cache used 5,141 MiB (5.0 GiB) of VRAM; at 32k, 2,872 MiB (2.8 GiB).
- **Port 8080 could not be bound** on this machine although nothing was listed as listening (likely a WSL
  mirrored-networking reservation); llama-server moved to port 8091.

## Agent behaviour (dev target only)

- **Unlimited thinking stalls the agent.** In the first smoke run (dev, E3) the model read all five files, then spent
  one turn thinking for 16,384 tokens (~13 minutes at 21 tokens/s; 47k characters) and hit OpenCode's per-turn output
  cap without making a tool call. The reasoning itself was on track — it was designing the same kind of construction
  as the reference compiler (Fourier dimensions in the embedding, a constant query at "=", keys modulated by
  cos/sin of the head's frequency). Fix: llama-server `--reasoning-budget 4096` with a short message inserted before
  the end-of-thinking tag, so each turn plans at most ~3 minutes and then acts. Decided on the dev target before the
  protocol freeze.
- **Repetition loops inside tool calls.** In the second smoke run the model wrote `build_weights.py` through a single
  `write` call whose content degenerated into the same comment paragraph repeated until the 16,384-token output cap
  (so the tool call arrived truncated with empty input). Fix: `presence_penalty` 1.5 in every request (the default
  Ollama ships for Qwen 3.5, checked with `ollama show qwen3.5:9b`; Qwen recommends 0–2 against endless repetition) and
  an 8,192-token per-turn output cap so a loop costs ~6 minutes instead of ~13.
- **Reasoning moved into code comments, and the whole script into one tool call.** In the third smoke run there was
  no repetition loop: after its thinking was cut at the 4,096-token budget, the model kept designing *inside* the
  `write` call (a 504-line `build_weights.py`, 386 distinct lines, ending in paragraphs of comments weighing which head
  should carry k = 47). Three such writes hit the 8,192-token cap; OpenCode drops truncated responses from the
  history, so each retry started from the same context, and after the third OpenCode ended the session
  ("Response still truncated after two output token limit continuations"; 6 steps, 30 minutes, no weights). Fix,
  identical for every condition: a starter `build_weights.py` that already saves all-zero arrays of the right shapes,
  and a TASK.md section stating the time and step budget and asking for planning in thinking, short comments, and
  edits of at most ~100 lines with the self-test run after each. Runs that leave the starter untouched are recorded
  (`starter_unchanged`) and score as the all-zero model. The k = 47 confusion itself was the agent's: E3 says the heads
  move every key frequency; the table's k is the frequency of each head's attention pattern.
- **The model rewrites whole files, and the context filled with its own thinking.** In the fourth smoke run the agent
  ran the starter, then tried to replace it with a 362-line implementation in one `edit` (failed: old string did not
  match), then in one `write` (24,762 characters, truncated at 8,192 tokens) — the 100-line guidance was ignored.
  Every turn's thinking stays in the prompt (prompt tokens grew by the full completion of each turn, e.g.
  9,688 → 14,068 after a 4,206-token turn), and OpenCode compacted the conversation at ~29k and ~33k tokens of a 48k
  window; after each compaction the agent re-read all files. It eventually wrote the script through a shell heredoc:
  first graded agent replacement (accuracy 1.4 %, KL 2.5·10³ nats, HS 0.34, 13 steps, 40-minute cap). Fixes: a 16k
  per-turn output cap so a whole-file write fits next to ≤4k thinking; context 80k with OpenCode's model input limit set
  explicitly to context − output; the TASK.md guidance now says "build up in stages" instead of a line limit.
  Throughput at 80k is unchanged once warm (260–328 prompt / 20–22 generated tokens/s, 6,947–7,121 MiB VRAM); 96k peaked at
  7,436 MiB VRAM and was slower, so it was not used (`results/stage0/llm_tuning.json`). The machine has < 1 GB of free RAM
  while the server holds ~13.6 GB of the memory-mapped experts, so the first requests after each server start are slow
  (pages read from disk) — the server is kept running across a batch of runs.
- **The thinking budget, not the output cap, was the root cause — and the presence penalty was the wrong preset.**
  The fifth smoke run thought for its full 4,096 tokens (ending mid-question about how to set the embedding norm), then
  wrote 718 lines (537 distinct) of mostly comment-reasoning until the new 16k cap. Raising the cap only made each
  failure longer. Re-reading the [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ("Best
  Practices"): it recommends an output length of 32,768 tokens for most queries (81,920 for competition-level maths and coding), and for thinking-mode precise coding
  temperature 0.6, top-p 0.95, top-k 20, min-p 0 **and presence penalty 0.0** (1.5 is its general-task preset with
  temperature 1.0; the 1.5 chosen after run 2 came from Ollama's general default and was a mistake for this task).
  A 4k thinking budget is far outside the model's trained regime, and forcing the end of thinking moved the unfinished
  design into the code. New settings: presence penalty 0.0, thinking budget 16,384, per-turn output 24,576 (thinking +
  8k for one whole-file write), OpenCode input limit 57,344 of the 80k context, and a 60-minute run cap (one full turn
  at ~21 tokens/s takes ~20 minutes; at 40 minutes a run had room for about one and a half turns).
- **A bigger thinking budget did not resolve the design either.** In the sixth smoke run (coding preset, 16k budget)
  each of three turns used the full 16k thinking tokens (~15 minutes) and then took a trivial action (run the starter,
  re-read files); one whole-file write after 40k characters of thinking was again mostly comments (414 of 512 lines)
  and was truncated. The run ended at chance (starter unchanged, accuracy 0.0088, HS 0). The comments showed where it
  was stuck: how ReLU neurons obtain *products* of a- and b-terms when attention delivers their weighted sum.
- **The true explanation was vaguer than the wrong one — a design flaw, fixed.** The Clock text (E1–E4, W-freq,
  W-role, W-scramble) said only "the ReLU neurons form products", while the Pizza text (W-alg) stated the neuron
  function exactly, ReLU(cos(w_k a + φ) + cos(w_k b + φ)); the false explanation was therefore easier to implement,
  which would bias H2. Checked on the dev target that the precise Clock statement is true: the a- and b-terms of each
  neuron's pre-activation have median amplitude ratio 0.999 and median phase difference 0.1°, quadratic terms are
  smaller (median amplitude 0.33 vs 1.50), and one frequency explains a median 97.9 % of pre-activation variance
  (development notes: no committed script computes these medians; the per-neuron fits they summarise are in
  `results/explanations/dev/analysis.json`). The Clock MLP step (and the math/code format variants) now states pre-activation ≈ bias + A·[cos(w_k a + φ) +
  cos(w_k b + φ)] and that the ReLU of this sum contains the products; E-paper keeps the paper's original wording.
- **Custom harness instead of OpenCode.** Several stage-0 failures were OpenCode behaviour rather than model
  behaviour: truncated replies dropped from the history (so retries repeated them), the session ending after three
  truncations, compaction at a fraction of the window followed by re-reading every file, and every turn's thinking
  kept in the prompt. Following the user's suggestion (switch harness if the agent stays at chance), a minimal
  harness reusing [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)'s design was added
  (`src/vmi/simulator/harness.py`): bash as the only tool, one command per reply, linear history, observations as
  user messages (so the chat template keeps only the latest thinking), a truncated reply kept with a note, same
  sandbox, relay, limits, sampling and event format. Smoke run 7 changed harness and explanation text at once (one hour
  per run); this confound is stated wherever the switch is reported.
- **Harness bug: replies without a closed thinking tag.** Smoke run 7 ended after 3 minutes on three "format errors"
  although each reply was a correct THOUGHT plus one bash block: the model had not closed its thinking tag, so
  llama-server's reasoning parser returned everything in `reasoning_content` and `content` was empty. The harness now
  parses `reasoning_content` when `content` is empty (with a test).
- **With the harness the protocol problems are gone, but the agent is still at chance.** Smoke run 8 (E3, harness):
  all files read in 4 minutes, one 14-minute design turn, then a build–self-test–fix cycle every 2–3 minutes for 38
  steps, ending cleanly at the time limit. Every build stayed at chance (best accuracy 0.98 %; graded: 0.95 %,
  HS 0.33). Its construction was conceptually wrong: a and b share the same embedding dimensions, products were
  attempted with a hand-made polarisation trick on the mixed coordinates, every neuron wrote to the same two output
  dimensions, and it used neither the stated neuron form nor the allowed least-squares read-out. This now looks like
  the capability limit of a 3B-active model on this construction task rather than a protocol problem.
- **Claude Code mode (user decision: only if all Qwens stay at chance; Haiku → Sonnet → Opus; API credits, $20
  cap).** The same offline sandbox, with Claude Code pointed at the in-sandbox relay; outside, a proxy
  (`src/vmi/simulator/anthropic_proxy.py`) adds the API key and workspace id (neither enters the sandbox), allows only
  /v1/messages and the run's models, rejects server-side tools with internet access, prices each response's usage
  into `results/claude/ledger.jsonl` before completing it, and refuses requests that would exceed the total or per-run
  cap. A plumbing check (`vmi check --claude`) first failed with "This API key is not scoped to a workspace" (fixed by
  sending `anthropic-workspace-id`), then passed at $0.0002. A pre-push secret scan (`scripts/check_secrets.py`, also a
  test) checks every tracked file for the exact key values and real key formats.

## Grader and reference compiler (dev target only)

- **float32 grading broke a correct replacement.** The compiled E3 model reached 100 % accuracy in the numpy self-test
  (float64) but 0.4 % in the torch grader (float32): least squares on nearly collinear features produced unembedding
  weights up to 8.5·10⁴ whose cancellations fail in float32. Two fixes: (1) the grader now evaluates both models in
  float64 (exact for float32 targets, and consistent with the agent's self-test, so an agent cannot pass the self-test
  and then fail grading through rounding); (2) the compiler uses ridge least squares (λ = 1e-8·tr(XᵀX)/d) to keep weights
  moderate.
- **Four non-uniform heads can starve the MLP of the symmetric term.** With every head's attention depending on (a, b),
  no linear combination of head outputs yields ½(v_a + v_b), and a synthetic all-same-sign head configuration compiled to
  only 84–95 % accuracy. Trained models instead show a pair of heads with opposite coefficients at one frequency (paper
  Table 2; dev target heads 0 and 2), whose sum is nearly uniform — with that structure the compiler reaches 100 %.
  The compiler also lets the unembedding read the residual stream directly (skip path), which recovered 84 → 95 % in the
  adversarial case. (The 84–95 % figures are development notes, not backed by a committed results file.)
- **Bitwise reproduction depends on which runs are trained together.** Re-training `wd1_s1` and `wd1_s10` together
  (grokviz's vectorised trainer) did not reproduce the published weights, which were trained in a batch of 22 runs:
  different batch sizes change floating-point kernels, and grokking trajectories amplify the difference.
  `wd1_s10` still learned the same key frequencies {19, 31, 45, 47} and scores HS = 0.978 against the published
  model; `wd1_s1` learned {3, 6, 23, 39} instead of {2, 3, 23, 37, 39} (HS = 0.900). This gives a useful reference:
  two independently trained models with the same frequencies score HS ≈ 0.98
  (`results/training/reproduction_check.json`).

## Round 1 — a unix socket path limit failed one Claude run (2026-11-21)

**What happened.** The no-self-test ablation run ended in 3 minutes with no weights. The transcript showed Claude
Code retrying the API ten times with `error_status: null` — no HTTP response at all — and the proxy log showed why:
`OSError: AF_UNIX path too long`. The proxy listens on a unix socket inside the run's sandbox folder, whose name is the
run ID; `ablations__T0__E3-noselftest__claudehaiku55-claudecode__tdef__s0` made the path 108 bytes, and Python refuses
paths of 108 bytes or more because `sun_path` is 108 bytes including the terminating NUL ([man 7 unix](https://man7.org/linux/man-pages/man7/unix.7.html);
the kernel itself accepts 108 non-null bytes, per the page's BUGS section; wording corrected in audit 6).

**Fix.** The socket now lives in `~/vmi-sim/k/<first 16 hex chars of md5(run ID)>/llm.sock` (≈ 47 bytes for any ID),
bound to the same `/llm` inside the sandbox. The hyperparameter W-freq effort-low and effort-high run IDs (108- and
109-byte socket paths) would also have overflowed; they ran after the fix.

**Swapping a running bash script.** Two runs were executing `run_one.sh` when it had to change. Bash re-reads a script
file by byte offset as it executes, so editing it in place can make a running script execute shifted lines. A
10-second experiment confirmed that `mv new run_one.sh` from WSL onto `/mnt/c` keeps the running process on the old
content, so the fix was swapped in by atomic rename without touching the two runs in progress.

**Learning.** Separate "the agent failed" from "the infrastructure failed" by a mechanical rule (did the agent get at
least one model response?) and archive, never delete, the failed record.

## Round 1 — the simulator fills gaps from memory (2026-11-21)

**What happened.** On the published checkpoint (T-paper), the algorithm-only explanation E1 produced a replacement
that scored like a full explanation, while E1 on T0 (analysed in no paper) scored far below even the random-initialisation level (HS about 0.05). The T0 E1 build script
(`main__T0__E1…s0`) hard-coded `KEY_FREQS = [14, 35, 41, 42, 52]` with the comment "choice: not specified by
EXPLANATION.md" — the paper's published frequencies; the T-paper E1 script (s0) used `KEYS = [14, 35, 41, 42, 52, 71]`.
By the key-frequency rule all three T0 E1 replacements contain four of the paper's five frequencies (14, 35, 41, 52;
`results/frequencies_round2.json`); runs s1 and s2 hard-coded `[14, 35, 41, 52, 73]` (73 ≡ −40 mod 113). The same
happened for E0 on T0 (one run of three) and for E1 on the dev target in stage 0 (its `build_weights.py` sets
`FREQS = [14, 35, 41, 42, 52]`; `replacements/dev__dev__E1__claudehaiku55-claudecode__tdef__s12/` on the HF repo; checked
in audit 8).

**How it was checked.** Not only by reading scripts: the key-frequency rule used for the targets was applied to every
replacement's weights (`analysis/frequencies.py`), giving the overlap of the frequencies actually built in with the
target's and with the paper's sets (table in `01-round1/experiment.md`). Then by a target swap, which needs no
frequency threshold: the same T0 E0/E1 weights were graded against the published checkpoint, where they score far
higher than against T0. Comparing two runs (E1 on T0 vs. E1 on T-paper) mixes agent randomness with target choice;
re-grading one set of weights against two targets isolates the target.

**Learning.** Weights-blind does not mean knowledge-blind. Any simulation score on a published model is confounded by
memorisation; targets whose mechanism the simulator cannot have seen (no paper; weights and key frequencies uploaded
only on 2026-10-21, after the simulator's June 2026 training-data cutoff; no internet in the sandbox) and a held-out target
trained after the freeze are necessary, not optional. The
same risk applies to replication-based evaluations of LLM interpretability agents on well-studied circuits ([Haklay et al. 2026](https://arxiv.org/abs/2603.20101)).

## Round 1 — small local simulators never build a working network (2026-11-21)

**What happened.** Qwen3.5-4B (E2, E3, E4, W-freq) and Qwen3.5-9B (E2) all ended with a network whose output is the
same for every input (task accuracy exactly 1/113, HS 0). They used up the 60-step or 60-minute budget editing
`build_weights.py` and re-running the self-test (Python tracebacks in the logs of 4 of the 5 runs, counted in audit 8
from each run's `transcript.md`/`events.ndjson`, published under `replacements/<run_id>/` on the HF repo; the committed
`agent.errors` field counts harness errors, not Python exceptions), and the final weights
never passed it. Extra explanation detail did
not help: E4, with the full per-neuron table, failed like E2.

**Decision.** The user stopped the remaining local runs (9B E3–W-freq, all four 35B conditions) and dropped local
conditions from rounds 2–3. That is a stop on results, which the preregistration rules out, so it is logged as
deviation 5 with its consequence: H5's capability trend and H6's cross-simulator ranking become untestable.

**Learning.** The method has a capability floor: the simulator must write about a hundred lines of correct numpy from a
page of maths, and the 4–9B models (4-bit, in this harness) cannot. A capability axis needs simulators on both sides
of that floor, for example Haiku, Sonnet and Opus, rather than models that all sit below it.

## Round 1–2 — reading the agents' own transcripts (2026-11-22)

**What happened.** Extracting the agents' visible messages and self-test results (`analysis/agent_examples.py`)
showed three things the scores alone did not. (1) The memorisation is explicit: in the T0 algorithm-only run (`main__T0__E1…s0`) the
agent writes that the frequencies are unspecified and that it will "pick five standard ones (14, 35, 41, 42, 52)", the
published paper's set (the key-frequency rule finds four of them, 14, 35, 41, 52, in its weights). (2) The low-effort E3 run (accuracy 0.562 with the right frequencies) failed because of a layout bug in its own script: its constants put head 3's output and the readout in the same residual dimensions (P3 = RO = 52), so head 3's attention output was added onto the answer. Re-running its script with only that number changed gives accuracy 1.0000 at almost the same HS (0.91 against 0.92 as submitted; `vmi diagnose`, results/diagnoses/). The agent never found the bug: it blamed bias-free ReLU neurons, tuned four constants around it, and finally rejected its most accurate setting (0.785) as less faithful to the explanation. An earlier version of this entry guessed that the run stopped on a worse build because each rewrite replaced the last; the transcript shows a deliberate choice, and the intervention shows the real cause. (3) The small local model had the right frequencies in its script but never got a working network before its time
ran out, with replies cut off at the output limit.

**Learning.** Transcripts are evidence, not decoration, but an agent's own diagnosis can be wrong: the cause has to be
tested by intervening on its code, not read off its messages (nor guessed by the analyst, as this entry first did).
A low task accuracy with the right frequencies can be an implementation bug unrelated to the explanation, which is one
more reason the grader reports HS separately from behaviour. By default the API returns no thinking text (summaries
can be requested; these runs did not), so only the visible messages can be quoted.

## Independent audit (2026-11-22)

**What it found and how it was fixed** (reports, analysis code, and deviation 8 of the preregistration):

- **Invariance overclaimed.** HS was described as invariant to any invertible linear map. R² is invariant to invertible
  maps of the predictor only ([Kornblith et al. 2019, Table 1](https://arxiv.org/abs/1905.00414)): completeness is
  invariant to the replacement's basis, soundness only to orthogonal maps and uniform scaling; HS = 1 (up to the ridge
  term) under any exact invertible linear equivalence. Wording fixed; new grader test (see the literature entry above).
- **"Built the paper's frequencies" was too strong.** The key-frequency rule finds four of the five (14, 35, 41, 52) in
  each T0 E1 replacement, and two runs hard-coded `[14, 35, 41, 52, 73]` (`results/frequencies_round2.json`).
- **Diagnosis layout.** The low-effort E3 diagnosis moved head 3 to unused dimensions (P3 = 62); the script's own
  docstring puts heads 2/3 at 42–51, i.e. P3 = 42 was intended. The diagnosis now tests both
  (`src/vmi/analysis/diagnose.py`).
- **HS has no zero floor.** Random-initialisation networks (HS 0.236, `results/controls/random_init_T0.json`) score
  above working networks built on other frequencies (retrains 0.119, `results/baselines/T0.json`; W-freq 0.070), so
  low HS values are not a calibrated distance.
- **Preregistered statistics not computed.** H3's bootstrap CIs, H4's test against distilled students, the per-target
  replication tests (only directions were reported) and H8 over every E3/E4 replacement × all 401 checkpoints (rounds
  1–2: one replacement × 41 checkpoints, labelled "all checkpoints"). Now computed, H8 re-run as preregistered (deviation 8). Also
  disclosed there (c): the redrawn W-freq/W-scramble explanations and T-paper's extra retrains never ran, and the
  compiler's 100 % accuracy check was relaxed to > 99 %.
- **W-alg is not length-matched:** about 60 % of E3's characters on every target
  (`results/explanations/*/W-alg/length.json`; `test_length_matched_controls` checks only W-freq and W-role).
- **Citations cited for more than they say.** iWSP (Mehta et al. 2026) already uses CKA and a no-description control;
  Heap et al. 2025 evaluate SAE features with fuzzing and detection scores; Zhong et al. 2023's Clock and Pizza differ
  in their logit patterns, so they are not a same-outputs, different-internals example. Reviews corrected.
- **Claude runs had no step cap.** Text describing a Claude run "within the 60-step budget" used the local harness's
  limit; Claude runs were limited by time and budget only. (The second audit corrected the wording "Claude Code has no
  step cap": the CLI has a `--max-turns` option, which was not used.)

**Learning.** Check the code's own measurements (the key-frequency rule, the length files) before writing "built X" or
"length-matched", and check that each preregistered statistic is actually computed by the analysis code rather than
only described in the preregistration.

## Second independent audit — what a fresh pass caught (2026-11-22)

**What happened.** A second audit, made without reading the earlier reviews, re-read the code, the agents' own build
scripts and transcripts, the Hugging Face target repository and every citation (`reviews/audit-2-2026-11-22.md`,
deviation 9).

- **"No published analysis" was false.** The target repository publishes each run's key frequencies
  (`wd1_s1/analysis/key_freqs.json` lists T0's). The contamination argument still holds, but for different reasons:
  no paper, publication only after the simulator's training cutoff, no internet in the sandbox. *Learning:* check a
  data source's own files before describing what is and is not public.
- **Identical inputs are replicates.** E0 and E1 contain nothing target-specific, so their runs on T0 and on T-paper
  are draws from one condition. Reporting only the T0 runs selected half of the evidence, and the key-frequency
  threshold under-counted what the agents built (a script listing all five paper frequencies was detected as two).
  *Learning:* before comparing conditions, check whether the agent's input differs at all; prefer threshold-free
  measures (spectrum share) and the agent's own declarations next to thresholded ones.
- **A preregistered hypothesis has a scope.** H5's "agent ≤ compiler" was tested on T0; the seed targets are an
  extension. Mixing them produced six different phrasings across documents. *Learning:* generate one wording from one
  function and keep the preregistered comparison apart from extensions.
- **The agent is not a literal compiler.** It repairs a self-inconsistent explanation into a working adder and tunes
  constants against its task self-test in scratch copies the static audit never sees. *Learning:* audit the process
  (transcripts), not only the final script, and say what the score then measures.
- **Correlation along one training run needs a control.** Two quantities that both trend with training correlate;
  replacements built without the target's mechanism give the specificity check H8 lacked.
- **Small details that were simply wrong:** "Claude Code has no step cap" (it has `--max-turns`, unused here); Yip et
  al. give no closed-form construction; two cited works were missing from the reference list; the cited Heap et al.
  title belongs to a later arXiv version; the exact JT branch silently fell back to Monte-Carlo for nine values.

## Third independent audit (2026-11-22, deviation 10)

- **Recompute verdicts from the preregistration, not from the code.** An independent script written from §5 (exact
  permutation tests, scipy exact rank tests) reproduced every stored test result, and still found two reporting
  errors: H1's hypothesis has a clause ("E0 lies below E1") with no preregistered test, and the verdict silently
  dropped it although the clause fails; the T-paper replication left out the E1 runs it had. *Learning:* every clause
  of a hypothesis needs a verdict, even a direction-only one, and "the same tests" means the same levels.
- **An ablation must remove what it claims to remove.** The self-test ablation took away `selftest.py`, but the task
  labels are the task itself and `forward.py` exposes them; all three agents measured accuracy with their own code.
  The counter of "self-test runs" only counted `selftest.py`, so it hid this. *Learning:* read the transcripts of an
  ablation before interpreting it, and count the behaviour (accuracy checks), not the tool.
- **A symmetry argument covers only the sites it was made for.** "The residual stream is defined up to GL(128), so use
  linear maps and no wider" was applied to `mlp_post` too, where the symmetry is permutation and positive scaling; that
  wider alignment is a plausible reason the scrambled neuron table passes HS. *Learning:* derive the alignment class
  per site.
- **Test the ground truth of a control.** W-alg assumes T0 is a Clock. Zhong et al.'s distance irrelevance confirms
  it, and confirms that the literal Pizza build is a Pizza; but it also shows that the replacements built from the
  *true* explanations are Pizza-like: the rendered explanations prescribe the same MLP neuron function as Pizza and do
  not convey what makes T0 a Clock. *Learning:* measure the property a wrong explanation is supposed to get wrong on
  the true explanation's replacements as well.
- **Gradient-based diagnostics can point off the data manifold.** Hand-built replacements compute near-constant
  attention from their embeddings; the full gradient through the attention scores then makes Zhong et al.'s gradient
  symmetricity strongly negative even for a literal Pizza. Holding the pattern fixed recovers the expected value.
  *Learning:* prefer output-level diagnostics (distance irrelevance) and check gradient diagnostics on a known case.
- **Ties count.** "T1 has the fewest key frequencies" was a tie with `wd1_s9`; "public since" was a creation date.
  *Learning:* state the selection rule and the evidence exactly.

## Fourth independent audit (2026-11-22, deviation 11)

- **A control can fail for a reason other than the one it was built to detect.** H4b expected a planted model to drift
  away monotonically under further training; instead its function dipped and recovered (snapshot accuracy in
  `results/controls/planted_calibration_T0.json`), and KL followed it. The abstract read this as a grader failure.
  *Learning:* record the property the control assumes (here: monotone functional drift) and check it before reading
  the verdict as evidence about the grader.
- **Give a trajectory correlation a positive control, not only negative ones.** HS of the E3/E4 replacements against
  T0's checkpoints rises as restricted loss falls (H8), but during circuit formation final T0 itself, the agent-free
  compiler build and T0 restricted to its key frequencies rise in exactly the same way (ρ ≈ 1 over the formation
  phase), and after grokking the restricted loss creeps up while HS drifts monotonically, which sets the size of the
  whole-trajectory ρ — final T0 itself scores below every replacement (`results/audit4/h8_positive_control.json`).
  Negative controls (W-freq, E1, E0) showed specificity to T0's frequencies; only the positive control showed that the
  statistic cannot rank faithfulness. *Learning:* for "the
  score tracks X over training", compare with the end point itself, and ask whether the score adds anything beyond
  similarity to that end point.
- **Name the real coverage of small-sample intervals.** With three runs, the "95 %" percentile bootstrap interval of a
  mean is the range of the runs and covers about 75 % (our calculation, confirmed by simulation in audit 5; Hesterberg
  2015 says only that percentile intervals are too narrow in small samples). "Too narrow" understated this. *Learning:*
  compute what an interval means at the sample size used, not only that it is imperfect.
- **An independent re-implementation is cheap insurance.** A numpy forward pass and a scikit-learn ridge written from
  the preregistration reproduced every stored score (`results/audit4/regrade.json`), and enumeration-based exact tests
  reproduced every verdict. *Learning:* do this before reading any result, not after four audits.
- **Citations drift toward the citing paper's story.** Most corrections were narrowings: a source that compares
  features "does no hidden-state comparison", a hypothesis of the authors becomes a "because", a scoped memorisation
  finding becomes "any evaluation". *Learning:* quote the sentence being cited, with its conditions.

## Fifth independent audit (2026-11-22, deviation 12)

- **A "key frequency" is a property of one matrix.** The explainer takes key frequencies from W_L = W_U·W_out and then
  tells the agent that every component uses them. On T0, frequency 2 exists only in W_L (no neurons; below threshold in
  the embedding; consistent with a harmonic of 37), and several embeddings carry harmonic frequencies that no
  explanation mentions (`results/audit5/checks.json`). The τ = 0.35 sensitivity check had already shown the symptom
  (the compiler's HS rises when frequency 2 is dropped) without the cause. *Learning:* measure each component's own
  frequency set before writing a sentence about "all components", and treat a sensitivity result that improves the
  score as a pointer to a defect, not only as robustness.
- **Separate a metric's blind spot from the build that feeds it.** The scrambled-table "blind spot" held for the
  agents' rebuilds, but the agent-free compiler's literal rebuild of the same table lost 0.09 HS on T0. *Learning:*
  before attributing a non-separation to the metric, check it on a build that follows the wrong explanation literally.
- **Exact methods make a re-derivation seed-free.** With three runs per group, every bootstrap resample can be listed
  (27 per group), and every rank-test permutation too, so an independent check of every stored statistic needs no
  random numbers and no tolerance for Monte-Carlo error (`scripts/audit5/hypotheses_check.py`; 88 of 88 items). *Learning:* at small N,
  prefer enumeration to sampling in both the analysis and its audit.
- **Counting "after results" honestly.** A summary said seven deviations followed results; the first deviation itself
  said per-run scores had been seen. *Learning:* derive such counts from the deviation texts, not from memory of which
  ones felt results-driven.
- **Check the inputs of the tests, not only the tests.** Re-deriving verdicts from stored scores verifies the
  statistics; re-grading the scores with a second implementation written from the agents' own specification
  (`ARCHITECTURE.md`) and the preregistration verifies the grader as well (306 gradings, `scripts/audit5/regrade_independent.py`).
  It also surfaced a choice the preregistration leaves open — the R² denominator about the training-fold or the
  held-out mean — whose effect (≤ 7.6e-4 in HS) is now measured instead of assumed. *Learning:* when writing a
  preregistered metric, state the fold procedure and the R² denominator, not just the seed.

## Sixth independent audit (2026-11-22, deviation 13)

- **A pattern count is a classifier; validate it before quoting it.** The paper's "41 of 96 runs announce a sweep"
  came from a regular expression over the agents' narration. Reading every matched sentence found 2 false matches
  ("attention weights tuned by the explanation's α/β", "before deciding whether to tune anything") and 3 mentions of
  tuning with no search described; reading the candidate sentences of the other runs found 8 searches the pattern
  missed ("try smaller bias scales", "grid the bias", "the bias scan plateaued"). The read count is 44
  (`results/audit6/sweep_labels.json`). The errors happened to cancel, but the claim attached to the count ("keeping the
  value with the best accuracy or fit") was never measured. *Learning:* when a count from a pattern supports a claim,
  read its matches and a broad sample of its non-matches, and say only what the pattern detects.
- **A verdict rule applied after the verdict is seen needs both readings on the page.** H2's "cannot tell at this N"
  comes from a rule added in deviation 1 and applied at N = 3 by deviation 7 after the N = 3 verdicts were first read;
  §5's falsification column, read literally, gives "not supported". Both are defensible (a 3-vs-3 exact test has no
  power), and showing both costs one clause. *Learning:* when a protocol amendment relabels a negative verdict, keep
  the literal reading visible next to it.
- **Describe a figure by what is drawn.** Zhong et al.'s Fig. 7 has phase lines in attention-rate × width space,
  fitted to the labels "gradient symmetricity > 98 %" and "distance irrelevance < 0.6"; two earlier passes had
  described lines at those values. *Learning:* quote the caption when citing a figure's boundaries.
- **Check what a source says about your own design choice.** The README called restricting HS to linear maps "our
  inference, not theirs", but Sutter et al. suggest it themselves; Nanda et al. take their main model's key
  frequencies from the embedding (six frequencies, of which the five used downstream; corrected in audit 7), not from
  W_L, which on T0 is exactly where frequency 2 appears or not. *Learning:* when a design
  choice is motivated by a paper, look for that paper's own recommendation before claiming or disclaiming it.

## Seventh independent audit (2026-11-22, deviation 14)

- **An audit that matches patterns certifies only those patterns.** The static audit of `build_weights.py` looks for
  gradient-descent and optimiser vocabulary, and the paper read "build scripts are audited" as compliance with "no loop
  that updates weights to reduce a loss". Reading every final script found 7 of 96 with a derivative-free search over
  one constant (3–231 values, or 20,000 random frequency sets, or an 8-step calibration), kept by a fit, task accuracy
  or a formula (`results/audit7/build_script_searches.json`). None is in a hypothesis test, and dropping them changes no
  seed-target reading except one sign-only count. *Learning:* state what an automated check detects, and validate it
  by reading the inputs once; for round 3, flag any variable that keeps "the best" candidate.
- **A figure is a source too.** "The published explanation leaves the logit scale unspecified" was wrong: Nanda et al.
  plot the fitted α_k over training (App. C.1.5, Fig. 14) without giving numbers in the text. *Learning:* check the
  appendix figures before saying a paper does not give a quantity.
- **Paraphrases drift toward the claim they support.** Gross et al. measure proof length, not explanation length;
  Gromov's weights are approximate; Huang et al. report causal effects "similar to the random baseline", not "little
  to no causal efficacy"; Poché et al.'s shortcut hypothesis is about task priors. Each paraphrase was close but
  stronger or broader than the source. *Learning:* paraphrase from the source's own sentence and keep its qualifiers.
- **Exact enumeration is cheap at N = 3.** Every preregistered test at this N has a small null distribution
  (369,600 assignments at most), and the bootstrap interval of a three-run mean is exactly the range of the runs, so
  the verdicts can be re-derived with no statistics library and no Monte-Carlo error (`scripts/audit7/verdicts.py`).

## Eighth independent audit (2026-11-23, deviation 15)

- **Lead with the preregistered verdict.** A rule added after some scores were seen ("cannot tell" for a powerless
  test) was used as the primary label for seven audits. Each audit noted the literal reading, but none put it first.
  The preregistration had anticipated the N = 3 floor in §6 and still defined "cannot tell" only for undefined tests.
  Lesson: when a post-hoc rule and the preregistered rule disagree, the preregistered verdict is primary, and the
  post-hoc one is shown beside it with its deviation number.
- **Ablations need a sandbox-wide check, not a file-wise one.** The self-test ablation asserted that `TASK.md` no
  longer mentioned the self-test, but the starter script still did, and every agent read it. Lesson: for an ablation,
  grep every file the agent can see, and test that check.
- **A control is only as strong as its hardest case.** H7's wrong targets share at most one key frequency with T0, so H7
  measures frequency identity, the same thing HS already separates. Lesson: include a negative control that shares
  the property the metric is known to see (here, the key frequencies).
- **Sources for "obvious" numbers.** Two learnings that were true but cited nothing (a stage-0 build's frequencies and
  Python tracebacks in the local runs' logs) now point to the run folders on Hugging Face. An agent sweep had suggested
  removing one of them as wrong; reading the logs showed it was right.


## Ninth independent audit (2026-11-23, deviation 16)

- **Check what a baseline actually receives, not what it is called.** The "logit-distilled" students were trained with
  KL at temperature 1 against a teacher that puts 0.9999998 of its mass on the answer. The wrong-class logits got
  almost no gradient, so the students copied the answers and not the logits (centred-logit FVU against T0 about 1).
  Hinton et al. (2015, §2) raise the temperature for exactly this reason. Lesson: for every baseline, measure the
  property its name promises (here, logit agreement) before reading anything from it.
- **"Outputs cannot see the mechanism" depends on which outputs.** Output KL failed, but a grokked model's centred
  logits are Σ_k α_k cos(w_k(a + b − c)) and carry its key frequencies. A logit comparison with one fitted scale
  separated the explanations about as well as HS. Lesson: when a primary output metric fails, try the strongest output
  comparison before concluding that internal states are needed; report probability-level and logit-level results
  separately.
- **"No case exists" needs a search of the data you already have.** The paper said that no Clock-side build of an
  explanation existed. The project's own Clock/Pizza table listed five builds with q ≥ 0.4, and their HS answered the
  question the sentence said was unanswerable. Lesson: before writing "not measured", filter the existing results for
  the case.
- **Explain failed sub-predictions instead of only reporting them.** E0 above E1 was reported as a failed prediction
  for eight audits. The frequency table already showed why: broad-spectrum builds sit at HS's floor, which is above
  what sparse builds on other frequencies get. Lesson: a failed direction is a finding about the metric; look for its
  cause in the data before moving on.
- **Pre-specify post-hoc tests too.** The logit-matched students' design, training criterion and reading were
  committed and pushed before training, so their result could not shape their interpretation.
