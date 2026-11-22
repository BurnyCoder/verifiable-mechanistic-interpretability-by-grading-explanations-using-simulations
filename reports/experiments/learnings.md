# Learnings

Running log of what was learned from the literature, from errors and from development, and why each fix was chosen.
Newest entries at the bottom. Per-experiment learnings live in each experiment's folder.

## From the literature and SOTA review (2026-11-21)

- **Same behaviour ≠ same mechanism is documented in exactly this task family.** Clock vs Pizza networks
  ([Zhong et al. 2023](https://arxiv.org/abs/2306.17844)) and Tracr's compressed models (layer cosine ≈ 0.8 at full
  accuracy; [Lindner et al. 2023](https://arxiv.org/abs/2301.05062)) motivated a hidden-state metric next to output KL.
- **The alignment class must match the model's symmetries.** Without LayerNorm the residual stream is defined only up
  to GL(128), so CKA/Procrustes (orthogonal invariance) would penalise functionally identical replacements; unrestricted
  non-linear maps make anything match anything ([Sutter et al. 2025](https://arxiv.org/abs/2507.08802)). Hence
  cross-validated *linear* regression in both directions as the primary hidden-state score. The test
  `test_invariant_to_residual_basis_change` confirms HS = 1 and CKA < 1 under a random invertible change of basis.
- **Agents can recall published circuits** ([Haklay et al. 2026](https://arxiv.org/abs/2603.20101)); the paper's
  frequencies {14, 35, 41, 42, 52} are public. Hence unpublished targets and a held-out target trained after the freeze.
- **Simulators can ignore explanations** ([Poché et al. 2026](https://arxiv.org/abs/2609.08585)) → E0 (no explanation)
  and E1 (no model-specific numbers) controls.

## Engineering (2026-11-21)

- **Ollama vs llama-server.** Ollama's OpenAI endpoint ignores client context settings and defaults to 4k context on
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
  batch of agent runs. At 64k context the non-expert weights plus KV cache used 5.1 GB of VRAM; at 32k, 2.9 GB.
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
  Throughput at 80k is unchanged once warm (260–309 prompt / 21 generated tokens/s, 6.9–7.1 GB VRAM); 96k peaked at
  7.4 GB VRAM and was slower, so it was not used (`results/stage0/llm_tuning.json`). The machine has < 1 GB of free RAM
  while the server holds ~13.6 GB of the memory-mapped experts, so the first requests after each server start are slow
  (pages read from disk) — the server is kept running across a batch of runs.
- **The thinking budget, not the output cap, was the root cause — and the presence penalty was the wrong preset.**
  The fifth smoke run thought for its full 4,096 tokens (ending mid-question about how to set the embedding norm), then
  wrote 718 lines (537 distinct) of mostly comment-reasoning until the new 16k cap. Raising the cap only made each
  failure longer. Re-reading the [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ("Best
  Practices"): it recommends ≥32,768 output tokens for most queries, and for thinking-mode precise coding
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
  smaller (median amplitude 0.33 vs 1.50), and one frequency explains a median 97.9 % of pre-activation variance. The
  Clock MLP step (and the math/code format variants) now states pre-activation ≈ bias + A·[cos(w_k a + φ) +
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
  adversarial case.
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
run ID; `ablations__T0__E3-noselftest__claudehaiku55-claudecode__tdef__s0` made the path 108 bytes, and Linux allows
107 (`sun_path` is 108 bytes including the terminating NUL; [man 7 unix](https://man7.org/linux/man-pages/man7/unix.7.html)).

**Fix.** The socket now lives in `~/vmi-sim/k/<first 16 hex chars of md5(run ID)>/llm.sock` (≈ 47 bytes for any ID),
bound to the same `/llm` inside the sandbox. Run IDs with seed ≥ 10 (round 3) would have pushed more conditions over
the limit, so this would have recurred.

**Swapping a running bash script.** Two runs were executing `run_one.sh` when it had to change. Bash re-reads a script
file by byte offset as it executes, so editing it in place can make a running script execute shifted lines. A
10-second experiment confirmed that `mv new run_one.sh` from WSL onto `/mnt/c` keeps the running process on the old
content, so the fix was swapped in by atomic rename without touching the two runs in progress.

**Learning.** Separate "the agent failed" from "the infrastructure failed" by a mechanical rule (did the agent get at
least one model response?) and archive, never delete, the failed record.

## Round 1 — the simulator fills gaps from memory (2026-11-21)

**What happened.** On the published checkpoint (T-paper), the algorithm-only explanation E1 produced a replacement
that scored like a full explanation, while E1 on the unpublished T0 scored at the floor. The build script hard-coded
`KEY_FREQS = [14, 35, 41, 42, 52]` with the comment "choice: not specified by EXPLANATION.md" — the paper's published
frequencies. The same happened on T0 for E0 and E1 and on the dev target in stage 0.

**How it was checked.** Not only by reading scripts: the key-frequency rule used for the targets was applied to every
replacement's weights (`analysis/frequencies.py`), giving the overlap of the frequencies actually built in with the
target's and with the paper's sets (table in `01-round1/experiment.md`). Then by a target swap, which needs no
frequency threshold: the same T0 E0/E1 weights were graded against the published checkpoint, where they score far
higher than against T0. Comparing two runs (E1 on T0 vs. E1 on T-paper) mixes agent randomness with target choice;
re-grading one set of weights against two targets isolates the target.

**Learning.** Weights-blind does not mean knowledge-blind. Any simulation score on a published model is confounded by
memorisation; unpublished targets (and a held-out target trained after the freeze) are necessary, not optional. The
same applies to any LLM-based interpretability evaluation on famous models ([Haklay et al. 2026](https://arxiv.org/abs/2603.20101)).

## Round 1 — small local simulators never build a working network (2026-11-21)

**What happened.** Qwen3.5-4B (E2, E3, E4, W-freq) and Qwen3.5-9B (E2) all ended with a network whose output is the
same for every input (task accuracy exactly 1/113, HS 0). They used up the 60-step or 60-minute budget editing
`build_weights.py` and re-running the self-test (Python exceptions in 4 of the 5 event logs), and the final weights
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
showed three things the scores alone did not. (1) The memorisation is explicit: in the algorithm-only run the agent
writes that the frequencies are unspecified and that it will "pick five standard ones (14, 35, 41, 42, 52)", the
published paper's set. (2) The low-effort E3 run (accuracy 0.562 with the right frequencies) failed because of a layout bug in its own script: its constants put head 3's output and the readout in the same residual dimensions (P3 = RO = 52), so head 3's attention output was added onto the answer. Re-running its script with only that number changed gives accuracy 1.0000 at the same HS (`vmi diagnose`, results/diagnoses/). The agent never found the bug: it blamed bias-free ReLU neurons, tuned four constants around it, and finally rejected its most accurate setting (0.785) as less faithful to the explanation. An earlier version of this entry guessed that the run stopped on a worse build because each rewrite replaced the last; the transcript shows a deliberate choice, and the intervention shows the real cause. (3) The small local model had the right frequencies in its script but never got a working network before its time
ran out, with replies cut off at the output limit.

**Learning.** Transcripts are evidence, not decoration, but an agent's own diagnosis can be wrong: the cause has to be
tested by intervening on its code, not read off its messages (nor guessed by the analyst, as this entry first did).
A low task accuracy with the right frequencies can be an implementation bug unrelated to the explanation, which is one
more reason the grader reports HS separately from behaviour. The model's hidden thinking is not returned by the API,
so only its visible messages can be quoted.
