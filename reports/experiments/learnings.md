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
- **Port 8080 could not be bound** on this machine although nothing was listed as listening (likely a WSL
  mirrored-networking reservation); llama-server moved to port 8091.

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
