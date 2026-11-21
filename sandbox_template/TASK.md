# Task: build a network from its explanation

A transformer was trained to compute `(a + b) mod 113`. You cannot see its weights. `EXPLANATION.md` describes how
it works (it may also point to a CSV table in this folder). Your job: **write the weights of a network with the same
architecture (`ARCHITECTURE.md`) that implements the mechanism described in `EXPLANATION.md` as faithfully as you
can.**

## How your network will be judged

Your network is compared with the original on all 12,769 inputs:

1. **Outputs:** how close your output probabilities are to the original's.
2. **Internal activations:** how well the original's activations at the "=" position can be linearly predicted from
   yours, and vice versa, at three places: after attention (`resid_mid`), the MLP neurons after ReLU (`mlp_post`), and
   after the MLP (`resid_post`). The comparison allows any invertible linear change of the residual-stream basis and
   any reordering/positive rescaling of neurons, so you do not need the original's exact basis — but you do need the
   same *quantities* to be computed in the same places.

So a network that merely gets the right answers in a different way scores worse than one that follows the
explanation. If the explanation leaves something unspecified, make a reasonable choice and note it in a one-line
comment.

## Rules

- Complete the starter script `build_weights.py` in this folder; it must create `weights.npz` (11 arrays, shapes in
  `ARCHITECTURE.md`; the starter already saves all-zero arrays of the right shapes). Use only Python 3 and numpy. It
  must finish in under 60 seconds.
- **No training.** Do not use gradient descent, optimisers, or any loop that updates weights to reduce a loss.
  Direct construction is expected. Closed-form linear algebra (for example `np.linalg.lstsq` for one linear layer)
  is allowed.
- `forward.py` (reference forward pass) and `selftest.py` (checks shapes and task accuracy) are provided; do not edit
  them. Run `python3 build_weights.py` and then `python3 selftest.py` and fix problems until the self-test passes and
  the accuracy is as high as you can get it.
- There is no internet access. Work only inside this folder.

## Working within your budget

You have about {MINUTES} minutes and {STEPS} steps, and generating a long message takes time, so very long messages
cost minutes. Do your planning while thinking, not in code comments; keep comments short. Build up
`build_weights.py` in stages and run the script and the self-test after each stage, so that a working
`weights.npz` exists early and then improves.

When you are done, `build_weights.py` and `weights.npz` must exist in this folder. Then stop.
