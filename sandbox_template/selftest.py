"""Check weights.npz: names, shapes, finiteness, and accuracy on the task (a + b) mod 113.

Usage: python3 selftest.py [path/to/weights.npz]   (default: ./weights.npz)
This only checks the task itself; it knows nothing about the original model.
"""

import sys

import numpy as np

from forward import P, all_inputs, forward, labels

SHAPES = {
    "W_E": (128, 114), "W_pos": (3, 128), "W_Q": (4, 32, 128), "W_K": (4, 32, 128), "W_V": (4, 32, 128),
    "W_O": (128, 128), "W_in": (512, 128), "b_in": (512,), "W_out": (128, 512), "b_out": (128,), "W_U": (128, 114),
}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "weights.npz"
    try:
        with np.load(path, allow_pickle=False) as f:
            w = {k: f[k].astype(np.float64) for k in f.files}
    except Exception as exc:  # noqa: BLE001 — report any loading problem plainly
        print(f"FAIL: cannot load {path}: {exc}")
        return 1
    ok = True
    for name, shape in SHAPES.items():
        if name not in w:
            print(f"FAIL: missing array {name}")
            ok = False
        elif w[name].shape != shape:
            print(f"FAIL: {name} has shape {w[name].shape}, expected {shape}")
            ok = False
        elif not np.all(np.isfinite(w[name])):
            print(f"FAIL: {name} contains NaN or Inf")
            ok = False
    extra = sorted(set(w) - set(SHAPES))
    if extra:
        print(f"note: extra arrays ignored: {extra}")
    if not ok:
        return 1
    acts = forward(w, all_inputs())
    logits = acts["logits"][:, -1, :P]  # read-out at the '=' position, 113 number classes
    pred = logits.argmax(-1)
    y = labels()
    acc = float((pred == y).mean())
    z = logits - logits.max(-1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    print(f"shapes OK; accuracy on all 12769 inputs: {acc:.4f}")
    print(f"mean probability of the correct answer: {prob[np.arange(len(y)), y].mean():.4f}")
    print(f"logit std at '=': {logits.std():.3f}; fraction of dead neurons at '=': "
          f"{float((acts['mlp_post'][:, -1].max(0) == 0).mean()):.3f}")
    wrong = np.nonzero(pred != y)[0][:5]
    for i in wrong:
        a, b = divmod(int(i), P)
        print(f"  example error: {a} + {b} -> predicted {pred[i]}, correct {y[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
