"""Zhong et al. (2023) Clock/Pizza diagnostics: known answers on synthetic logit patterns and on a constant-attention model.

Global context: `vmi.analysis.clock_pizza` checks the post-hoc assumption that T0 is a Clock model (audit 3).
"""

import numpy as np
import torch
from grokviz.model import GrokTransformer, ModelShape

from vmi.analysis import clock_pizza as cp

P = 113
A, B = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")


def test_pizza_pattern_has_distance_irrelevance_zero():
    # Pizza's correct logit depends on a − b only (factor |cos(w_k(a − b)/2)|), so it is constant within a difference
    L = sum(np.abs(np.cos(np.pi * k * (A - B) / P)) for k in (14, 35, 41))
    assert cp.distance_irrelevance(L) < 1e-12


def test_clock_like_pattern_independent_of_difference_has_distance_irrelevance_near_one():
    rng = np.random.default_rng(0)
    L = 10 + rng.normal(size=(P, P))  # no dependence on a − b
    assert 0.95 < cp.distance_irrelevance(L) <= 1.01


def test_constant_attention_gives_gradient_symmetricity_one():
    torch.manual_seed(0)
    m = GrokTransformer(ModelShape()).double()
    with torch.no_grad():
        m.blocks[0].attn.W_Q.zero_()  # scores 0 → uniform attention over the three positions
    assert abs(cp.gradient_symmetricity(m) - 1.0) < 1e-9


def test_input_dependent_attention_breaks_gradient_symmetry():
    torch.manual_seed(0)
    m = GrokTransformer(ModelShape()).double()
    with torch.no_grad():
        m.blocks[0].attn.W_Q.mul_(30.0)
        m.blocks[0].attn.W_K.mul_(30.0)  # sharp, token-dependent attention
    assert cp.gradient_symmetricity(m) < 0.99
