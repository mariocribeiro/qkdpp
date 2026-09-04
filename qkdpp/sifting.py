"""Basis reconciliation (sifting) and QBER estimation.

Basis announcements are not counted as leakage: the basis choice is independent
of the bit value, so revealing it gives Eve nothing about the key. Sampled bits
used for QBER estimation are removed from the string entirely, so their cost is
a shorter raw key rather than a leakage term.
"""

import numpy as np


def sift(raw, keep_basis=None):
    """Keep positions that were detected and measured in matching bases.

    keep_basis: None keeps both bases; 0 or 1 keeps only that basis (e.g. the
    key basis in an asymmetric protocol).
    """
    m = raw.detected & (raw.alice_bases == raw.bob_bases)
    if keep_basis is not None:
        m &= raw.alice_bases == keep_basis
    return raw.alice_bits[m].copy(), raw.bob_bits[m].copy(), m


def estimate_qber(alice, bob, fraction=0.1, seed=None, channel=None):
    """Sample-based QBER estimate. Returns (qber, alice_rest, bob_rest, n_sampled).

    `alice` is a KeyHolder; sampled positions are revealed through it and then
    dropped from both strings.
    """
    n = len(alice)
    k = max(1, int(round(fraction * n)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    keep = np.ones(n, dtype=bool)
    keep[idx] = False

    a_sample = alice.sample(idx)
    if channel is not None:
        channel.message()
    qber = float(np.mean(a_sample != bob[idx]))
    return qber, keep, k


def true_qber(alice_bits, bob_bits):
    """Diagnostic only: requires both raw strings in the clear."""
    return float(np.mean(np.asarray(alice_bits) != np.asarray(bob_bits)))
