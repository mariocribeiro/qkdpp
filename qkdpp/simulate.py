"""Synthetic raw-data generator, for testing the pipeline without an experiment.

Models a prepare-and-measure BB84 run: Alice draws a bit and a basis, Bob draws
a basis, the channel loses pulses with probability 1 - eta and flips matched-basis
outcomes with probability qber. Mismatched bases give uniformly random outcomes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RawData:
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bits: np.ndarray
    bob_bases: np.ndarray
    detected: np.ndarray

    def __len__(self):
        return len(self.alice_bits)


def generate_raw(n_pulses, qber=0.02, eta=1.0, p_z=0.5, seed=None):
    rng = np.random.default_rng(seed)
    a_bits = rng.integers(0, 2, n_pulses, dtype=np.uint8)
    a_bases = (rng.random(n_pulses) >= p_z).astype(np.uint8)   # 0 = Z, 1 = X
    b_bases = (rng.random(n_pulses) >= p_z).astype(np.uint8)
    detected = rng.random(n_pulses) < eta

    match = a_bases == b_bases
    flip = rng.random(n_pulses) < qber
    b_bits = np.where(match,
                      a_bits ^ flip.astype(np.uint8),
                      rng.integers(0, 2, n_pulses, dtype=np.uint8)).astype(np.uint8)

    return RawData(a_bits, a_bases, b_bits, b_bases, detected)


def generate_sifted(n, qber=0.02, seed=None):
    """Shortcut: a correlated pair of already-sifted strings over a BSC."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n, dtype=np.uint8)
    b = a ^ (rng.random(n) < qber).astype(np.uint8)
    return a, b
