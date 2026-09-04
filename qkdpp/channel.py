"""Public classical channel with mandatory leakage accounting.

Every bit of information about Alice's string that reaches Bob must pass
through PublicChannel.reveal(). KeyHolder wraps Alice's bits so that they
cannot be read directly: the only exported operations are parity queries
and hash tags, both metered.

Swapping PublicChannel for a socket-backed implementation turns the whole
pipeline into a real two-party protocol without touching the algorithms.
"""

import numpy as np


class PublicChannel:
    def __init__(self):
        self.leaked_bits = 0
        self.messages = 0
        self.breakdown = {}

    def reveal(self, nbits=1, tag="misc"):
        self.leaked_bits += nbits
        self.breakdown[tag] = self.breakdown.get(tag, 0) + nbits
        return nbits

    def message(self, n=1):
        self.messages += n

    def reset(self):
        self.__init__()

    def __repr__(self):
        return (f"PublicChannel(leaked={self.leaked_bits} bits, "
                f"messages={self.messages}, breakdown={self.breakdown})")


class KeyHolder:
    """Alice's side. Bits are private; access is metered."""

    def __init__(self, bits, channel):
        self._bits = np.asarray(bits, dtype=np.uint8)
        self.ch = channel

    def __len__(self):
        return len(self._bits)

    def parity(self, idx, tag="cascade"):
        self.ch.reveal(1, tag)
        return int(self._bits[idx].sum() & 1)

    def toeplitz_tag(self, seed, tag_len, tag="verification"):
        """Alice computes a hash tag of her own string locally and publishes it.
        The computation is local; publishing costs tag_len bits."""
        from .extract import toeplitz_apply
        self.ch.reveal(tag_len, tag)
        return toeplitz_apply(seed, self._bits, tag_len)

    def sample(self, idx, tag="parameter_estimation"):
        """Reveal sampled positions. These are discarded from the key, so the
        cost is the loss of raw bits, not a leakage term -- hence tag only."""
        self.ch.reveal(len(idx), tag)
        return self._bits[idx].copy()

    def _diagnostic_bits(self):
        """Simulation diagnostics only, deliberately private. Nothing in the
        reconciliation path may call this."""
        return self._bits.copy()


def _gf2_matvec(M, v):
    return (M.astype(np.int64) @ v.astype(np.int64) & 1).astype(np.uint8)