"""Error verification and privacy amplification.

Both use Toeplitz matrices, which form a universal_2 family and are therefore
valid both as verification hashes and as quantum-proof strong extractors via
the leftover hash lemma.

The matrix is never materialised densely: row i of a Toeplitz matrix built from
a seed d is the reversed window d[i:i+n], so a stride view gives the whole
matrix for free and the product is done in row chunks.
"""

import numpy as np


def toeplitz_seed(n_in, n_out, rng):
    return rng.integers(0, 2, n_in + n_out - 1, dtype=np.uint8)


def toeplitz_apply(seed, bits, n_out, method="auto"):
    """GF(2) product of the Toeplitz matrix defined by `seed` with `bits`.

    Row i of the matrix is seed[n-1+i-j] for j = 0..n-1, so the product is a
    slice of the linear convolution of seed with bits:

        out[i] = conv(seed, bits)[n - 1 + i]

    which the FFT evaluates in O(N log N) instead of O(n * n_out). Convolution
    values are bounded by n, far below 2**53, so float64 rounding is exact for
    any block size of practical interest.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if n_out <= 0:
        return np.zeros(0, dtype=np.uint8)
    if len(seed) < n + n_out - 1:
        raise ValueError("seed too short for the requested output length")
    d = np.asarray(seed[: n + n_out - 1], dtype=np.uint8)

    if method == "auto":
        method = "fft" if n * n_out > 2 ** 22 else "direct"

    if method == "direct":
        win = np.lib.stride_tricks.sliding_window_view(d, n)
        x = bits[::-1].astype(np.float64)
        out = np.empty(n_out, dtype=np.uint8)
        for s in range(0, n_out, 1024):
            e = min(s + 1024, n_out)
            blk = np.ascontiguousarray(win[s:e]).astype(np.float64)
            out[s:e] = (blk @ x).astype(np.int64) & 1
        return out

    L = 1 << int(np.ceil(np.log2(len(d) + n - 1)))
    conv = np.fft.irfft(np.fft.rfft(d, L) * np.fft.rfft(bits, L), L)
    return (np.rint(conv[n - 1: n - 1 + n_out]).astype(np.int64) & 1).astype(np.uint8)


def verify(alice, bob, tag_len=64, seed=None, channel=None):
    """Confirm both strings agree. Failure probability <= 2**-tag_len."""
    rng = np.random.default_rng(seed)
    n = len(bob)
    s = toeplitz_seed(n, tag_len, rng)
    t_bob = toeplitz_apply(s, bob, tag_len)
    t_alice = alice.toeplitz_tag(s, tag_len)
    if channel is not None:
        channel.message()
    return bool(np.array_equal(t_alice, t_bob)), tag_len


def amplify(bits, n_out, seed=None):
    """Privacy amplification. The seed is public and must be shared by both
    parties; it is not counted as leakage."""
    rng = np.random.default_rng(seed)
    bits = np.asarray(bits, dtype=np.uint8)
    s = toeplitz_seed(len(bits), n_out, rng)
    return toeplitz_apply(s, bits, n_out), s
