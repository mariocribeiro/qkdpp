"""Cascade information reconciliation (Brassard-Salvail 1993).

Bob drives the protocol and only ever learns about Alice's string through
KeyHolder.parity(), so channel.leaked_bits is by construction an upper bound
on leak_IR.

The implementation tracks the known parity difference of every block. This
matters for efficiency: when a bit is corrected, every block containing it
flips parity, so a block previously known to match is now known to mismatch.
Querying Alice again would waste one bit per cascaded block -- instead the
binary search starts immediately.

Block-size schedules:
  'original'  k1 = 0.73/Q, doubling            (Brassard-Salvail)
  'optimized' k1 = ALPHA/Q, doubling, then large blocks for the tail passes

The optimized schedule follows the spirit of Martinez-Mateo, Pacher, Peev,
Ciurana & Martin, QIC 15(5-6):453-477 (2015), arXiv:1407.3257, with ALPHA
tuned numerically here. It is NOT a faithful reimplementation of their best
variant; for f near 1.03 use their published parameters together with the
confidence-grouping of Pacher, Grabenweger, Martinez-Mateo & Martin, ISIT 2015.
"""

from collections import deque
import numpy as np

ALPHA = 1.0   # k1 = ALPHA / Q, tuned numerically (see notes/TUNING.md)


def _schedule(qber, n, n_passes, kind):
    # A block spanning the whole string is useless: parity is permutation-
    # invariant, so a pass with one block is IDENTICAL every time it repeats
    # and contributes zero new information after the first occurrence. Any
    # even-sized error cluster inside it is then invisible forever. Capping
    # at n // 2 guarantees at least 2 blocks per pass, so each pass's fresh
    # random permutation has a real chance of splitting a hidden cluster.
    max_block = max(2, n // 2)

    # A raw sample can report qber=0 even when the true rate is not (e.g. a
    # 100-bit sample missing every error at a true 1-2% rate). Flooring the
    # estimate at "can't claim confidence below ~1 error per n bits sampled"
    # keeps k1 from blowing up past max_block in exactly that case.
    q_floor = max(1.0 / max(n, 4), 1e-4)
    q = min(max(qber, q_floor), 0.4)

    if kind == "original":
        k = max(2, int(np.ceil(0.73 / q)))
        return [min(max_block, k * 2 ** i) for i in range(n_passes)]
    if kind == "optimized":
        k1 = max(2, int(2 ** np.round(np.log2(ALPHA / q))))
        ks = [k1, 2 * k1, 4 * k1]
        while len(ks) < n_passes:
            ks.append(max(ks[-1], n // 2))
        return [min(max_block, k) for k in ks[:n_passes]]
    raise ValueError(f"unknown schedule {kind!r}")


def _binary_search(alice, bob, idx):
    """Block is known to have odd parity difference: locate and flip the error."""
    idx = np.asarray(idx)
    while len(idx) > 1:
        half = idx[: len(idx) // 2]
        if alice.parity(half) ^ int(bob[half].sum() & 1):
            idx = half
        else:
            idx = idx[len(idx) // 2:]
    pos = int(idx[0])
    bob[pos] ^= 1
    return pos


def reconcile(alice, bob, qber, n_passes=10, schedule="optimized",
              seed=None, channel=None):
    """Correct `bob` in place to agree with Alice's string.

    alice : KeyHolder
    bob   : uint8 array, modified in place
    """
    n = len(bob)
    rng = np.random.default_rng(seed)
    sizes = _schedule(qber, n, n_passes, schedule)

    blocks, membership, state = [], [], []
    queue = deque()
    corrections = 0

    for p, k in enumerate(sizes):
        perm = rng.permutation(n)
        blk = [perm[i:i + k] for i in range(0, n, k)]
        mem = np.empty(n, dtype=np.int64)
        for b, ix in enumerate(blk):
            mem[ix] = b
        blocks.append(blk)
        membership.append(mem)
        state.append(np.full(len(blk), -1, dtype=np.int8))   # -1 = unknown
        if channel is not None:
            channel.message()

        queue.extend((p, b) for b in range(len(blk)))

        while queue:
            pp, bb = queue.popleft()
            ix = blocks[pp][bb]
            if state[pp][bb] == -1:
                state[pp][bb] = alice.parity(ix) ^ int(bob[ix].sum() & 1)
            if state[pp][bb] == 0:
                continue

            pos = _binary_search(alice, bob, ix)
            corrections += 1
            state[pp][bb] = 0
            if channel is not None:
                channel.message()

            for q_ in range(p + 1):
                cb = int(membership[q_][pos])
                if (q_, cb) == (pp, bb) or state[q_][cb] == -1:
                    continue
                state[q_][cb] ^= 1
                if state[q_][cb] == 1:
                    queue.append((q_, cb))

    return {"corrections": corrections, "block_sizes": sizes}
