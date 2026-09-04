"""End-to-end post-processing pipeline."""

from dataclasses import dataclass, field
import numpy as np

from .channel import PublicChannel, KeyHolder
from . import sifting, cascade, extract, keylen
from .keylen import AsymptoticPolicy


@dataclass
class Result:
    key: np.ndarray
    n_sifted: int
    n_reconciled: int
    qber: float
    leak_ec: int
    leak_ev: int
    corrections: int
    qber_true: float
    leak_total: int
    efficiency: float
    messages: int
    final_len: int
    verified: bool
    channel: PublicChannel = field(repr=False, default=None)

    def summary(self):
        return (f"sifted={self.n_sifted}  qber={self.qber:.4f}"
                f"({self.qber_true:.4f} true)  "
                f"leak_ec={self.leak_ec}  f_ec={self.efficiency:.3f}  "
                f"messages={self.messages}  final={self.final_len} bits")


def run(alice_bits, bob_bits, *, pe_fraction=0.1, n_passes=10,
        schedule="optimized", tag_len=64, e_ph=None,
        policy=None, seed=None):
    """Post-process a pair of sifted strings into a final key.

    This is a plain post-processing tool, not a security-proof reference
    implementation: it estimates the error rate by sacrificing a public
    sample, corrects errors while tracking exactly what crosses the public
    channel (and is therefore exposed to Eve), and compresses by that same
    estimate -- mirroring what a lab pipeline like AIT-QKD does. It does not
    add statistical confidence margins to the error-rate estimate; use
    keylen.qber_upper_bound(...) yourself and pass the result as e_ph if you
    need a defensible security bound instead of a point estimate.

    alice_bits, bob_bits : uint8 arrays of equal length
    e_ph                 : error rate to use for sizing privacy amplification.
                           If None, the measured PE-sample QBER is used
                           directly (a point estimate, not a confidence
                           bound). This is only meaningful for symmetric
                           qubit channels; supply e_ph explicitly for other
                           protocols (e.g. from a WLC/SDP analysis).
    policy               : KeyLengthPolicy; defaults to AsymptoticPolicy.
    """
    policy = policy or AsymptoticPolicy()
    rng = np.random.default_rng(seed)
    ch = PublicChannel()

    a = np.asarray(alice_bits, dtype=np.uint8)
    b = np.asarray(bob_bits, dtype=np.uint8).copy()
    if len(a) != len(b):
        raise ValueError("strings must have equal length")
    n_sifted = len(a)

    alice = KeyHolder(a, ch)
    qber, keep, n_pe = sifting.estimate_qber(
        alice, b, fraction=pe_fraction, seed=int(rng.integers(2**31)), channel=ch)

    alice = KeyHolder(a[keep], ch)
    b = b[keep]
    n = len(b)

    leak_before = ch.leaked_bits
    stats = cascade.reconcile(alice, b, qber, n_passes=n_passes, schedule=schedule,
                              seed=int(rng.integers(2**31)), channel=ch)
    leak_ec = ch.leaked_bits - leak_before

    ok, leak_ev = extract.verify(alice, b, tag_len=tag_len,
                                 seed=int(rng.integers(2**31)), channel=ch)
    if not ok:
        raise RuntimeError("verification failed: strings still differ after Cascade")

    leak_total = leak_ec + leak_ev
    e_ph = qber if e_ph is None else e_ph
    final_len = policy.length(n=n, e_ph=e_ph, leak_total=leak_total, n_test=n_pe)
    key, _ = extract.amplify(b, final_len, seed=int(rng.integers(2**31)))

    qber_true = stats["corrections"] / n
    denom = n * _h2(qber_true) if qber_true > 0 else np.inf
    return Result(key=key, n_sifted=n_sifted, n_reconciled=n, qber=qber,
                  leak_ec=leak_ec, leak_ev=leak_ev, leak_total=leak_total,
                  corrections=stats["corrections"], qber_true=qber_true,
                  efficiency=leak_ec / denom if denom else float("nan"),
                  messages=ch.messages, final_len=final_len,
                  verified=ok, channel=ch)


def run_raw(raw, keep_basis=None, **kw):
    """Sift first, then post-process."""
    a, b, _ = sifting.sift(raw, keep_basis=keep_basis)
    return run(a, b, **kw)


def _h2(x):
    from .keylen import h2
    return h2(x)
