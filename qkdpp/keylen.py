"""Secure key length policies.

This is the module that connects post-processing to the security proof. The
algorithms in this package are agnostic to it: they correct and extract, and
the policy decides how many bits may be kept.

AsymptoticPolicy is a sanity-check default, not a security claim.
FiniteKeyPolicy implements a textbook qubit-BB84 bound and must be checked
against your own analysis before any real claim.
ExternalPolicy is the intended production path: feed in the length coming from
the group's numerical optimisation (WLC / min-entropy bound).
"""

from dataclasses import dataclass
import numpy as np


def h2(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def qber_upper_bound(qber, n, n_test=None, eps_pe=1e-10):
    """One-sided confidence bound on the true error rate given a sample.

    A random sample can read low by chance (e.g. 0 errors in 100 bits at a
    true 1-2% rate); feeding the raw sample estimate straight into a key-
    length formula lets that luck inflate the claimed secure key. This
    inflates `qber` by the random-sampling confidence term of Tomamichel,
    Lim, Gisin & Renner, Nat. Commun. 3:634 (2012) -- the same term used
    inside FiniteKeyPolicy -- so any caller gets a defensible bound instead
    of a point estimate, without committing to the full finite-key formula.
    """
    n_test = n_test if n_test else n
    gamma = np.sqrt(np.log(2 / eps_pe) * (n + n_test)
                    / (2 * n * n_test * max(n_test - 1, 1)) * n_test)
    return float(min(0.5, qber + gamma))


class KeyLengthPolicy:
    def length(self, **ctx):
        raise NotImplementedError


@dataclass
class AsymptoticPolicy(KeyLengthPolicy):
    """l = n[1 - h(e_ph)] - leak_total. Devetak-Winter, no finite-size terms."""
    eps_pa: float = 1e-10

    def length(self, n, e_ph, leak_total, **_):
        pa_cost = 2 * np.log2(1 / (2 * self.eps_pa))
        return max(0, int(np.floor(n * (1 - h2(e_ph)) - leak_total - pa_cost)))


@dataclass
class FiniteKeyPolicy(KeyLengthPolicy):
    """Structure of Tomamichel, Lim, Gisin & Renner, Nat. Commun. 3:634 (2012).

    e_ph is inflated by a random-sampling confidence interval before the
    entropy is evaluated. UNVERIFIED against the original derivation: treat as
    a template, not as a proof. For decoy-state WCP sources this is not
    applicable at all -- see Lim, Curty, Walenta, Xu & Zbinden, PRA 89, 022307.
    """
    eps_pa: float = 1e-10
    eps_pe: float = 1e-10
    eps_cor: float = 1e-12

    def length(self, n, e_ph, leak_total, n_test=None, **_):
        e_up = qber_upper_bound(e_ph, n, n_test, self.eps_pe)
        pa_cost = 2 * np.log2(1 / (2 * self.eps_pa))
        cor_cost = np.log2(2 / self.eps_cor)
        return max(0, int(np.floor(n * (1 - h2(e_up)) - leak_total
                                   - pa_cost - cor_cost)))


@dataclass
class ExternalPolicy(KeyLengthPolicy):
    """Wrap a length (or a callable) supplied by an external security analysis.

    rate_per_signal: secure bits per sifted signal, e.g. the output of the
    WLC SDP for the observed statistics. Leakage already accounted for by the
    proof should not be subtracted twice -- set subtract_leak=False in that case.
    """
    rate_per_signal: float = None
    fn: callable = None
    subtract_leak: bool = True
    eps_pa: float = 1e-10

    def length(self, n, leak_total, **ctx):
        if self.fn is not None:
            return int(self.fn(n=n, leak_total=leak_total, **ctx))
        base = n * self.rate_per_signal
        if self.subtract_leak:
            base -= leak_total + 2 * np.log2(1 / (2 * self.eps_pa))
        return max(0, int(np.floor(base)))
