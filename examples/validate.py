"""Validation of the qkdpp pipeline: correctness, leakage accounting, randomness."""

import numpy as np
import qkdpp
from qkdpp.channel import PublicChannel, KeyHolder
from qkdpp import cascade, extract, keylen


def test_reconciliation_correctness(n=20000, trials=5):
    print("== reconciliation correctness and efficiency ==")
    for q in [0.005, 0.01, 0.02, 0.04, 0.08]:
        fs, oks = [], []
        for t in range(trials):
            a, b = qkdpp.generate_sifted(n, qber=q, seed=100 * t + int(q * 1000))
            ch = PublicChannel()
            alice = KeyHolder(a, ch)
            bb = b.copy()
            cascade.reconcile(alice, bb, q, seed=t, channel=ch)
            oks.append(np.array_equal(a, bb))
            fs.append(ch.leaked_bits / (n * keylen.h2(q)))
        print(f"  QBER={q:5.3f}  all_correct={all(oks)}  f_ec={np.mean(fs):.3f}")


def test_schedules(n=20000, q=0.02):
    print("== schedule comparison ==")
    for sch in ["original", "optimized"]:
        a, b = qkdpp.generate_sifted(n, qber=q, seed=42)
        ch = PublicChannel()
        alice = KeyHolder(a, ch)
        bb = b.copy()
        cascade.reconcile(alice, bb, q, schedule=sch, seed=1, channel=ch)
        f = ch.leaked_bits / (n * keylen.h2(q))
        print(f"  {sch:10s} f_ec={f:.3f}  msgs={ch.messages}  ok={np.array_equal(a, bb)}")


def test_verification_catches_failure(n=4000):
    print("== verification catches an uncorrected error ==")
    a, b = qkdpp.generate_sifted(n, qber=0.02, seed=5)
    ch = PublicChannel()
    alice = KeyHolder(a, ch)
    corrupted = a.copy()
    corrupted[123] ^= 1
    ok, _ = extract.verify(alice, corrupted, tag_len=64, seed=3)
    print(f"  detected mismatch = {not ok}")
    ok2, _ = extract.verify(alice, a.copy(), tag_len=64, seed=3)
    print(f"  accepted identical = {ok2}")


def test_toeplitz_linearity(n=2000, m=800):
    print("== Toeplitz extractor sanity ==")
    rng = np.random.default_rng(0)
    s = extract.toeplitz_seed(n, m, rng)
    x = rng.integers(0, 2, n, dtype=np.uint8)
    y = rng.integers(0, 2, n, dtype=np.uint8)
    lin = np.array_equal(
        extract.toeplitz_apply(s, x ^ y, m),
        extract.toeplitz_apply(s, x, m) ^ extract.toeplitz_apply(s, y, m))
    out = extract.toeplitz_apply(s, x, m)
    bias = abs(out.mean() - 0.5)
    print(f"  GF(2) linear = {lin}   output bias = {bias:.4f}")


def test_output_statistics(n=50000):
    print("== extracted key statistics ==")
    a, b = qkdpp.generate_sifted(n, qber=0.02, seed=11)
    r = qkdpp.run(a, b, seed=2)
    k = r.key
    ones = k.mean()
    # serial correlation
    corr = np.corrcoef(k[:-1], k[1:])[0, 1]
    # monobit chi-square
    chi = (k.sum() - len(k) / 2) ** 2 / (len(k) / 4)
    print(f"  len={len(k)}  ones={ones:.4f}  lag1_corr={corr:+.4f}  chi2={chi:.2f}")


def test_sifting_path(n_pulses=200000):
    print("== sifting from raw data ==")
    raw = qkdpp.generate_raw(n_pulses, qber=0.03, eta=0.3, p_z=0.7, seed=9)
    a, b, mask = qkdpp.sift(raw)
    expected = raw.detected & (raw.alice_bases == raw.bob_bases)
    print(f"  sifted={len(a)}  expected={expected.sum()}  "
          f"measured_qber={qkdpp.true_qber(a, b):.4f}")
    r = qkdpp.run_raw(raw, seed=4)
    print(f"  {r.summary()}")


def test_policies(n=20000):
    print("== key length policies ==")
    a, b = qkdpp.generate_sifted(n, qber=0.02, seed=13)
    for name, pol in [
        ("asymptotic", keylen.AsymptoticPolicy()),
        ("finite-key", keylen.FiniteKeyPolicy()),
        ("external r=0.4", keylen.ExternalPolicy(rate_per_signal=0.4)),
    ]:
        r = qkdpp.run(a, b, policy=pol, seed=6)
        print(f"  {name:16s} final={r.final_len:6d}  "
              f"rate={r.final_len / r.n_sifted:.4f}")


def test_leakage_is_complete(n=8000):
    print("== leakage accounting is exhaustive ==")
    a, b = qkdpp.generate_sifted(n, qber=0.03, seed=21)
    r = qkdpp.run(a, b, seed=8)
    total = sum(r.channel.breakdown.values())
    print(f"  breakdown sums to counter: {total == r.channel.leaked_bits}")
    print(f"  {r.channel.breakdown}")


if __name__ == "__main__":
    test_reconciliation_correctness()
    test_schedules()
    test_verification_catches_failure()
    test_toeplitz_linearity()
    test_output_statistics()
    test_sifting_path()
    test_policies()
    test_leakage_is_complete()
