"""Minimal end-to-end demo."""
import qkdpp

# 1. no experimental data: simulate a full BB84 run and sift it
raw = qkdpp.generate_raw(300_000, qber=0.03, eta=0.2, p_z=0.7, seed=0)
r = qkdpp.run_raw(raw, keep_basis=0, seed=0)
print("from raw:   ", r.summary())
print("  leakage:  ", r.channel.breakdown)

# 2. already-sifted strings, which is the usual case for existing data
a, b = qkdpp.generate_sifted(16_384, qber=0.02, seed=1)
r = qkdpp.run(a, b, seed=1)
print("from sifted:", r.summary())

# 3. key length taken from an external security analysis instead of the default
r = qkdpp.run(a, b, policy=qkdpp.ExternalPolicy(rate_per_signal=0.45), seed=1)
print("external:   ", r.summary())
