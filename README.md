# qkdpp

A small, dependency-light Python package for the classical post-processing
stage of a QKD experiment: **sifting → parameter estimation → error
correction → verification → privacy amplification**.

It exists to turn a pair of correlated bit strings from an experiment (or a
simulation) into a final key, with every bit that crosses the public channel
tracked explicitly — so the leakage accounting is correct by construction,
not by convention.

## Design philosophy

This is a **simple, protocol-agnostic post-processing tool**, not a
security-proof reference implementation. It makes one significant
simplification on purpose: the error rate used to size privacy amplification
is the *raw* estimate from the public sample, not a statistically
conservative confidence bound. That mirrors how lab pipelines (e.g.
[AIT-QKD](https://github.com/axdhill/ait-qkd)) typically work, and keeps
the tool easy to reason about for day-to-day experimental use.

If you need a defensible security margin instead of a point estimate (e.g.
for a real deployed key), `qkdpp.keylen.qber_upper_bound(...)` implements the
random-sampling confidence bound of Tomamichel, Lim, Gisin & Renner, *Nat.
Commun.* 3:634 (2012) — pass its output as `e_ph` to `qkdpp.run(...)`. For an
actual security *proof* tied to a specific protocol (finite-key effects,
phase-error estimation, decoy states, ...), feed the secure rate from your
own analysis in through `ExternalPolicy` instead of relying on either of the
built-in policies.

## Install

```bash
git clone https://github.com/<your-username>/qkdpp.git
cd qkdpp
pip install -e .
```

Only dependency: `numpy`. Running the notebook additionally needs `pandas`
and `jupyterlab` (`pip install -r requirements-notebook.txt`).

## Quick start

```python
import qkdpp

# already-sifted bit pair from an experiment
alice_bits = qkdpp.io.load_bits("alice_sifted.txt")
bob_bits   = qkdpp.io.load_bits("bob_sifted.txt")

result = qkdpp.run(alice_bits, bob_bits)
print(result.summary())
# sifted=9970  qber=0.0241(0.0196 true)  leak_ec=1450  f_ec=1.160  messages=188  final=5925 bits

qkdpp.io.save_bits("final_key.txt", result.key)
```

Starting from raw (unsifted) prepare-and-measure data instead:

```python
raw = qkdpp.generate_raw(300_000, qber=0.03, eta=0.2, p_z=0.7, seed=0)  # or your own RawData
result = qkdpp.run_raw(raw, keep_basis=0)
```

See `examples/` for more (simulated data, external key-length policy, CSV
from a real experiment), `notebooks/qkdpp_pipeline.ipynb` for a full worked
example on an experimental dataset, and `notebooks/qkdpp_vs_aitqkd.ipynb` for
a side-by-side comparison against
[AIT-QKD](https://github.com/axdhill/ait-qkd).

## Pipeline stages

| stage | module | what it does |
|---|---|---|
| sifting | `sifting.py` | keep positions with matching bases (+ detected, for raw prepare-and-measure data) |
| parameter estimation | `sifting.estimate_qber` | reveal a random public sample, measure the error rate, discard the sample |
| error correction | `cascade.py` | Cascade (Brassard–Salvail 1993), block-size schedule in the spirit of Martinez-Mateo et al. 2014 |
| verification | `extract.verify` | Toeplitz-hash check that the strings now agree |
| privacy amplification | `extract.amplify` | Toeplitz extraction down to the secure length |
| key length | `keylen.py` | `AsymptoticPolicy` (default), `FiniteKeyPolicy`, or `ExternalPolicy` |
| leakage accounting | `channel.py` | `PublicChannel`/`KeyHolder` — every public bit is metered, by construction |

## Known limitations

- **Not a cryptographic RNG.** Randomness throughout (including the
  Toeplitz seed used for privacy amplification) comes from
  `numpy.random.default_rng` (PCG64). That's fine for simulation,
  benchmarking, and processing experimental data for analysis — it is
  **not** appropriate for extracting a key you intend to actually use as a
  cryptographic secret. Swap in a CSPRNG (e.g. `os.urandom`-seeded) for that.
- `e_ph = qber` (the default) is only physically meaningful for a symmetric
  qubit channel. For MDI-QKD, CV-QKD, or anything where the phase-error rate
  isn't simply the measured bit-error rate, supply `e_ph` explicitly.
- `FiniteKeyPolicy` implements the structure of Tomamichel et al. 2012 but
  is an unverified template, not a checked reproduction of the paper's
  bound — treat it as a starting point, not a citation-ready result.
- Cascade's block-size schedule needs at least a couple of hundred bits per
  block to behave sensibly; very small blocks (a few hundred bits or fewer)
  can occasionally fail to converge if the public-sample QBER estimate reads
  far from the true rate. `qkdpp.run(...)` raises `RuntimeError` rather than
  returning a wrong key if this happens.

## License

MIT — see `LICENSE`.
