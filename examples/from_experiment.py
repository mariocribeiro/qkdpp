"""Post-process an already-sifted bit pair from a lab experiment.

Expects two plain bit files (see qkdpp.io.load_bits for supported formats:
.npy, .txt, .csv, .bin). Run from the repo root as:

    python examples/from_experiment.py alice_sifted.txt bob_sifted.txt
"""
import sys
import qkdpp

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)

    alice_bits = qkdpp.io.load_bits(sys.argv[1])
    bob_bits = qkdpp.io.load_bits(sys.argv[2])

    r = qkdpp.run(alice_bits, bob_bits)
    print(r.summary())
    print("leakage breakdown:", r.channel.breakdown)

    qkdpp.io.save_bits("final_key.txt", r.key)
    print("final key saved to final_key.txt")
