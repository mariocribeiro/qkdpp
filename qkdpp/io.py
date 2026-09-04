"""Loading and saving bit strings.

Accepted formats, chosen by extension:
  .npy   numpy array of 0/1
  .txt   whitespace-separated or a single run of '0'/'1' characters
  .csv   one column of 0/1, optional header
  .bin   packed bits, 8 per byte, MSB first (needs n_bits to strip padding)
"""

from pathlib import Path
import numpy as np


def load_bits(path, n_bits=None):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".bin":
        arr = np.unpackbits(np.fromfile(path, dtype=np.uint8))
    elif ext in (".txt", ".dat", ".csv"):
        text = path.read_text().strip()
        if ext == ".csv" or any(c in text for c in ",\n"):
            rows = [r.strip() for r in text.replace(",", "\n").splitlines() if r.strip()]
            rows = [r for r in rows if r[0] in "01"]
            arr = np.array([int(r) for r in rows], dtype=np.uint8)
        elif " " in text:
            arr = np.array(text.split(), dtype=np.uint8)
        else:
            arr = np.frombuffer(text.encode(), dtype=np.uint8) - ord("0")
    else:
        raise ValueError(f"unsupported extension {ext!r}")

    arr = np.asarray(arr, dtype=np.uint8).ravel()
    if n_bits is not None:
        arr = arr[:n_bits]
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{path} contains values other than 0 and 1")
    return arr


def save_bits(path, bits, packed=False):
    path = Path(path)
    bits = np.asarray(bits, dtype=np.uint8)
    if path.suffix.lower() == ".npy":
        np.save(path, bits)
    elif packed or path.suffix.lower() == ".bin":
        np.packbits(bits).tofile(path)
    else:
        path.write_text("".join(map(str, bits.tolist())))
    return path


def load_pair(alice_path, bob_path, n_bits=None):
    a = load_bits(alice_path, n_bits)
    b = load_bits(bob_path, n_bits)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return a, b
