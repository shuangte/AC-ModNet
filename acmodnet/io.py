"""Reading SMILES lists from .txt / .csv (optionally gzipped)."""
from __future__ import annotations

import pandas as pd


def read_smiles(path: str, column: str = "smiles") -> list[str]:
    """A .csv needs a ``smiles`` column; any other file is one SMILES per line."""
    if path.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)[column].astype(str).tolist()
    opener = __import__("gzip").open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [line.split()[0] for line in f if line.strip()]


def torch_load(path: str, map_location=None):
    """torch.load for trusted checkpoints, on both old (<1.13) and new PyTorch."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)
