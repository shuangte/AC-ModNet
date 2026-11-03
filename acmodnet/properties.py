"""Molecular properties and the class bins used in the AC-ModNet paper.

Each property is split into left-open intervals (lo, hi]; the first interval
also contains its lower edge. Class ids start at 0.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, Descriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

INF = math.inf
BINS = {
    "logp": [-INF, 0.0, 1.0, 2.0, 3.0, 4.0, INF],
    "qed": [-INF, 0.5, 0.6, 0.7, 0.8, 0.9, INF],
    "sas": [-INF, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, INF],
}
PROPERTIES = tuple(BINS)


def n_classes(prop: str) -> int:
    return len(BINS[prop]) - 1


def compute(mol: Chem.Mol, prop: str) -> float:
    if prop == "logp":
        return Descriptors.MolLogP(mol)
    if prop == "qed":
        return QED.qed(mol)
    if prop == "sas":
        return sascorer.calculateScore(mol)
    raise ValueError(f"unknown property {prop!r}; choose from {PROPERTIES}")


def to_class(value: float, prop: str) -> int:
    inner = BINS[prop][1:-1]
    return int(np.searchsorted(inner, value, side="left"))


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None
