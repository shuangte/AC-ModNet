"""Compute logP, QED and SAscore for each molecule and assign the paper's classes.

    python -m acmodnet.label --data data/zinc250k.txt.gz --out data/zinc250k_labeled.csv
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem

from . import properties as P
from .io import read_smiles


def label_one(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    row = {"smiles": smiles}
    for prop in P.PROPERTIES:
        value = P.compute(mol, prop)
        row[prop] = round(value, 5)
        row[f"{prop}_class"] = P.to_class(value, prop)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ncpu", type=int, default=4)
    a = p.parse_args(argv)

    smiles = read_smiles(a.data)
    with Pool(a.ncpu) as pool:
        rows = [r for r in pool.map(label_one, smiles, chunksize=512) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"Labeled {len(df)}/{len(smiles)} molecules -> {a.out}")
    for prop in P.PROPERTIES:
        counts = df[f"{prop}_class"].value_counts().sort_index().to_dict()
        print(f"  {prop} class counts: {counts}")


if __name__ == "__main__":
    main()
