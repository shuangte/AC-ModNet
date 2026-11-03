"""Validity, uniqueness, novelty and class accuracy of generated molecules.

    python -m acmodnet.evaluate --generated results/logp_generated.csv \
        --train data/zinc250k_labeled.csv --out results/logp_metrics.csv

Per target class, plus the mean over classes:
* validity    = valid SMILES / generated
* uniqueness  = unique canonical SMILES / valid
* novelty     = unique molecules not in the training set / unique
* accuracy    = unique molecules whose computed class equals the target / unique
* disc_agree  = unique molecules the discriminator assigns to the target class / unique
"""
from __future__ import annotations

import argparse

import pandas as pd

from .io import read_smiles
from .properties import canonical


def metrics(group: pd.DataFrame, train: set[str]) -> dict:
    n = len(group)
    valid = group[group["canonical"].notna()]
    unique = valid.drop_duplicates("canonical")
    nu = max(len(unique), 1)
    return {
        "generated": n,
        "validity": len(valid) / n if n else 0.0,
        "uniqueness": len(unique) / max(len(valid), 1),
        "novelty": (~unique["canonical"].isin(train)).sum() / nu,
        "accuracy": (unique["property_class"] == unique["target_class"]).sum() / nu,
        "disc_agree": (unique["disc_class"] == unique["target_class"]).sum() / nu,
    }


def evaluate(generated: pd.DataFrame, train_smiles: list[str]) -> pd.DataFrame:
    train = {c for c in map(canonical, train_smiles) if c}
    generated = generated.copy()
    generated["canonical"] = generated["smiles"].map(lambda s: canonical(s) if isinstance(s, str) else None)
    rows = [{"class": c, **metrics(g, train)} for c, g in generated.groupby("target_class")]
    table = pd.DataFrame(rows)
    mean = table.drop(columns="class").mean().to_dict()
    mean["generated"] = int(table["generated"].sum())
    return pd.concat([table, pd.DataFrame([{"class": "mean", **mean}])], ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", required=True)
    p.add_argument("--train", required=True, help="training SMILES (.txt/.csv, may be gzipped)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    table = evaluate(pd.read_csv(a.generated), read_smiles(a.train))
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(table.to_string(index=False))
    if a.out:
        table.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
