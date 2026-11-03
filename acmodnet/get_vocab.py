"""Extract the motif vocabulary used by Hier-VAE.

    python -m acmodnet.get_vocab --data data/zinc250k.txt.gz --out data/vocab.txt --ncpu 8
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

from rdkit import RDLogger

from hgraph import MolGraph

from .io import read_smiles


def motifs(smiles_list):
    vocab = set()
    for s in smiles_list:
        tree = MolGraph(s).mol_tree
        for _, attr in tree.nodes(data=True):
            vocab.add(attr["label"])
            for _, inter in attr["inter_label"]:
                vocab.add((attr["smiles"], inter))
    return vocab


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ncpu", type=int, default=4)
    a = p.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")

    smiles = sorted(set(read_smiles(a.data)))
    size = len(smiles) // a.ncpu + 1
    chunks = [smiles[i:i + size] for i in range(0, len(smiles), size)]
    with Pool(a.ncpu) as pool:
        vocab = set().union(*pool.map(motifs, chunks))
    with open(a.out, "w") as f:
        for x, y in sorted(vocab):
            f.write(f"{x} {y}\n")
    print(f"{len(vocab)} vocabulary entries from {len(smiles)} molecules -> {a.out}")


if __name__ == "__main__":
    main()
