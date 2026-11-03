"""Tensorize molecules into graph batches for Hier-VAE training.

    python -m acmodnet.preprocess --data data/zinc250k.txt.gz --vocab data/vocab.txt \
        --out data/vae_tensors --ncpu 8
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool

import torch
from rdkit import RDLogger

from hgraph import MolGraph, PairVocab, common_atom_vocab

from .io import read_smiles


def tensorize(batch, vocab):
    graphs, (tree, graph), orders = MolGraph.tensorize(batch, vocab, common_atom_vocab)
    to_np = lambda x: x.numpy() if isinstance(x, torch.Tensor) else x  # noqa: E731
    return graphs, ([to_np(x) for x in tree], [to_np(x) for x in graph]), orders


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batches_per_file", type=int, default=1000)
    p.add_argument("--ncpu", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")

    with open(a.vocab) as f:
        vocab = PairVocab([line.split() for line in f if line.strip()], cuda=False)
    smiles = read_smiles(a.data)
    random.Random(a.seed).shuffle(smiles)
    batches = [smiles[i:i + a.batch_size] for i in range(0, len(smiles), a.batch_size)]

    with Pool(a.ncpu) as pool:
        data = pool.map(partial(tensorize, vocab=vocab), batches)

    os.makedirs(a.out, exist_ok=True)
    n = a.batches_per_file
    for k, i in enumerate(range(0, len(data), n)):
        with open(os.path.join(a.out, f"tensors-{k}.pkl"), "wb") as f:
            pickle.dump(data[i:i + n], f, pickle.HIGHEST_PROTOCOL)
    print(f"{len(smiles)} molecules -> {len(data)} batches in {a.out}")


if __name__ == "__main__":
    main()
