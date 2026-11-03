"""Encode labeled molecules into Hier-VAE latent vectors for AC-GAN training.

    python -m acmodnet.encode --data data/zinc250k_labeled.csv --property logp \
        --vocab data/vocab.txt --model ckpt/vae/model.ckpt.final --out data/latent_logp.pt

The output holds ``vecs`` (N x hidden_size, the encoder's root vectors before the
mean/variance heads) and ``labels`` (N, class ids for the chosen property).
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from hgraph import MolGraph, common_atom_vocab
from hgraph.hgnn import make_cuda

from . import properties as P
from .vae import add_vae_args, load_vae, load_vocab


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV from acmodnet.label")
    p.add_argument("--property", choices=P.PROPERTIES, required=True)
    p.add_argument("--model", required=True, help="Hier-VAE checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    add_vae_args(p)
    args = p.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")

    df = pd.read_csv(args.data)
    vocab = load_vocab(args.vocab)
    args.vocab = vocab
    model = load_vae(args.model, args)

    vecs, labels, skipped = [], [], 0
    smiles, classes = df["smiles"].tolist(), df[f"{args.property}_class"].tolist()
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), args.batch_size)):
            batch, batch_y = [], []
            for s, y in zip(smiles[i:i + args.batch_size], classes[i:i + args.batch_size]):
                try:  # molecules with motifs outside the vocabulary cannot be encoded
                    MolGraph.tensorize([s], vocab, common_atom_vocab)
                    batch.append(s)
                    batch_y.append(y)
                except Exception:
                    skipped += 1
            if not batch:
                continue
            _, tensors, _ = MolGraph.tensorize(batch, vocab, common_atom_vocab)
            root_vecs, _, _, _ = model.encoder(*make_cuda(tensors))
            vecs.append(root_vecs.cpu())
            labels += batch_y

    data = {"vecs": torch.cat(vecs), "labels": torch.tensor(labels, dtype=torch.long),
            "property": args.property, "n_classes": P.n_classes(args.property)}
    torch.save(data, args.out)
    print(f"Encoded {len(labels)} molecules ({skipped} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
