"""Generate molecules of a target property class with the trained AC-GAN + Hier-VAE.

    python -m acmodnet.generate --gan ckpt/gan_logp/gan_last.pt --model ckpt/vae/model.ckpt.final \
        --vocab data/vocab.txt --property logp --n_per_class 1000 --out results/logp_generated.csv

Each row: target class, the SMILES, its computed property value and class, and
the discriminator's predicted class and realness score.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from hgraph import DEVICE

from . import properties as P
from .gan import GANConfig, load_gan
from .vae import add_vae_args, decode_safe, load_vae


@torch.no_grad()
def generate_class(G, D, vae, cfg, target: int, n: int, batch_size: int, max_decode_step: int):
    rows = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        labels = torch.full((b,), target, dtype=torch.long, device=DEVICE)
        vecs = G(torch.randn(b, cfg.latent_dim, device=DEVICE), labels)
        validity, cls = D(vecs)
        smiles = decode_safe(vae, vecs, max_decode_step)
        for s, v, c in zip(smiles, torch.sigmoid(validity).tolist(), cls.argmax(1).tolist()):
            rows.append({"target_class": target, "smiles": s, "disc_class": c, "disc_real": round(v, 4)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gan", required=True)
    p.add_argument("--model", required=True, help="Hier-VAE checkpoint")
    p.add_argument("--property", choices=P.PROPERTIES, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--classes", type=int, nargs="*", default=None, help="default: all classes")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_decode_step", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    g = p.add_argument_group("only for legacy GAN checkpoints saved as (G, D, optD) tuples")
    for name, default in GANConfig().to_dict().items():
        g.add_argument(f"--{name}", type=int, default=default)
    add_vae_args(p)
    args = p.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)

    legacy_cfg = GANConfig(**{k: getattr(args, k) for k in GANConfig().to_dict()})
    G, D, cfg = load_gan(args.gan, DEVICE, legacy_cfg)
    vae = load_vae(args.model, args)

    rows = []
    for c in tqdm(args.classes if args.classes is not None else range(cfg.n_classes), desc="classes"):
        rows += generate_class(G, D, vae, cfg, c, args.n_per_class, args.batch_size, args.max_decode_step)

    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"]) if isinstance(r["smiles"], str) and r["smiles"] else None
        r["valid"] = mol is not None
        r[args.property] = round(P.compute(mol, args.property), 5) if mol else None
        r["property_class"] = P.to_class(r[args.property], args.property) if mol else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} molecules -> {args.out}")


if __name__ == "__main__":
    main()
