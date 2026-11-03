"""Helpers for building and loading the Hier-VAE (vendored hgraph2graph)."""
from __future__ import annotations

import argparse

import torch

from hgraph import DEVICE, HierVAE, PairVocab, common_atom_vocab

from .io import torch_load


def add_vae_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Hier-VAE architecture (must match the checkpoint)")
    g.add_argument("--vocab", required=True)
    g.add_argument("--rnn_type", default="LSTM")
    g.add_argument("--hidden_size", type=int, default=250)
    g.add_argument("--embed_size", type=int, default=250)
    g.add_argument("--latent_size", type=int, default=32)
    g.add_argument("--depthT", type=int, default=15)
    g.add_argument("--depthG", type=int, default=15)
    g.add_argument("--diterT", type=int, default=1)
    g.add_argument("--diterG", type=int, default=3)
    g.add_argument("--dropout", type=float, default=0.0)


def load_vocab(path: str) -> PairVocab:
    with open(path) as f:
        pairs = [line.strip("\r\n ").split() for line in f if line.strip()]
    return PairVocab(pairs)


def build_vae(args: argparse.Namespace) -> HierVAE:
    vae_args = argparse.Namespace(**vars(args))
    vae_args.vocab = load_vocab(args.vocab) if isinstance(args.vocab, str) else args.vocab
    vae_args.atom_vocab = common_atom_vocab
    return HierVAE(vae_args).to(DEVICE)


def load_vae(path: str, args: argparse.Namespace) -> HierVAE:
    """Load a checkpoint saved as (state_dict, optimizer, step, beta) or a bare state_dict."""
    model = build_vae(args)
    ckpt = torch_load(path, map_location=DEVICE)
    model.load_state_dict(ckpt[0] if isinstance(ckpt, (tuple, list)) else ckpt)
    return model.eval()


@torch.no_grad()
def decode(model: HierVAE, vecs: torch.Tensor, max_decode_step: int = 200) -> list[str]:
    """Decode encoder-space vectors (hidden_size) to SMILES via the VAE's mean head."""
    root_vecs, _ = model.rsample(vecs, model.R_mean, model.R_var, perturb=False)
    return model.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True,
                                max_decode_step=max_decode_step)


def decode_safe(model: HierVAE, vecs: torch.Tensor, max_decode_step: int = 200) -> list[str | None]:
    """Like decode(), but a molecule that crashes the decoder becomes None instead of
    losing the whole batch (the hgraph decoder has fixed-size node buffers)."""
    try:
        return decode(model, vecs, max_decode_step)
    except Exception:
        out = []
        for v in vecs:
            try:
                out.append(decode(model, v[None], max_decode_step)[0])
            except Exception:
                out.append(None)
        return out
