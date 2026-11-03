"""Pre-train the Hier-VAE (adapted from hgraph2graph's train_generator.py).

    python -m acmodnet.train_vae --train data/vae_tensors --vocab data/vocab.txt \
        --save_dir ckpt/vae --epoch 20
"""
from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
from rdkit import RDLogger
from torch import nn, optim
from tqdm.auto import tqdm

from hgraph import DataFolder

from .io import torch_load
from .vae import add_vae_args, build_vae


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True,
                   help="folder written by acmodnet.preprocess (batch size is set there)")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--load_model", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip_norm", type=float, default=5.0)
    p.add_argument("--step_beta", type=float, default=0.001)
    p.add_argument("--max_beta", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--kl_anneal_iter", type=int, default=2000)
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--anneal_rate", type=float, default=0.9)
    p.add_argument("--anneal_iter", type=int, default=25000)
    p.add_argument("--print_iter", type=int, default=50)
    p.add_argument("--save_iter", type=int, default=5000)
    add_vae_args(p)
    args = p.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    model = build_vae(args)
    print(f"Hier-VAE: {sum(x.nelement() for x in model.parameters()) / 1000:.0f}K params")
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    total_step = beta = 0
    if args.load_model:
        state, opt_state, total_step, beta = torch_load(args.load_model)
        model.load_state_dict(state)
        optimizer.load_state_dict(opt_state)

    meters = np.zeros(6)
    for _ in range(args.epoch):
        for batch in tqdm(DataFolder(args.train, batch_size=None)):
            total_step += 1
            model.zero_grad()
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            meters += np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])

            if total_step % args.print_iter == 0:
                m = meters / args.print_iter
                print(f"[{total_step}] beta {beta:.3f} | KL {m[0]:.2f} | loss {m[1]:.3f} | "
                      f"word {m[2]:.2f} {m[3]:.2f} | topo {m[4]:.2f} | assm {m[5]:.2f}")
                meters[:] = 0
            if total_step % args.save_iter == 0:
                ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
                torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))
            if total_step % args.anneal_iter == 0:
                scheduler.step()
            if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
                beta = min(args.max_beta, beta + args.step_beta)

    torch.save((model.state_dict(), optimizer.state_dict(), total_step, beta),
               os.path.join(args.save_dir, "model.ckpt.final"))
    print(f"done: {total_step} steps -> {args.save_dir}/model.ckpt.final")


if __name__ == "__main__":
    main()
