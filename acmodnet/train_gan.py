"""Train the property-conditioned AC-GAN on Hier-VAE latent vectors.

    python -m acmodnet.train_gan --data data/latent_logp.pt --save_dir ckpt/gan_logp --n_epochs 40

Training follows the paper:
* Discriminator loss: 0.5 * (adv_real + 2 * cls_real) + 0.5 * adv_fake.
* Generator loss: 0.475 * adv + 0.475 * cls + a small distance term to real vectors.
* Adaptive balance: the generator gets more updates per discriminator step while
  the discriminator wins easily, and stops training if the discriminator's fake
  loss has collapsed to ~0 for many steps in a row.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .gan import Discriminator, GANConfig, Generator, init_weights
from .io import torch_load


def generator_steps(d_fake_loss: float) -> int:
    """Number of generator updates per batch, given the last discriminator fake loss."""
    return 3 if d_fake_loss < 0.1 else 2


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def validate(G, D, loader, cfg, device) -> dict:
    G.eval(), D.eval()
    real_acc, fake_acc, n = 0.0, 0.0, 0
    for vecs, labels in loader:
        vecs, labels = vecs.to(device), labels.to(device)
        _, real_cls = D(vecs)
        z = torch.randn(len(labels), cfg.latent_dim, device=device)
        _, fake_cls = D(G(z, labels))
        real_acc += accuracy(real_cls, labels) * len(labels)
        fake_acc += accuracy(fake_cls, labels) * len(labels)
        n += len(labels)
    G.train(), D.train()
    return {"val_real_acc": real_acc / n, "val_fake_acc": fake_acc / n}


def run(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)

    data = torch_load(args.data)
    vecs, labels = data["vecs"].float(), data["labels"].long()
    cfg = GANConfig(n_classes=args.n_classes or int(data["n_classes"]), latent_dim=args.latent_dim,
                    latent_z=vecs.size(1), deep_dim=args.deep_dim, img_size=args.img_size,
                    channels=args.channels)

    dataset = TensorDataset(vecs, labels)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 4)
    if len(train_loader) == 0:
        raise ValueError("not enough data for one training batch")

    G, D = Generator(cfg).to(device), Discriminator(cfg).to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    adv_loss, cls_loss = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    log = open(os.path.join(args.save_dir, "train_log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "d_fake_loss", "d_real_loss", "d_real_acc", "g_loss", "g_fake_acc"])

    t0, step, d_fake_value, collapsed, train_g = time.time(), 0, 1.0, 0, True
    history = []
    for epoch in range(args.n_epochs):
        for real, y in train_loader:
            real, y = real.to(device), y.to(device)
            b = real.size(0)
            ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)

            # ---- adaptive balance between generator and discriminator ----
            if train_g and d_fake_value < 1e-5:
                collapsed += 1
                if collapsed > args.patience:
                    print("discriminator fake loss collapsed: generator training stopped")
                    train_g = False
            else:
                collapsed = 0
            g_steps = generator_steps(d_fake_value) if train_g else 0

            # ---- generator ----
            g_loss_sum, g_acc_sum = 0.0, 0.0
            z = torch.randn(b, cfg.latent_dim, device=device)
            gen_y = torch.randint(0, cfg.n_classes, (b,), device=device)
            fake = G(z, gen_y)
            for _ in range(g_steps):
                opt_G.zero_grad()
                z = torch.randn(b, cfg.latent_dim, device=device)
                gen_y = torch.randint(0, cfg.n_classes, (b,), device=device)
                fake = G(z, gen_y)
                validity, pred_cls = D(fake)
                distance = ((fake - real) ** 2).sum() / cfg.latent_z
                g_loss = (args.w_adv * adv_loss(validity, ones) + args.w_cls * cls_loss(pred_cls, gen_y)
                          + args.w_dist * distance)
                g_loss.backward()
                opt_G.step()
                g_loss_sum += g_loss.item()
                g_acc_sum += accuracy(pred_cls, gen_y)

            # ---- discriminator ----
            opt_D.zero_grad()
            fake_validity, _ = D(fake.detach())
            d_fake = adv_loss(fake_validity, zeros)
            real_validity, real_cls = D(real)
            d_real = adv_loss(real_validity, ones) + 2.0 * cls_loss(real_cls, y)
            d_loss = 0.5 * d_real + 0.5 * d_fake
            d_loss.backward()
            opt_D.step()
            d_fake_value = d_fake.item()

            d_acc = accuracy(real_cls, y)
            g_loss_avg = g_loss_sum / g_steps if g_steps else float("nan")
            g_acc_avg = g_acc_sum / g_steps if g_steps else float("nan")
            writer.writerow([epoch, step, round(d_fake_value, 6), round(d_real.item(), 6),
                             round(d_acc, 4), round(g_loss_avg, 6), round(g_acc_avg, 4)])
            if step % args.print_iter == 0:
                print(f"[epoch {epoch + 1}/{args.n_epochs} step {step}] D fake {d_fake_value:.4f} "
                      f"real {d_real.item():.4f} acc {d_acc:.0%} | G {g_loss_avg:.4f} acc {g_acc_avg:.0%}")
            step += 1

        metrics = validate(G, D, val_loader, cfg, device)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(f"epoch {epoch + 1}: val real-class acc {metrics['val_real_acc']:.1%}, "
              f"generated-class acc {metrics['val_fake_acc']:.1%}")
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(),
                    "config": cfg.to_dict(), "property": data.get("property"), "epoch": epoch + 1},
                   os.path.join(args.save_dir, "gan_last.pt"))

    log.close()
    print(f"done in {time.time() - t0:.0f}s -> {args.save_dir}/gan_last.pt")
    return {"history": history, "config": cfg}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help=".pt file from acmodnet.encode")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--n_epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--b1", type=float, default=0.5)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--n_classes", type=int, default=None, help="default: from the data file")
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--deep_dim", type=int, default=512)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--w_adv", type=float, default=0.475)
    p.add_argument("--w_cls", type=float, default=0.475)
    p.add_argument("--w_dist", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_fraction", type=float, default=0.01)
    p.add_argument("--print_iter", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
