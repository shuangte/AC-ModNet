"""AC-GAN that generates Hier-VAE latent vectors conditioned on a property class.

Both networks work on a vector of length ``latent_z`` (the VAE's hidden size),
but reshape it to a ``channels x img_size x img_size`` feature map so that
convolutional blocks can be used: the generator ends with a linear projection
from the map to the vector, the discriminator starts with one from the vector
to the map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .io import torch_load


@dataclass
class GANConfig:
    n_classes: int = 6
    latent_dim: int = 100   # noise dimension
    latent_z: int = 250     # length of the molecular latent vector (VAE hidden size)
    deep_dim: int = 512
    img_size: int = 64
    channels: int = 6

    def to_dict(self):
        return asdict(self)


def init_weights(m: nn.Module) -> None:
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm2d" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.init_size = cfg.img_size // 4
        self.label_emb = nn.Embedding(cfg.n_classes, cfg.latent_dim)
        self.l1 = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.deep_dim),
            nn.Linear(cfg.deep_dim, 128 * self.init_size ** 2),
            nn.Tanh(),
        )
        # BatchNorm2d(c, 0.8) sets eps=0.8, as in the original training code.
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, cfg.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(cfg.channels * cfg.img_size ** 2, cfg.deep_dim),
            nn.Linear(cfg.deep_dim, cfg.latent_z),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.l1(self.label_emb(labels) * noise)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return self.l2(x.flatten(1))


class Discriminator(nn.Module):
    """Returns (real/fake logit, class logits)."""

    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.channels, self.img_size = cfg.channels, cfg.img_size

        def block(c_in, c_out, bn=True):
            layers = [nn.Conv2d(c_in, c_out, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                layers.append(nn.BatchNorm2d(c_out, 0.8))
            return layers

        self.l3 = nn.Sequential(
            nn.Linear(cfg.latent_z, cfg.deep_dim),
            nn.Linear(cfg.deep_dim, cfg.channels * cfg.img_size ** 2),
        )
        self.conv_blocks = nn.Sequential(
            *block(cfg.channels, 16, bn=False), *block(16, 32), *block(32, 64), *block(64, 128),
        )
        flat = 128 * (cfg.img_size // 16) ** 2
        self.adv_layer = nn.Sequential(nn.Linear(flat, 128), nn.Linear(128, 1))
        self.aux_layer = nn.Sequential(nn.Linear(flat, 64 * cfg.n_classes),
                                       nn.Linear(64 * cfg.n_classes, cfg.n_classes))

    def forward(self, vecs: torch.Tensor):
        x = self.l3(vecs).view(vecs.size(0), self.channels, self.img_size, self.img_size)
        x = self.conv_blocks(x).flatten(1)
        return self.adv_layer(x).squeeze(1), self.aux_layer(x)


def load_gan(path: str, device, cfg: GANConfig | None = None):
    """Load a checkpoint from acmodnet.train_gan, or a legacy (G, D, optD) tuple with ``cfg`` given."""
    ckpt = torch_load(path, map_location=device)
    if isinstance(ckpt, dict):
        cfg = GANConfig(**ckpt["config"])
        g_state, d_state = ckpt["generator"], ckpt["discriminator"]
    else:
        if cfg is None:
            raise ValueError("legacy checkpoint: pass the GAN architecture arguments")
        g_state, d_state = ckpt[0], ckpt[1]
    G, D = Generator(cfg).to(device), Discriminator(cfg).to(device)
    G.load_state_dict(g_state)
    D.load_state_dict(d_state)
    return G.eval(), D.eval(), cfg
