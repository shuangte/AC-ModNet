"""End-to-end smoke test on 200 ZINC molecules with tiny models (CPU, ~1-2 min).

It checks that every stage runs and hands the right files to the next one;
the tiny models are far too small to produce meaningful molecules.
"""
import gzip
import itertools
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

os.environ.setdefault("ACMODNET_DEVICE", "cpu")

from acmodnet import encode, evaluate, generate, get_vocab, label, preprocess, train_gan, train_vae  # noqa: E402
from acmodnet import properties as P  # noqa: E402
from acmodnet.gan import Discriminator, GANConfig, Generator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL_VAE = ["--hidden_size", "32", "--embed_size", "32", "--latent_size", "8", "--depthT", "3", "--depthG", "3"]


def test_property_bins():
    assert P.to_class(-0.5, "logp") == 0 and P.to_class(0.0, "logp") == 0 and P.to_class(0.01, "logp") == 1
    assert P.to_class(4.5, "logp") == 5
    assert P.to_class(0.5, "qed") == 0 and P.to_class(0.95, "qed") == 5
    assert P.to_class(1.5, "sas") == 0 and P.to_class(6.0, "sas") == 6
    assert [P.n_classes(p) for p in P.PROPERTIES] == [6, 6, 7]


def test_gan_shapes():
    cfg = GANConfig(n_classes=6, latent_z=250)
    G, D = Generator(cfg), Discriminator(cfg)
    vecs = G(torch.randn(4, cfg.latent_dim), torch.tensor([0, 1, 2, 5]))
    adv, cls = D(vecs)
    assert vecs.shape == (4, 250) and adv.shape == (4,) and cls.shape == (4, 6)


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    with gzip.open(ROOT / "data" / "zinc250k.txt.gz", "rt") as f:
        (tmp / "mini.txt").write_text("".join(itertools.islice(f, 200)))
    return tmp


def test_full_pipeline(mini):
    t = str(mini)
    get_vocab.main(["--data", f"{t}/mini.txt", "--out", f"{t}/vocab.txt", "--ncpu", "2"])
    label.main(["--data", f"{t}/mini.txt", "--out", f"{t}/labeled.csv", "--ncpu", "2"])
    preprocess.main(["--data", f"{t}/mini.txt", "--vocab", f"{t}/vocab.txt", "--out", f"{t}/tensors",
                     "--batch_size", "16", "--ncpu", "2"])
    train_vae.main(["--train", f"{t}/tensors", "--vocab", f"{t}/vocab.txt", "--save_dir", f"{t}/vae",
                    "--epoch", "1", *SMALL_VAE])
    vae = f"{t}/vae/model.ckpt.final"
    encode.main(["--data", f"{t}/labeled.csv", "--property", "qed", "--vocab", f"{t}/vocab.txt",
                 "--model", vae, "--out", f"{t}/latent.pt", *SMALL_VAE])
    latent = torch.load(f"{t}/latent.pt")
    assert latent["vecs"].shape == (200, 32) and latent["n_classes"] == 6

    train_gan.main(["--data", f"{t}/latent.pt", "--save_dir", f"{t}/gan", "--n_epochs", "2",
                    "--batch_size", "16", "--img_size", "16", "--channels", "2", "--deep_dim", "32",
                    "--val_fraction", "0.1"])
    generate.main(["--gan", f"{t}/gan/gan_last.pt", "--model", vae, "--vocab", f"{t}/vocab.txt",
                   "--property", "qed", "--n_per_class", "4", "--max_decode_step", "20",
                   "--out", f"{t}/generated.csv", *SMALL_VAE])
    gen = pd.read_csv(f"{t}/generated.csv")
    assert len(gen) == 24 and set(gen["target_class"]) == set(range(6))

    table = evaluate.evaluate(gen, (mini / "mini.txt").read_text().split())
    assert list(table["class"])[-1] == "mean"
    assert table[["validity", "uniqueness", "novelty", "accuracy"]].apply(lambda c: c.between(0, 1)).all().all()
