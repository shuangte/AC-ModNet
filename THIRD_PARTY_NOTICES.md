# Third-party code

## hgraph2graph (`hgraph/`)

Source: https://github.com/wengong-jin/hgraph2graph
License: MIT, Copyright (c) 2020 Wengong Jin, Regina Barzilay, Tommi Jaakkola (see `hgraph/LICENSE`).

Changes made in this repository:
* `hgraph/device.py` added; all hard-coded `.cuda()` calls replaced with `.to(DEVICE)` so the
  code runs on CPU or any GPU.
* `nn.*Loss(size_average=False)` replaced with the equivalent `reduction="sum"` (the old
  argument is deprecated in current PyTorch).
* `chemutils.get_clique_mol` falls back to a kekulized copy when newer RDKit versions fail to
  write a fragment SMILES.
