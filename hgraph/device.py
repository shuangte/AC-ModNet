"""Device selection for the vendored hgraph code.

Upstream hgraph2graph hard-codes ``.cuda()``. Here every tensor goes to
``DEVICE`` instead, so the same code runs on GPU or CPU. Override with the
environment variable ``ACMODNET_DEVICE`` (e.g. ``cpu``, ``cuda:1``).
"""
import os

import torch

DEVICE = torch.device(os.environ.get("ACMODNET_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
