"""Training loops and training-time utilities.

Stages are trained sequentially, each freezing the previous one:

1. ``train_sr1``   — super-resolution backbone
2. ``train_zhead`` — redshift head, over a swappable input representation
3. ``train_sr2``   — physics-informed residual refiner

The redshift comparison presented in the paper (LR vs SR vs HR) is produced by
running ``train_zhead`` four times with different
:mod:`~specsr.training.zhead_sources`, holding architecture, loss, optimiser and
data split fixed. See :mod:`specsr.training.zhead_sources` for why that is now
one script rather than four.
"""

from __future__ import annotations

from .losses import sr1_deblend_loss, sr2_loss
from .zhead_sources import SOURCE_NAMES, build_source
from .ztransform import RedshiftTransform

__all__ = [
    "SOURCE_NAMES",
    "build_source",
    "RedshiftTransform",
    "sr1_deblend_loss",
    "sr2_loss",
]
