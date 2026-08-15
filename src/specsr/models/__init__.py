"""Model architectures for the three-stage super-resolution pipeline.

- :class:`~specsr.models.sr1.SuperRes1D` — super-resolution backbone
- :class:`~specsr.models.zhead.ZHead1D` — redshift inference
- :class:`~specsr.models.sr2.SR2Attention` — physics-informed residual refiner

Stages are trained sequentially, each consuming the frozen checkpoint of the
previous one.
"""

from __future__ import annotations

from .blocks import ResidualBlock1D, build_param_groups, get_activation, highpass, smooth1d
from .lines import LINE_LIST_REST_AA, LINE_NAMES, LINE_WAVELENGTHS_AA, line_wavelengths_um
from .sr1 import SuperRes1D
from .sr2 import SR2Attention, build_line_mask, constrain_delta, gaussian_line_mask
from .zhead import ZHead1D, heteroscedastic_nll

__all__ = [
    "ResidualBlock1D",
    "build_param_groups",
    "get_activation",
    "highpass",
    "smooth1d",
    "LINE_LIST_REST_AA",
    "LINE_NAMES",
    "LINE_WAVELENGTHS_AA",
    "line_wavelengths_um",
    "SuperRes1D",
    "SR2Attention",
    "build_line_mask",
    "constrain_delta",
    "gaussian_line_mask",
    "ZHead1D",
    "heteroscedastic_nll",
]
