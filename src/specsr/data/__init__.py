"""Dataset construction, loading and leak-free splitting.

The preprocessing pipeline turns raw JADES products into paired ``.npz``
datasets; :mod:`~specsr.data.datasets` loads them, and
:mod:`~specsr.data.splits` splits them over parent galaxies rather than rows.
"""

from __future__ import annotations

from .augment import AugmentationConfig, augment_pair
from .build import BuildConfig, build_dataset
from .datasets import FixedGridSpectraDataset, PairedSpectra, normalize_spectrum
from .grid import DEFAULT_GRID, LogWavelengthGrid, resample_flux_conserving
from .ingest import discover_spectra, group_by_target, read_spectrum
from .splits import get_or_make_split, make_group_split, parent_group_ids
from .stitch import StitchedSpectrum, stitch_gratings

__all__ = [
    "BuildConfig",
    "build_dataset",
    "AugmentationConfig",
    "augment_pair",
    "DEFAULT_GRID",
    "LogWavelengthGrid",
    "resample_flux_conserving",
    "discover_spectra",
    "group_by_target",
    "read_spectrum",
    "StitchedSpectrum",
    "stitch_gratings",
    "FixedGridSpectraDataset",
    "PairedSpectra",
    "normalize_spectrum",
    "get_or_make_split",
    "make_group_split",
    "parent_group_ids",
]
