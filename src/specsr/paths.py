"""Filesystem locations used by specsr.

Nothing here is hard-coded to one machine. Every location can be overridden by
an environment variable, which is what makes the package usable outside the
original development checkout:

``SPECSR_JADES_ROOT``
    Root of the *raw* JWST/JADES release tree (the directory containing
    ``DR3/``, ``DR4/``, ``DR5/``). Raw data is large and is deliberately kept
    outside the repository; only derived products are built here.

``SPECSR_DATA_DIR``
    Where derived products (paired/augmented ``.npz`` datasets) are written.
    Defaults to ``./data`` relative to the current working directory.

``SPECSR_CACHE_DIR``
    Scratch space for downloaded checkpoints and intermediate caches.
    Defaults to the platform cache directory.

``SPECSR_OUTPUT_DIR``
    Where anything a *user* might want to keep is written -- figures,
    predictions, evaluation tables. Defaults to ``./outputs`` relative to the
    current working directory.

The distinction between the last two is deliberate. A cache may be deleted at
any time without losing anything; an output is a result. Scattering results
across the working directory is how they get mixed up with the code and then
committed by accident, which is a large part of why this repository's git
history is twenty times the size of its contents.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "jades_root",
    "data_dir",
    "cache_dir",
    "output_dir",
    "release_dir",
    "require_jades_root",
]


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else None


def jades_root() -> Path | None:
    """Root of the raw JADES release tree, or ``None`` if unset."""
    return _env_path("SPECSR_JADES_ROOT")


def require_jades_root() -> Path:
    """Like :func:`jades_root` but raises with an actionable message."""
    root = jades_root()
    if root is None:
        raise RuntimeError(
            "Raw JADES data location is not configured. Set SPECSR_JADES_ROOT to "
            "the directory containing DR3/, DR4/, DR5/, e.g.\n"
            "    export SPECSR_JADES_ROOT=/path/to/JADES_data"
        )
    if not root.is_dir():
        raise RuntimeError(f"SPECSR_JADES_ROOT does not exist: {root}")
    return root


def release_dir(release: str = "DR4") -> Path:
    """Directory for a specific data release, e.g. ``release_dir('DR4')``."""
    root = require_jades_root()
    path = root / release
    if not path.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise RuntimeError(
            f"Release {release!r} not found under {root}. Available: {available}"
        )
    return path


def data_dir() -> Path:
    """Where derived datasets are written. Created on demand."""
    path = _env_path("SPECSR_DATA_DIR") or (Path.cwd() / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(subdir: str | None = None) -> Path:
    """Where the package writes results, created on demand.

    Everything a user might want to keep goes here: figures, prediction caches,
    evaluation tables. Resolved from ``SPECSR_OUTPUT_DIR``, defaulting to
    ``./outputs``.

    ``subdir`` names a category, e.g. ``output_dir("figures")``. Categories are
    created lazily, so an installation that never makes a figure never grows a
    ``figures/`` directory.

    Point ``SPECSR_OUTPUT_DIR`` somewhere outside the checkout to keep results
    entirely separate from the source tree.
    """
    path = _env_path("SPECSR_OUTPUT_DIR") or (Path.cwd() / "outputs")
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Scratch/cache directory. Created on demand."""
    path = _env_path("SPECSR_CACHE_DIR")
    if path is None:
        base = os.environ.get("XDG_CACHE_HOME")
        path = (Path(base) if base else Path.home() / ".cache") / "specsr"
    path.mkdir(parents=True, exist_ok=True)
    return path
