"""Torch datasets over the paired spectra products.

Each row is one (low-resolution prism, medium-resolution grating) pair on a
shared wavelength grid, plus the reference uncertainty, the catalogue redshift
and provenance.

Normalisation is **per spectrum**: each spectrum is standardised by its own mean
and standard deviation. That is deliberate — it carries no information between
rows, so unlike a dataset-wide normalisation it cannot leak anything about the
held-out split. The reference uncertainty is divided by the same scale so it
stays in the normalised units the loss works in, and the per-row mean and scale
are returned so predictions can be pushed back into physical units for
evaluation and plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["FixedGridSpectraDataset", "PairedSpectra", "normalize_spectrum"]


def normalize_spectrum(x: np.ndarray, eps: float = 1e-25) -> tuple[np.ndarray, float, float]:
    """Standardise one spectrum by its own mean and standard deviation.

    Returns ``(normalised, mean, std)``. NaN-aware, and the scale is floored so a
    flat or empty spectrum cannot produce a division by zero.
    """
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    if std < eps:
        std = eps
    return (x - mean) / std, mean, std


class PairedSpectra(Dataset):
    """Paired low/high-resolution spectra from a built ``.npz`` product.

    Parameters
    ----------
    npz_path
        Path to a product written by the preprocessing pipeline.
    normalize_flux
        Standardise each spectrum by its own statistics (default). When off,
        fluxes are returned as stored but the per-row statistics are still
        computed and returned, so de-normalisation helpers keep working.
    target_key
        Which reference to train against — ``"flux_high"`` for the stitched
        grating spectrum, or ``"flux_high_smoothed"`` for its smoothed variant.
        The matching ``*_err`` array is used as the reference uncertainty.
    dtype
        Storage dtype. ``float32`` halves memory against the ``float64`` the
        products are written in, which matters: at the log wavelength grid
        (~6.7k samples) a 52k-row product is several GB per array.

    Returns
    -------
    Each item is a dict so that call sites index by name rather than by
    position. The previous tuple-based interface made it easy to silently
    transpose two fields at a call site; a dict makes that a ``KeyError``.

    ``flux_low``, ``flux_high``, ``flux_high_err`` are ``(L,)`` tensors;
    ``z`` is a scalar tensor; ``flux_high_mean``/``flux_high_std`` are the
    per-row de-normalisation statistics; ``index`` is the row index.
    """

    #: Arrays required to be present in a product.
    REQUIRED = ("flux_low", "z")

    def __init__(
        self,
        npz_path: str | Path,
        normalize_flux: bool = True,
        target_key: str = "flux_high",
        dtype: torch.dtype = torch.float32,
    ):
        self.npz_path = Path(npz_path)
        self.target_key = target_key
        self.normalize_flux = normalize_flux

        with np.load(self.npz_path, allow_pickle=True) as data:
            required = (*self.REQUIRED, target_key, f"{target_key}_err")
            missing = [k for k in required if k not in data]
            if missing:
                raise KeyError(
                    f"{self.npz_path} is missing {missing}. Available: {sorted(data.files)}"
                )

            flux_lo = np.asarray(data["flux_low"], dtype=np.float64)
            flux_hi = np.asarray(data[target_key], dtype=np.float64)
            flux_hi_err = np.asarray(data[f"{target_key}_err"], dtype=np.float64)
            self.z = torch.as_tensor(np.asarray(data["z"], dtype=np.float64), dtype=dtype)

            self.wavelength = (
                torch.as_tensor(np.asarray(data["wavelength_high"]), dtype=dtype)
                if "wavelength_high" in data
                else None
            )
            # Provenance, when the product carries it. Kept as numpy: these are
            # used for splitting and bookkeeping, not fed to the model.
            self.parent_id = np.asarray(data["parent_id"]) if "parent_id" in data else None
            self.ra = np.asarray(data["ra"]) if "ra" in data else None
            self.dec = np.asarray(data["dec"]) if "dec" in data else None
            self.field = np.asarray(data["field"]).astype(str) if "field" in data else None

        n = len(flux_lo)
        means = np.empty(n, dtype=np.float64)
        stds = np.empty(n, dtype=np.float64)

        if normalize_flux:
            for i in range(n):
                flux_lo[i], _, _ = normalize_spectrum(flux_lo[i])
                flux_hi[i], means[i], stds[i] = normalize_spectrum(flux_hi[i])
                # Threshold matches normalize_spectrum's eps. At 1e-9 it never
                # fired for these units (HR std ~1e-21), leaving the error in
                # physical units while the flux was normalised -- which made
                # every downstream 1/err^2 clamp to var_floor and go uniform.
                flux_hi_err[i] = flux_hi_err[i] / (stds[i] if stds[i] > 1e-25 else 1.0)
                # Masked pixels use a sentinel err=1.0; normalised that is ~1e20
                # and overflows float32 when squared. Cap well above any real
                # normalised error so they keep ~zero weight but stay finite.
                flux_hi_err[i] = np.clip(flux_hi_err[i], 0.0, 1e3)
        else:
            for i in range(n):
                means[i] = float(np.nanmean(flux_hi[i]))
                s = float(np.nanstd(flux_hi[i]))
                stds[i] = s if s >= 1e-25 else 1e-25

        self.flux_low = torch.as_tensor(flux_lo, dtype=dtype)
        self.flux_high = torch.as_tensor(flux_hi, dtype=dtype)
        self.flux_high_err = torch.as_tensor(flux_hi_err, dtype=dtype)
        self.flux_high_mean = torch.as_tensor(means, dtype=dtype)
        self.flux_high_std = torch.as_tensor(stds, dtype=dtype)

    def __len__(self) -> int:
        return len(self.flux_low)

    @property
    def n_samples(self) -> int:
        """Length of the wavelength axis."""
        return self.flux_low.shape[-1]

    def __getitem__(self, idx: int) -> dict:
        return {
            "flux_low": self.flux_low[idx],
            "flux_high": self.flux_high[idx],
            "flux_high_err": self.flux_high_err[idx],
            "z": self.z[idx],
            "flux_high_mean": self.flux_high_mean[idx],
            "flux_high_std": self.flux_high_std[idx],
            "index": idx,
        }

    def denormalize(self, x: torch.Tensor, idx) -> torch.Tensor:
        """Map normalised flux back to the physical units of the reference."""
        mean = self.flux_high_mean[idx]
        std = self.flux_high_std[idx]
        if x.ndim > mean.ndim:
            shape = (-1,) + (1,) * (x.ndim - mean.ndim)
            mean = mean.reshape(shape)
            std = std.reshape(shape)
        return x * std + mean

    def __repr__(self) -> str:
        return (
            f"PairedSpectra({self.npz_path.name}, n={len(self)}, "
            f"L={self.n_samples}, target={self.target_key!r}, "
            f"normalized={self.normalize_flux})"
        )


class FixedGridSpectraDataset(Dataset):
    """Paired spectra on a fixed grid, returned as a tuple.

    ``(low, high, high_err, z, high_mean, high_std)`` — the last two let a caller
    de-normalise back into physical HR units, which is what line fluxes and
    residuals mean anything in.

    This is the loader the training scripts use, and it lives here rather than in
    ``train/`` so the normalisation exists once. It previously had a second copy
    in ``train/sr1_best/train_sr1.py``; the HR-error fix below then had to be
    applied twice, which is exactly the kind of duplication that lets one copy
    quietly keep a bug.

    :class:`PairedSpectra` is the dict-returning equivalent and the better API
    for new code; this exists because the trained checkpoints were produced
    against the tuple form.
    """

    def __init__(self, npz_path, normalize_flux: bool = True,
                 target_key_raw: str = "flux_high"):
        data = np.load(npz_path, allow_pickle=True)
        flux_lo = data["flux_low"]
        flux_hi = data[target_key_raw]
        flux_hi_err = data[target_key_raw + "_err"]
        self.z = data["z"]

        high_mean: list = []
        high_std: list = []

        if normalize_flux:
            lo_n, hi_n, err_n = [], [], []
            for f_lo, f_hi, e_hi in zip(flux_lo, flux_hi, flux_hi_err, strict=True):
                f_lo_norm, _, _ = normalize_spectrum(f_lo)
                f_hi_norm, m_hi, s_hi = normalize_spectrum(f_hi)
                # Threshold must match normalize_spectrum's eps. At 1e-9 it never
                # fired -- HR std here is ~1e-21 -- so the divisor stayed 1.0 and
                # the error kept physical units while the flux was normalised to
                # unit variance. Every downstream 1/err**2 then hit var_floor and
                # came out uniform, silently disabling inverse-variance weighting.
                e_scaled = e_hi / (s_hi if s_hi > 1e-25 else 1.0)
                # Masked pixels carry a sentinel err=1.0 in physical units, which
                # normalises to ~1e21 and overflows float32 when squared. Cap far
                # above any real normalised error (~0.2) so those pixels keep
                # ~zero weight while everything stays finite.
                e_scaled = np.clip(e_scaled, 0.0, 1e3)

                lo_n.append(f_lo_norm)
                hi_n.append(f_hi_norm)
                err_n.append(e_scaled)
                high_mean.append(m_hi)
                high_std.append(s_hi)

            flux_lo = np.array(lo_n)
            flux_hi = np.array(hi_n)
            flux_hi_err = np.array(err_n)
        else:
            for f_hi in flux_hi:
                high_mean.append(np.nanmean(f_hi))
                s_hi = float(np.nanstd(f_hi))
                high_std.append(s_hi if s_hi >= 1e-25 else 1e-25)

        self.lowres = torch.tensor(flux_lo, dtype=torch.float32)
        self.highres = torch.tensor(flux_hi, dtype=torch.float32)
        self.highres_err = torch.tensor(flux_hi_err, dtype=torch.float32)
        self.high_mean = torch.tensor(np.array(high_mean), dtype=torch.float32)
        self.high_std = torch.tensor(np.array(high_std), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.lowres)

    def __getitem__(self, idx):
        return (
            self.lowres[idx],
            self.highres[idx],
            self.highres_err[idx],
            self.z[idx],
            self.high_mean[idx],
            self.high_std[idx],
        )
