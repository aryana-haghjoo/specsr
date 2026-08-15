"""Augmentation of paired spectra, with explicit provenance.

Each galaxy is expanded into one original plus ``n_aug`` stochastic realizations:
a small Gaussian redshift offset, and flux perturbations scaled to the local
flux. The intent is to teach robustness to line-position variability and to
noise, rather than to add information.

Provenance is not optional
--------------------------
Every emitted row records ``parent_id``, the index of the galaxy it came from.
The original products did not: their ``id`` column was a running row index, so
after augmentation there was no way to tell which rows shared a parent. Splitting
then had to be reconstructed from ``(ra, dec, field)``, and any split drawn over
rows put ~16 near-duplicate siblings of every held-out galaxy into training.

Carrying ``parent_id`` makes the grouping explicit rather than inferred, so a
leak-free split is the obvious thing to write rather than something that has to
be recovered after the fact.

Redshift shifts are translations on a log grid
----------------------------------------------
Because ``log(lambda * (1 + z)) = log(lambda) + log(1 + z)``, applying a redshift
offset on a logarithmic wavelength grid is a *uniform shift* along the axis, the
same number of samples at every wavelength. On the old linear grid the same
operation stretched the spectrum, moving red features further than blue ones and
changing the sampling of a line depending on where it sat.

This also matches the model: the convolutional stages are translation
equivariant, so an augmentation that is a pure translation exercises exactly the
symmetry the architecture already has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import LogWavelengthGrid

__all__ = ["AugmentationConfig", "shift_redshift_on_log_grid", "augment_pair"]


@dataclass(frozen=True)
class AugmentationConfig:
    """Parameters of the augmentation.

    Attributes
    ----------
    n_aug
        Realizations generated per galaxy, in addition to the original.
    sigma_z
        Standard deviation of the Gaussian redshift offset.
    noise_frac
        Flux perturbation as a fraction of the local absolute flux.
    seed
        Base seed. Each galaxy derives its own generator from this and its
        parent index, so a build is reproducible and adding galaxies does not
        change the realizations of existing ones.
    """

    n_aug: int = 20
    sigma_z: float = 0.3
    noise_frac: float = 0.10
    seed: int = 42


def shift_redshift_on_log_grid(
    flux: np.ndarray,
    valid: np.ndarray,
    z_from: float,
    z_to: float,
    grid: LogWavelengthGrid,
    err: np.ndarray | None = None,
):
    """Move a spectrum from redshift ``z_from`` to ``z_to``.

    On a log grid this is a shift of ``log((1 + z_to) / (1 + z_from))`` in
    ``log lambda``, i.e. a constant number of samples. The shift is generally
    fractional, so neighbouring samples are linearly combined; the validity mask
    is shifted with the same weights and a sample is kept only if **both**
    contributing samples were valid, so a shift cannot manufacture data at the
    edge of a gap.

    Samples shifted in from outside the grid are marked invalid rather than
    wrapped or extrapolated: there is no measurement there.
    """
    n = flux.size
    step = np.log(grid.lambda_max / grid.lambda_min) / (n - 1)
    delta = np.log((1.0 + z_to) / (1.0 + z_from)) / step

    lo = int(np.floor(delta))
    frac = float(delta - lo)

    def _shift(arr, fill):
        out = np.full(n, fill, dtype=arr.dtype if arr.dtype != bool else arr.dtype)
        src = np.arange(n) - lo
        ok = (src >= 0) & (src < n)
        out[ok] = arr[src[ok]]
        return out, ok

    a, ok_a = _shift(flux.astype(float), np.nan)
    b, ok_b = _shift(np.roll(flux.astype(float), -1), np.nan)
    va, _ = _shift(valid, False)
    vb, _ = _shift(np.roll(valid, -1), False)

    shifted = (1.0 - frac) * a + frac * b
    # Both contributors must be real, or the interpolation invents flux.
    shifted_valid = va & vb & ok_a & ok_b & np.isfinite(shifted)
    shifted = np.where(shifted_valid, shifted, np.nan)

    if err is None:
        return shifted, shifted_valid

    ea, _ = _shift(err.astype(float), np.nan)
    eb, _ = _shift(np.roll(err.astype(float), -1), np.nan)
    # Linear combination of independent samples -> quadrature with the weights.
    shifted_err = np.sqrt(((1.0 - frac) * ea) ** 2 + (frac * eb) ** 2)
    shifted_err = np.where(shifted_valid, shifted_err, np.nan)
    return shifted, shifted_valid, shifted_err


def augment_pair(
    flux_low: np.ndarray,
    flux_low_err: np.ndarray,
    valid_low: np.ndarray,
    flux_high: np.ndarray,
    flux_high_err: np.ndarray,
    valid_high: np.ndarray,
    z: float,
    parent_id: int,
    grid: LogWavelengthGrid,
    config: AugmentationConfig | None = None,
) -> list[dict]:
    """Expand one galaxy into ``1 + n_aug`` rows.

    The first row is the unmodified original. Each subsequent row applies the
    *same* redshift offset to both members of the pair — they are the same
    galaxy, so a shift that moved them differently would teach the model a
    wavelength mapping that does not exist — and independent noise to each, since
    their measurement noise is genuinely independent.

    Returns a list of dicts, each carrying ``parent_id`` and the realized ``z``.
    """
    config = config or AugmentationConfig()
    rng = np.random.default_rng([config.seed, parent_id])

    rows = [
        {
            "flux_low": np.asarray(flux_low, dtype=float),
            "flux_low_err": np.asarray(flux_low_err, dtype=float),
            "valid_low": np.asarray(valid_low, dtype=bool),
            "flux_high": np.asarray(flux_high, dtype=float),
            "flux_high_err": np.asarray(flux_high_err, dtype=float),
            "valid_high": np.asarray(valid_high, dtype=bool),
            "z": float(z),
            "parent_id": int(parent_id),
            "is_original": True,
        }
    ]

    for _ in range(config.n_aug):
        z_new = float(z + rng.normal(0.0, config.sigma_z))
        # Negative redshift is unphysical and would invert the shift direction.
        z_new = max(z_new, 0.0)

        lo_f, lo_v, lo_e = shift_redshift_on_log_grid(
            flux_low, valid_low, z, z_new, grid, err=flux_low_err
        )
        hi_f, hi_v, hi_e = shift_redshift_on_log_grid(
            flux_high, valid_high, z, z_new, grid, err=flux_high_err
        )

        for f, e, v in ((lo_f, lo_e, lo_v), (hi_f, hi_e, hi_v)):
            sigma = config.noise_frac * np.abs(np.nan_to_num(f))
            noise = rng.normal(0.0, np.where(sigma > 0, sigma, 1e-30))
            f[v] += noise[v]
            # The added noise is part of the uncertainty budget from here on.
            e[v] = np.sqrt(np.nan_to_num(e[v]) ** 2 + sigma[v] ** 2)

        rows.append(
            {
                "flux_low": lo_f,
                "flux_low_err": lo_e,
                "valid_low": lo_v,
                "flux_high": hi_f,
                "flux_high_err": hi_e,
                "valid_high": hi_v,
                "z": z_new,
                "parent_id": int(parent_id),
                "is_original": False,
            }
        )

    return rows
