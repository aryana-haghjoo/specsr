"""Combine the medium gratings into one high-resolution reference.

Coverage
--------
The three medium gratings overlap rather than abutting::

    G140M  0.70 -------- 2.20
    G235M         1.66 --------------- 4.00
    G395M                       2.87 --------------- 5.48

Each covers only part of the 1.0-5.3 um grid on its own (45%, 48% and 36%
respectively), but the overlaps fall exactly where individual arms degrade at
their edges: G235M's unusable blue edge is covered by G140M, and its unusable red
run by G395M. Stitched, coverage is **98.3% +/- 2.8%** of the grid, and no
wavelength is invalid for a majority of targets. The residual ~1.7% is
target-specific.

Never invent the target
-----------------------
The remaining invalid samples matter more than their 1.7% suggests, because this
is the *reference the model is trained to reproduce*.

The original pipeline filled every non-finite sample with the per-spectrum
median::

    def replace_nans(arr):
        median = np.median(arr[~np.isnan(arr)])
        return np.where(np.isnan(arr), median, arr)

Applied to ``flux_high``, that trains the model to emit a flat constant wherever
the detector did not measure — and penalises it for placing a real emission line
there. Filling is strictly worse than excluding: zero teaches "no flux here",
the median teaches "featureless continuum here", and both are fabrications.

So this module returns an explicit ``valid`` mask alongside flux and error, and
leaves invalid samples as ``nan`` so that a caller which ignores the mask fails
loudly instead of training on a fabricated value.

The consuming contract has two halves, and both are required
(see :func:`specsr.training.losses.sr1_deblend_loss`):

1. Replace invalid samples with a **neutral numerical value** — zero, in the
   per-spectrum normalised units the losses work in. Not because the value means
   anything, but because the line mask is built by smoothing the reference over a
   wide kernel, and an arbitrary number at an unmeasured wavelength would be
   dragged into neighbouring *measured* samples and mistaken for a line.
2. Pass ``valid`` to the loss so those samples contribute no gradient.

Filling without masking is what the original pipeline did. Masking without
filling leaves the fill value free to contaminate neighbourhood statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import DEFAULT_GRID, LogWavelengthGrid, resample_flux_conserving

__all__ = ["StitchedSpectrum", "stitch_gratings", "resample_with_mask"]


@dataclass
class StitchedSpectrum:
    """A reference spectrum on the common grid, with an explicit validity mask.

    Attributes
    ----------
    wavelength
        The common grid, microns.
    flux, flux_err
        Inverse-variance combined across arms where they overlap. Invalid
        samples hold ``nan`` — deliberately, so that a caller which forgets the
        mask fails loudly rather than training on a fabricated value.
    valid
        True where at least one arm genuinely measured this wavelength.
    n_arms
        How many arms contributed to each sample. Useful for diagnostics: a
        sample built from two arms in an overlap region is better constrained
        than one from a single arm at the edge of its range.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    valid: np.ndarray
    n_arms: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of the grid that is genuinely measured."""
        return float(self.valid.mean())

    def __repr__(self) -> str:
        return (
            f"StitchedSpectrum(N={self.wavelength.size}, "
            f"coverage={100 * self.coverage:.1f}%, "
            f"max_arms={int(self.n_arms.max()) if self.n_arms.size else 0})"
        )


def resample_with_mask(
    wave_in: np.ndarray,
    flux_in: np.ndarray,
    err_in: np.ndarray,
    wave_out: np.ndarray,
    min_coverage: float = 0.5,
):
    """Resample one arm onto ``wave_out``, tracking which samples are real.

    Validity is resampled as an indicator alongside the flux, so a bin that
    draws mostly on non-finite input is marked invalid rather than silently
    inheriting whatever finite neighbours happened to overlap it.
    ``min_coverage`` is the fraction of a bin that must come from valid input.
    """
    finite = np.isfinite(flux_in)
    if finite.sum() < 2:
        n = np.asarray(wave_out).size
        return (
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.zeros(n, dtype=bool),
        )

    flux_clean = np.where(finite, flux_in, 0.0)
    err_clean = np.where(finite & np.isfinite(err_in), err_in, 0.0)

    # Resample the indicator on the same footing as the flux, so "how much of
    # this output bin was actually measured" is answered by the same overlaps.
    cov = resample_flux_conserving(wave_in, finite.astype(float), wave_out, fill=0.0)
    cov = np.nan_to_num(cov, nan=0.0)
    valid = cov >= min_coverage

    flux_out, err_out = resample_flux_conserving(
        wave_in, flux_clean, wave_out, err_in=err_clean, fill=np.nan
    )
    # Undo the zero-fill dilution: bins are partly covered, so rescale by the
    # covered fraction to recover the mean over the *measured* part alone.
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_out = np.where(valid, flux_out / np.clip(cov, 1e-6, None), np.nan)
        err_out = np.where(valid, err_out / np.clip(cov, 1e-6, None), np.nan)

    return flux_out, err_out, valid


def stitch_gratings(
    arms: dict[str, dict],
    grid: LogWavelengthGrid | None = None,
    min_coverage: float = 0.5,
) -> StitchedSpectrum:
    """Combine medium gratings into one reference on the common grid.

    Parameters
    ----------
    arms
        ``{name: {"wavelength": ..., "flux": ..., "flux_err": ...}}``, as
        returned by :func:`specsr.data.ingest.read_spectrum`.
    grid
        Target grid; defaults to :data:`specsr.data.grid.DEFAULT_GRID`.
    min_coverage
        Fraction of an output bin that must be genuinely measured for that
        sample to count as valid.

    Notes
    -----
    Overlapping arms are combined by **inverse-variance weighting**, which is the
    right estimator when the arms are independent measurements of the same
    quantity: it weights each by how well it constrains the flux, so the noisier
    edge of one arm does not degrade the well-measured middle of another. A
    simple average would let a barely-detected edge sample pull the combined
    value around.

    Invalid samples are left as ``nan`` and marked ``valid=False``. They are
    never filled — see the module docstring.
    """
    grid = grid or DEFAULT_GRID
    wave_out = grid.centers()
    n = wave_out.size

    num = np.zeros(n)       # sum of w_i * f_i
    den = np.zeros(n)       # sum of w_i
    n_arms = np.zeros(n, dtype=int)

    for arm in arms.values():
        f, e, v = resample_with_mask(
            np.asarray(arm["wavelength"], dtype=float),
            np.asarray(arm["flux"], dtype=float),
            np.asarray(arm["flux_err"], dtype=float),
            wave_out,
            min_coverage=min_coverage,
        )
        # An arm contributes only where it is valid AND has a usable error.
        usable = v & np.isfinite(f) & np.isfinite(e) & (e > 0)
        w = np.zeros(n)
        w[usable] = 1.0 / e[usable] ** 2
        num[usable] += w[usable] * f[usable]
        den += w
        n_arms += usable.astype(int)

    valid = den > 0
    flux = np.full(n, np.nan)
    err = np.full(n, np.nan)
    flux[valid] = num[valid] / den[valid]
    err[valid] = 1.0 / np.sqrt(den[valid])

    return StitchedSpectrum(
        wavelength=wave_out,
        flux=flux,
        flux_err=err,
        valid=valid,
        n_arms=n_arms,
    )
