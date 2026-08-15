"""The common wavelength grid, and flux-conserving resampling onto it.

Why this module exists
----------------------
The original preprocessing interpolated **both** members of each pair onto
``linspace(1.0, 5.0, 2500)`` — a constant Δλ = 0.001601 µm grid. Measured
against native DR4 sampling:

=======  ===============  =====================================
arm      median Δλ (µm)   effect of the 2,500-point linear grid
=======  ===============  =====================================
prism    0.00613          3.8x upsampling — the wanted direction
G140M    0.00064          **2.50x downsampling**
G235M    0.00107          **1.50x downsampling**
G395M    0.00180          0.89x — roughly preserved
=======  ===============  =====================================

Below ~3.2 µm the high-resolution reference ended up with fewer than two samples
per R~1000 resolution element, i.e. undersampled. The model was therefore
trained against targets whose resolution the gridding had already reduced, and
the grid — not the model — set the ceiling on achievable sharpness. This is the
likeliest explanation for super-resolved spectra looking coarser than the
high-resolution spectra they were trained against.

The rule
--------
**Regrid the low-resolution data up onto the high-resolution sampling. Never
resample the high-resolution spectra down.**

A logarithmic grid holds ``lambda / delta_lambda`` constant, which is the natural
sampling for a spectrograph: resolution scales with wavelength, so a fixed
*fractional* step matches the instrument everywhere instead of being too coarse
in the blue and wasteful in the red.

Flux is an area, not a sample
-----------------------------
The physically meaningful quantity in an emission line is its *integrated* flux
— the area under the curve, in erg s^-1 cm^-2 — not the height of the curve at
particular abscissae. Resampling must therefore integrate, not interpolate.
:func:`resample_flux_conserving` is the default for that reason.

Measured on native G140M sampling with R~1000 lines:

==========================  =====================  ====================
grid and method             integrated flux error  line peak retained
==========================  =====================  ====================
old linear, interpolation           -1.196%                79.2%
old linear, conserving              -0.011%                72.4%
log R=4000, interpolation           -0.098%                90.0%
**log R=4000, conserving**        **-0.032%**            **97.8%**
==========================  =====================  ====================

Two things follow. The old pipeline discarded roughly a fifth of every
emission line's peak and about 1% of its integrated flux *before the model saw
the data*. And on a grid that is too coarse there is no good option: conserving
the integral costs peak height, and preserving peak height costs the integral.
Only a sufficiently fine grid lets both be kept, which is why the grid — not the
resampling method — was the real defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LogWavelengthGrid", "DEFAULT_GRID", "resample_flux_conserving", "sampling_report"]


@dataclass(frozen=True)
class LogWavelengthGrid:
    """A logarithmic (constant resolving power) wavelength grid.

    Attributes
    ----------
    lambda_min, lambda_max
        Bounds in microns.
    resolving_power
        The constant ``lambda / delta_lambda`` of the *grid*. This is a sampling
        density, not the instrument's spectral resolution: it must comfortably
        exceed the native sampling of every arm, or the grid degrades the data
        it is meant to carry.
    """

    lambda_min: float = 1.0
    lambda_max: float = 5.3
    resolving_power: float = 4000.0

    @property
    def n_samples(self) -> int:
        """Number of grid points implied by the resolving power."""
        return int(np.ceil(self.resolving_power * np.log(self.lambda_max / self.lambda_min)))

    def centers(self) -> np.ndarray:
        """Grid centres, shape ``(n_samples,)``."""
        return np.geomspace(self.lambda_min, self.lambda_max, self.n_samples)

    def edges(self) -> np.ndarray:
        """Bin edges, shape ``(n_samples + 1,)``.

        Geometric mid-points, so that edges are evenly spaced in log wavelength
        exactly as the centres are.
        """
        c = self.centers()
        inner = np.sqrt(c[:-1] * c[1:])
        first = c[0] ** 2 / inner[0]
        last = c[-1] ** 2 / inner[-1]
        return np.concatenate([[first], inner, [last]])

    def samples_per_resolution_element(self, wavelength_um, R_instrument: float = 1000.0):
        """How many grid samples span one instrumental resolution element.

        Nyquist needs >= 2; below that the grid cannot represent the features
        present in the data.
        """
        lam = np.asarray(wavelength_um, dtype=float)
        # For a log grid the step is lam / R_grid everywhere.
        d_lambda_grid = lam / self.resolving_power
        d_lambda_res = lam / R_instrument
        return d_lambda_res / d_lambda_grid  # == R_grid / R_instrument

    def __repr__(self) -> str:
        return (
            f"LogWavelengthGrid({self.lambda_min}-{self.lambda_max} um, "
            f"R={self.resolving_power:g}, N={self.n_samples})"
        )


#: The project default. ``R = 4000`` is the smallest round value at or above the
#: finest native sampling encountered anywhere in DR4 (lambda/dlambda = 3737, in
#: G235M), so nothing is downsampled. Range extends to 5.3 um to cover G395M.
DEFAULT_GRID = LogWavelengthGrid(lambda_min=1.0, lambda_max=5.3, resolving_power=4000.0)


def _bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Mid-point bin edges for an arbitrary (possibly irregular) grid."""
    c = np.asarray(centers, dtype=float)
    inner = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (inner[0] - c[0])
    last = c[-1] + (c[-1] - inner[-1])
    return np.concatenate([[first], inner, [last]])


def resample_flux_conserving(
    wave_in: np.ndarray,
    flux_in: np.ndarray,
    wave_out: np.ndarray,
    err_in: np.ndarray | None = None,
    fill: float = np.nan,
):
    """Resample a spectrum onto ``wave_out``, conserving integrated flux.

    Each output bin receives the integral of the input over the overlapping
    portion of each input bin, divided by the output bin width. Equivalently:
    the output is the *average flux density* over the output bin, so
    ``sum(flux * bin_width)`` is preserved up to the edges of the covered range.

    Why not plain interpolation
    ---------------------------
    Linear interpolation samples the curve at new abscissae; it does not
    integrate. On a narrow emission line, interpolation can land between the
    peak samples and systematically lose line flux, while a flux-conserving
    rebin preserves the line's integrated flux by construction. Accurate *line
    fluxes* are the quantity the science case depends on, so the conserving form
    is the right default.

    Parameters
    ----------
    wave_in, flux_in
        Input grid and flux density, same length. ``wave_in`` must be increasing.
    wave_out
        Output grid centres.
    err_in
        Optional uncertainty on ``flux_in``. Propagated by summing in quadrature
        over the same overlaps, which is correct for independent input samples
        and mildly conservative when they are correlated.
    fill
        Value for output bins with no input coverage.

    Returns
    -------
    ``flux_out`` or ``(flux_out, err_out)`` if ``err_in`` was given.
    """
    wave_in = np.asarray(wave_in, dtype=float)
    flux_in = np.asarray(flux_in, dtype=float)
    wave_out = np.asarray(wave_out, dtype=float)

    if wave_in.ndim != 1 or flux_in.shape != wave_in.shape:
        raise ValueError("wave_in and flux_in must be 1-D and the same length")
    if not np.all(np.diff(wave_in) > 0):
        order = np.argsort(wave_in)
        wave_in, flux_in = wave_in[order], flux_in[order]
        if err_in is not None:
            err_in = np.asarray(err_in, dtype=float)[order]

    in_edges = _bin_edges_from_centers(wave_in)
    out_edges = _bin_edges_from_centers(wave_out)

    flux_out = np.full(wave_out.size, fill, dtype=float)
    err_out = np.full(wave_out.size, fill, dtype=float) if err_in is not None else None
    err_arr = np.asarray(err_in, dtype=float) if err_in is not None else None

    # Walk both edge arrays once: O(n_in + n_out) rather than O(n_in * n_out).
    lo_idx = np.searchsorted(in_edges, out_edges[:-1], side="right") - 1
    hi_idx = np.searchsorted(in_edges, out_edges[1:], side="left")

    for j in range(wave_out.size):
        lo = max(lo_idx[j], 0)
        hi = min(hi_idx[j], wave_in.size)
        if hi <= lo:
            continue
        left = np.maximum(in_edges[lo:hi], out_edges[j])
        right = np.minimum(in_edges[lo + 1 : hi + 1], out_edges[j + 1])
        overlap = np.clip(right - left, 0.0, None)
        total = overlap.sum()
        if total <= 0:
            continue
        seg = flux_in[lo:hi]
        good = np.isfinite(seg) & (overlap > 0)
        if not good.any():
            continue
        w = overlap[good]
        flux_out[j] = np.dot(seg[good], w) / w.sum()
        if err_arr is not None:
            e = err_arr[lo:hi][good]
            # Quadrature sum weighted the same way the flux was averaged.
            err_out[j] = np.sqrt(np.dot(np.nan_to_num(e) ** 2, w**2)) / w.sum()

    return (flux_out, err_out) if err_arr is not None else flux_out


def sampling_report(native_delta_lambda: dict, grid: LogWavelengthGrid | None = None) -> str:
    """Human-readable check that ``grid`` does not downsample any input arm.

    ``native_delta_lambda`` maps an arm name to ``(lambda_min, lambda_max,
    median_delta_lambda)`` in microns. Use this before committing to a grid: a
    grid that downsamples its inputs silently caps the achievable resolution.
    """
    grid = grid or DEFAULT_GRID
    lines = [f"{grid!r}", ""]
    lines.append(f"{'arm':8s} {'native l/dl':>12s} {'grid R':>8s} {'ratio':>7s}  verdict")
    worst = 0.0
    for arm, (_lo, hi, dlam) in sorted(native_delta_lambda.items()):
        native_R = hi / dlam  # finest sampling occurs at the red end
        worst = max(worst, native_R)
        ratio = grid.resolving_power / native_R
        verdict = "ok" if ratio >= 1.0 else "DOWNSAMPLES"
        lines.append(
            f"{arm:8s} {native_R:12.0f} {grid.resolving_power:8.0f} {ratio:7.2f}  {verdict}"
        )
    lines.append("")
    lines.append(
        f"finest native sampling anywhere: {worst:.0f}; "
        f"grid R={grid.resolving_power:g} -> "
        + ("no downsampling" if grid.resolving_power >= worst else "GRID DEGRADES THE DATA")
    )
    return "\n".join(lines)
