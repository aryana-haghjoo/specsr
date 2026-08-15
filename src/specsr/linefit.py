"""Gaussian emission-line fitting, and the S/N derived from it.

Fits a Gaussian on a linear continuum in a window around each expected line and
reports amplitude over continuum scatter. This is the measurement behind the
paper's S/N figure.

The continuum scatter is estimated from **sidebands** — an annulus around the
line, excluding its core — rather than from the whole window. Using the window
would fold the line itself into the noise estimate and depress the S/N of
exactly the strong lines the figure is about.

Note what this quantity is and is not: it references only the spectrum being
measured, never the HR truth, so a high S/N means "a confident detection of
*something*", not "the right line flux". Establishing that a line is *correct*
needs the reference, which is why
:func:`specsr.plotting.plot_line_flux_comparison` exists alongside it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "fit_line_sideband_weighted",
    "gauss_lin",
    "line_snr_from_fit",
    "mad_sigma",
    "measure_line_snr",
]


def gauss_lin(x, amp, mu, sigma, c0, c1):
    """Gaussian on a linear continuum."""
    sigma = np.clip(sigma, 1e-12, None)
    return c0 + c1 * (x - mu) + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mad_sigma(y):
    """Robust scatter via MAD, NaN-aware. NaN when there is too little data."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8:
        return np.nan
    return 1.4826 * float(np.median(np.abs(y - np.median(y))))


def _masks(x, mu0, fit_halfwin, core_halfwin, sb_gap, sb_width):
    x = np.asarray(x, float)
    fit = (x >= mu0 - fit_halfwin) & (x <= mu0 + fit_halfwin)
    core = np.abs(x - mu0) <= core_halfwin
    left = (x >= mu0 - (sb_gap + sb_width)) & (x <= mu0 - sb_gap)
    right = (x >= mu0 + sb_gap) & (x <= mu0 + (sb_gap + sb_width))
    return fit, core, left | right


def fit_line_sideband_weighted(
    x, y, mu0, *,
    fit_halfwin: float = 0.25,
    core_halfwin: float = 0.05,
    sb_gap: float = 0.03,
    sb_width: float = 0.12,
    sigma_bounds: tuple[float, float] = (0.001, 0.12),
    mu_bounds_half: float = 0.01,
    allow_negative_amp: bool = True,
    maxfev: int = 40000,
):
    """Fit one line. Returns a dict of parameters, or ``None`` if unfittable.

    Negative amplitudes are allowed by default: forcing positivity would turn a
    non-detection into a small positive bump and manufacture signal where there
    is none.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fit_m, core_m, side_m = _masks(x, mu0, fit_halfwin, core_halfwin, sb_gap, sb_width)

    xx, yy = x[fit_m], y[fit_m]
    if xx.size < 15:
        return None

    y_sb = y[side_m & (~core_m) & fit_m]
    if np.isfinite(y_sb).sum() < 30:
        return None

    # Floor the continuum scatter *relative* to the data, not at an absolute
    # 1e-3 as the notebook did. Physical flux here is ~1e-21, so that constant
    # won every comparison and every S/N came out as amp/1e-3, i.e. zero.
    sigma_cont = mad_sigma(y_sb)
    if not np.isfinite(sigma_cont) or sigma_cont <= 0:
        finite_sb = y_sb[np.isfinite(y_sb)]
        scale = float(np.nanmedian(np.abs(finite_sb))) if finite_sb.size else 0.0
        sigma_cont = max(scale * 1e-3, np.finfo(float).tiny)
    c0_est = float(np.median(y_sb[np.isfinite(y_sb)]))
    dx = float(np.median(np.diff(xx))) if xx.size > 1 else 0.002

    resid = yy - c0_est
    amp_pos, amp_neg = float(np.nanmax(resid)), float(np.nanmin(resid))
    amp0 = amp_pos if abs(amp_pos) >= abs(amp_neg) else amp_neg
    if not allow_negative_amp:
        amp0 = max(amp0, 1e-6)
    if abs(amp0) < sigma_cont * 1e-6:  # relative, for the same reason
        amp0 = sigma_cont * 1e-6

    sig0 = float(np.clip(2 * dx, *sigma_bounds))
    p0 = np.array([amp0, mu0, sig0, c0_est, 0.0], float)
    amp_lo, amp_hi = (-np.inf, np.inf) if allow_negative_amp else (0.0, np.inf)
    lo = np.array([amp_lo, mu0 - mu_bounds_half, sigma_bounds[0], -np.inf, -np.inf])
    hi = np.array([amp_hi, mu0 + mu_bounds_half, sigma_bounds[1], np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            gauss_lin, xx, yy, p0=p0, bounds=(lo, hi),
            sigma=np.full_like(xx, sigma_cont), absolute_sigma=True, maxfev=maxfev)
    except Exception:
        return None

    amp, mu, sig, c0, c1 = map(float, popt)
    amp_err = float(np.sqrt(pcov[0, 0])) if (
        pcov is not None and np.isfinite(pcov[0, 0]) and pcov[0, 0] > 0) else np.nan
    return {"amp": amp, "amp_err": amp_err, "mu": mu, "sigma": sig,
            "c0": c0, "c1": c1, "sigma_cont": float(sigma_cont)}


def line_snr_from_fit(fit):
    """``(amp/sigma_cont, amp/amp_err)`` from a fit, or ``(nan, nan)``."""
    if fit is None:
        return np.nan, np.nan
    amp, sigma_cont, amp_err = fit["amp"], fit["sigma_cont"], fit["amp_err"]
    sn_cont = abs(amp) / sigma_cont if np.isfinite(sigma_cont) and sigma_cont > 0 else np.nan
    sn_err = abs(amp) / amp_err if np.isfinite(amp_err) and amp_err > 0 else np.nan
    return float(sn_cont), float(sn_err)


def measure_line_snr(wavelength, z, spectra, lines_rest_um, line_names=None, **fit_kw):
    """Per-line S/N for several spectrum sets over the same objects.

    Parameters
    ----------
    spectra
        ``{"LR": array, "SR": array, "HR": array, ...}``, each ``(n, n_lambda)``.
    lines_rest_um
        Rest wavelengths, microns. Redshifted per object with ``z``.

    Returns
    -------
    ``{f"{line}_sn_{kind}": array}`` with NaN where a line falls off the grid or
    the fit fails — left as NaN rather than zero, since "not measurable" and
    "measured as zero" are different statements.
    """
    wavelength = np.asarray(wavelength, float)
    z = np.asarray(z, float)
    lines_rest_um = np.asarray(lines_rest_um, float)
    if line_names is None:
        line_names = [f"line_{i}" for i in range(lines_rest_um.size)]

    n = len(z)
    out = {f"{nm}_sn_{kind}": np.full(n, np.nan)
           for nm in line_names for kind in spectra}

    lo, hi = float(wavelength[0]), float(wavelength[-1])
    for i in range(n):
        for nm, lam_rest in zip(line_names, lines_rest_um, strict=True):
            mu0 = lam_rest * (1.0 + z[i])
            if not (lo < mu0 < hi):
                continue
            for kind, arr in spectra.items():
                fit = fit_line_sideband_weighted(wavelength, arr[i], mu0, **fit_kw)
                out[f"{nm}_sn_{kind}"][i] = line_snr_from_fit(fit)[0]
    return out
