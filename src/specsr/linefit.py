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
:func:`specsr.plotting.plot_line_flux_comparison` exists alongside it. That
figure takes its fluxes from this same fit (:func:`measure_line_fluxes`).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares

__all__ = [
    "HBETA_OIII_REST_UM",
    "fit_hbeta_oiii",
    "fit_line_sideband_weighted",
    "gauss_lin",
    "line_flux_from_fit",
    "line_snr_from_fit",
    "mad_sigma",
    "measure_line_fluxes",
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


C_KMS = 299792.458
# H-beta, [O III] 4959 and [O III] 5007: three lines within 9000 km/s that the
# prism merges into one feature.
HBETA_OIII_REST_UM = (0.486133, 0.495891, 0.500684)


def fit_hbeta_oiii(wavelength, flux, z, *, window_kms: float = 20000.0,
                   sigma_lo_kms: float = 40.0, sigma_hi_kms: float = 3000.0):
    """Joint fit of H-beta and the [O III] doublet: three Gaussians, one width.

    A single Gaussian cannot measure these lines at prism resolution. Centred on
    H-beta it widens until it holds the [O III] lines as well (fitted sigma
    ~8000 km/s for half the sample, and ten times the true flux); centred on
    5007 it absorbs 4959 and reads a third high. Fitting the three together at
    fixed centres with a shared width assigns the blended flux to the right
    line. It is the model ``scripts/doublet_deblending.py`` uses for the doublet
    ratio, over a wider window so that the prism's H-beta wing and the continuum
    beyond it are inside the fit.

    The fit runs in velocity about the doublet midpoint, with amplitudes bounded
    non-negative, for the reasons given there.

    Returns ``{"amp": (A_hb, A_4959, A_5007), "sigma_um": (...), "sigma_kms":
    float}``, with each ``sigma_um`` the shared velocity width at that line's
    wavelength, or ``None`` when the lines fall off the grid or the fit fails.
    """
    wavelength = np.asarray(wavelength, float)
    flux = np.asarray(flux, float)
    centres = np.asarray(HBETA_OIII_REST_UM) * (1.0 + float(z))
    lam0 = 0.5 * (centres[1] + centres[2])
    v_all = (wavelength - lam0) / lam0 * C_KMS
    m = np.abs(v_all) <= window_kms
    if centres[0] < wavelength[0] or centres[2] > wavelength[-1] or m.sum() < 30:
        return None
    v, y = v_all[m], flux[m]
    if not np.isfinite(y).all():
        return None
    vc = (centres - lam0) / lam0 * C_KMS

    def model(p):
        g = p[0] + p[1] * v
        for amp, v0 in zip(p[3:], vc, strict=True):
            g = g + amp * np.exp(-0.5 * ((v - v0) / p[2]) ** 2)
        return g

    med = float(np.median(y))
    amp0 = max(float(np.max(y) - med), 1e-6)
    p0 = [med, 0.0, 300.0, amp0 * 0.3, amp0 / 3.33, amp0]
    lo = [-np.inf, -np.inf, sigma_lo_kms, 0.0, 0.0, 0.0]
    hi = [np.inf, np.inf, sigma_hi_kms, np.inf, np.inf, np.inf]
    r = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi),
                      max_nfev=4000, method="trf")
    if not r.success:
        return None
    sig = float(r.x[2])
    # An amplitude pinned at its lower bound comes back as a denormal-sized
    # positive number, not as zero. Left alone it is plotted forty decades
    # below the line it failed to find; it is a non-detection and is returned
    # as one.
    amps = [float(a) if a > 1e-4 * amp0 else 0.0 for a in r.x[3:]]
    return {"amp": tuple(amps),
            "sigma_um": tuple(float(sig / C_KMS * c) for c in centres),
            "sigma_kms": sig}


def line_flux_from_fit(fit):
    """Integrated flux of the fitted Gaussian, ``sqrt(2 pi) * amp * sigma``.

    In the units of flux times wavelength of the spectrum that was fitted, and
    NaN for a failed fit. A negative amplitude gives a negative flux: it is a
    measurement of no emission, and is left to the caller to count.
    """
    if fit is None:
        return np.nan
    return float(np.sqrt(2.0 * np.pi) * fit["amp"] * fit["sigma"])


def measure_line_fluxes(wavelength, z, spectra, scales, lines_rest_um,
                        line_names=None, **fit_kw):
    """Per-line Gaussian-fit flux and S/N for several spectrum sets.

    The flux counterpart of :func:`measure_line_snr`, from the same fit, so the
    flux of a line and its S/N are two readings of one measurement.

    Parameters
    ----------
    spectra
        ``{"LR": array, "SR": array, "HR": array, ...}``, each ``(n, n_lambda)``,
        in the normalised flux the fit's starting values and bounds are tuned
        for.
    scales
        ``{kind: (n,) array}``, the per-spectrum factor that turns each set's
        normalised flux back into physical flux density. An additive offset
        needs no undoing: the fit carries its own continuum.

    Returns
    -------
    ``{f"{line}_{quantity}_{kind}": array}`` for the quantities ``flux``
    (physical flux density times the wavelength unit), ``sn``, ``sigma`` and
    ``mu`` (both in the wavelength unit). NaN where a line falls off the grid
    or the fit fails.
    """
    wavelength = np.asarray(wavelength, float)
    z = np.asarray(z, float)
    lines_rest_um = np.asarray(lines_rest_um, float)
    if line_names is None:
        line_names = [f"line_{i}" for i in range(lines_rest_um.size)]

    n = len(z)
    out = {f"{nm}_{q}_{kind}": np.full(n, np.nan)
           for nm in line_names for kind in spectra
           for q in ("flux", "sn", "sigma", "mu")}

    lo, hi = float(wavelength[0]), float(wavelength[-1])
    for i in range(n):
        for nm, lam_rest in zip(line_names, lines_rest_um, strict=True):
            mu0 = lam_rest * (1.0 + z[i])
            if not (lo < mu0 < hi):
                continue
            for kind, arr in spectra.items():
                fit = fit_line_sideband_weighted(wavelength, arr[i], mu0, **fit_kw)
                if fit is None:
                    continue
                out[f"{nm}_flux_{kind}"][i] = line_flux_from_fit(fit) * float(scales[kind][i])
                out[f"{nm}_sn_{kind}"][i] = line_snr_from_fit(fit)[0]
                out[f"{nm}_sigma_{kind}"][i] = fit["sigma"]
                out[f"{nm}_mu_{kind}"][i] = fit["mu"]
    return out


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
