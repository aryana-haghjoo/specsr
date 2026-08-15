"""Tests for the common wavelength grid and flux-conserving resampling.

The central claim these defend: the grid must not degrade the data it carries.
The original pipeline's linear grid downsampled the high-resolution reference
before training, so the grid rather than the model capped achievable sharpness.
"""

from __future__ import annotations

import numpy as np
import pytest

from specsr.data.grid import (
    DEFAULT_GRID,
    LogWavelengthGrid,
    _bin_edges_from_centers,
    resample_flux_conserving,
    sampling_report,
)

# Native DR4 sampling, measured from EXTRACT3PIX1D in the x1d products.
NATIVE = {
    "prism": (0.603, 5.452, 0.00613),
    "G140M": (0.700, 2.200, 0.00064),
    "G235M": (1.660, 3.999, 0.00107),
    "G395M": (2.870, 5.479, 0.00180),
}


def _integrate(w, f):
    return np.nansum(f * np.diff(_bin_edges_from_centers(w)))


def _line_spectrum(w, lines=((1.10, 40.0), (1.35, 60.0), (1.70, 25.0), (2.05, 50.0)), R=1000.0):
    """Continuum plus Gaussian emission lines of instrumental width."""
    f = np.ones_like(w)
    for c, amp in lines:
        sigma = c / R / 2.355
        f = f + amp * np.exp(-0.5 * ((w - c) / sigma) ** 2)
    return f


# --------------------------------------------------------------------------
# grid geometry
# --------------------------------------------------------------------------


def test_grid_is_constant_resolving_power():
    """A log grid holds lambda/dlambda fixed; that is the whole point."""
    g = DEFAULT_GRID
    c = g.centers()
    ratio = c[:-1] / np.diff(c)
    assert np.allclose(ratio, ratio[0], rtol=1e-3)
    # and that constant is the requested resolving power
    assert ratio[0] == pytest.approx(g.resolving_power, rel=1e-2)


def test_grid_spans_requested_range():
    g = DEFAULT_GRID
    c = g.centers()
    assert c[0] == pytest.approx(g.lambda_min)
    assert c[-1] == pytest.approx(g.lambda_max)
    assert c.size == g.n_samples


def test_edges_bracket_centers_and_are_monotonic():
    e = DEFAULT_GRID.edges()
    c = DEFAULT_GRID.centers()
    assert e.size == c.size + 1
    assert np.all(np.diff(e) > 0)
    assert np.all(e[:-1] <= c) and np.all(c <= e[1:])


def test_default_grid_downsamples_nothing():
    """The grid must be at least as fine as the finest native sampling anywhere,
    or it silently caps resolution — the defect this module exists to fix."""
    finest = max(hi / dlam for (_, hi, dlam) in NATIVE.values())
    assert DEFAULT_GRID.resolving_power >= finest, (
        f"grid R={DEFAULT_GRID.resolving_power} is below the finest native "
        f"sampling {finest:.0f}"
    )


def test_old_linear_grid_would_downsample():
    """Regression guard on the original defect, so it cannot quietly return."""
    old_dlam = 4.0 / 2499  # linspace(1, 5, 2500)
    # G140M native sampling is finer than the old grid step, i.e. downsampled.
    assert NATIVE["G140M"][2] < old_dlam
    assert old_dlam / NATIVE["G140M"][2] > 2.0  # by more than a factor of two


def test_samples_per_resolution_element_meets_nyquist():
    g = DEFAULT_GRID
    for lam in (1.0, 2.0, 3.0, 4.0, 5.0):
        n = g.samples_per_resolution_element(lam, R_instrument=1000.0)
        assert n >= 2.0, f"undersampled at {lam} um: {n:.2f} samples/element"


def test_sampling_report_flags_a_bad_grid():
    bad = LogWavelengthGrid(1.0, 5.3, resolving_power=1000.0)
    assert "DOWNSAMPLES" in sampling_report(NATIVE, bad)
    assert "DOWNSAMPLES" not in sampling_report(NATIVE, DEFAULT_GRID)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def test_resample_conserves_integrated_flux_when_upsampling():
    w_in = np.linspace(1.0, 5.3, 700)
    f_in = _line_spectrum(w_in, R=200.0)
    w_out = DEFAULT_GRID.centers()
    f_out = resample_flux_conserving(w_in, f_in, w_out)
    assert _integrate(w_out, f_out) == pytest.approx(_integrate(w_in, f_in), rel=2e-3)


def test_resample_preserves_line_peaks_far_better_than_the_old_grid():
    """The measurement that justifies the change.

    On native G140M sampling with R~1000 lines, the old linear grid retains only
    ~79% of an emission line's peak amplitude; the log grid retains ~98%.
    """
    w_nat = np.arange(1.0, 2.2, NATIVE["G140M"][2])
    f_nat = _line_spectrum(w_nat)
    peak_in = np.nanmax(f_nat) - 1.0

    def peak_retained(w_out):
        m = (w_out >= w_nat[0]) & (w_out <= w_nat[-1])
        f_out = resample_flux_conserving(w_nat, f_nat, w_out[m])
        return (np.nanmax(f_out) - 1.0) / peak_in

    old = peak_retained(np.linspace(1.0, 5.0, 2500))
    new = peak_retained(DEFAULT_GRID.centers())
    assert new > 0.95, f"log grid should retain >95% of peak, got {new:.1%}"
    assert new > old + 0.15, f"log grid ({new:.1%}) must clearly beat old ({old:.1%})"


def test_resample_propagates_errors():
    w_in = np.linspace(1.0, 5.3, 500)
    f_in = _line_spectrum(w_in, R=200.0)
    e_in = np.full_like(f_in, 0.1)
    f_out, e_out = resample_flux_conserving(w_in, f_in, DEFAULT_GRID.centers(), err_in=e_in)
    assert e_out.shape == f_out.shape
    assert np.all(e_out[np.isfinite(e_out)] > 0)


def test_resample_marks_uncovered_output_bins():
    """Output bins outside the input range must be flagged, not silently zero."""
    w_in = np.linspace(2.0, 3.0, 200)
    f_in = np.ones_like(w_in)
    w_out = DEFAULT_GRID.centers()
    f_out = resample_flux_conserving(w_in, f_in, w_out)
    assert np.isnan(f_out[w_out < 1.9]).all()
    assert np.isnan(f_out[w_out > 3.1]).all()
    covered = (w_out > 2.05) & (w_out < 2.95)
    assert np.isfinite(f_out[covered]).all()


def test_resample_handles_unsorted_input():
    w = np.linspace(1.0, 5.0, 100)
    f = _line_spectrum(w, R=100.0)
    order = np.argsort(-w)  # reversed
    out_sorted = resample_flux_conserving(w, f, DEFAULT_GRID.centers())
    out_rev = resample_flux_conserving(w[order], f[order], DEFAULT_GRID.centers())
    np.testing.assert_allclose(out_sorted, out_rev, equal_nan=True)


def test_resample_ignores_nans_in_input():
    w = np.linspace(1.0, 5.0, 400)
    f = _line_spectrum(w, R=200.0)
    f[100:110] = np.nan
    out = resample_flux_conserving(w, f, DEFAULT_GRID.centers())
    assert np.isfinite(out).sum() > 0.9 * out.size


def test_resample_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        resample_flux_conserving(np.linspace(1, 2, 10), np.ones(5), np.linspace(1, 2, 20))
