"""Tests for augmentation, provenance and the log-grid redshift shift."""

from __future__ import annotations

import numpy as np
import pytest

from specsr.data.augment import (
    AugmentationConfig,
    augment_pair,
    shift_redshift_on_log_grid,
)
from specsr.data.grid import DEFAULT_GRID

G = DEFAULT_GRID
W = G.centers()
N = W.size


def _line_at(lam_obs, amp=50.0, R=1000.0):
    sigma = lam_obs / R / 2.355
    return 1.0 + amp * np.exp(-0.5 * ((W - lam_obs) / sigma) ** 2)


def _peak(flux, valid):
    return W[np.nanargmax(np.where(valid, flux, -np.inf))]


# --------------------------------------------------------------------------
# the redshift shift
# --------------------------------------------------------------------------


@pytest.mark.parametrize("z0,z1", [(3.0, 3.3), (3.0, 2.6), (3.0, 4.0), (5.0, 5.2)])
def test_shift_moves_a_line_to_the_right_wavelength(z0, z1):
    lam_rest = 0.5007
    f = _line_at(lam_rest * (1 + z0))
    v = np.ones(N, bool)
    sf, sv = shift_redshift_on_log_grid(f, v, z0, z1, G)
    # within one grid sample of the expected observed wavelength
    expected = lam_rest * (1 + z1)
    assert abs(_peak(sf, sv) - expected) < expected / G.resolving_power * 1.5


def test_shift_is_uniform_in_pixels_on_a_log_grid():
    """The property that motivates the log grid: a redshift offset is the same
    number of samples everywhere, not a stretch."""
    v = np.ones(N, bool)
    z0, z1 = 3.0, 3.4
    step = np.log(G.lambda_max / G.lambda_min) / (N - 1)
    predicted = np.log((1 + z1) / (1 + z0)) / step

    shifts = []
    for lam_rest in (0.3727, 0.5007, 0.6563):
        f = _line_at(lam_rest * (1 + z0))
        sf, sv = shift_redshift_on_log_grid(f, v, z0, z1, G)
        shifts.append(np.nanargmax(np.where(sv, sf, -np.inf)) - np.argmax(f))
    # every line moves by the same number of samples, matching the prediction
    assert max(shifts) - min(shifts) <= 1
    assert abs(np.mean(shifts) - predicted) <= 1.5


def test_shift_marks_samples_moved_in_from_outside_as_invalid():
    """Nothing was measured beyond the grid edge; it must not be invented."""
    f = _line_at(2.0)
    v = np.ones(N, bool)
    sf, sv = shift_redshift_on_log_grid(f, v, 3.0, 4.0, G)
    assert not sv.all()
    assert np.isnan(sf[~sv]).all()
    # the invalid region is at one end, not scattered
    assert sv[N // 2]


def test_shift_does_not_manufacture_data_across_a_gap():
    """A fractional shift blends neighbours; if either is invalid the result is
    invalid, or the interpolation would fill a gap with plausible-looking flux."""
    f = _line_at(2.0)
    v = np.ones(N, bool)
    v[3000:3050] = False
    sf, sv = shift_redshift_on_log_grid(f, v, 3.0, 3.05, G)
    # the shifted gap must be at least as wide as the original
    assert (~sv).sum() >= 50
    assert np.isnan(sf[~sv]).all()


def test_shift_propagates_errors_in_quadrature():
    f = _line_at(2.0)
    v = np.ones(N, bool)
    e = np.full(N, 0.1)
    sf, sv, se = shift_redshift_on_log_grid(f, v, 3.0, 3.2, G, err=e)
    assert se.shape == sf.shape
    good = sv & np.isfinite(se)
    assert good.any()
    # a blend of two samples with sigma=0.1 lies between 0.0707 and 0.1
    assert (se[good] <= 0.1 + 1e-9).all()
    assert (se[good] >= 0.1 / np.sqrt(2) - 1e-9).all()


def test_zero_shift_is_identity():
    f = _line_at(2.0)
    v = np.ones(N, bool)
    sf, sv = shift_redshift_on_log_grid(f, v, 3.0, 3.0, G)
    assert np.allclose(sf[sv], f[sv])


# --------------------------------------------------------------------------
# augmentation and provenance
# --------------------------------------------------------------------------


@pytest.fixture
def pair():
    f = _line_at(2.0)
    return dict(
        flux_low=f, flux_low_err=np.full(N, 0.1), valid_low=np.ones(N, bool),
        flux_high=f.copy(), flux_high_err=np.full(N, 0.1), valid_high=np.ones(N, bool),
        z=3.0, grid=G,
    )


def test_every_row_records_its_parent(pair):
    """Provenance must be explicit. Without it a split cannot be drawn over
    galaxies without reconstructing the grouping from sky coordinates."""
    rows = augment_pair(parent_id=7, config=AugmentationConfig(n_aug=5), **pair)
    assert len(rows) == 6
    assert {r["parent_id"] for r in rows} == {7}
    assert sum(r["is_original"] for r in rows) == 1


def test_first_row_is_the_untouched_original(pair):
    rows = augment_pair(parent_id=0, config=AugmentationConfig(n_aug=3), **pair)
    assert rows[0]["is_original"]
    assert np.allclose(rows[0]["flux_low"], pair["flux_low"])
    assert rows[0]["z"] == pair["z"]


def test_both_members_of_a_pair_get_the_same_redshift_shift(pair):
    """They are the same galaxy. Shifting them differently would teach a
    wavelength mapping that does not exist."""
    rows = augment_pair(parent_id=1, config=AugmentationConfig(n_aug=4, seed=3), **pair)
    for r in rows[1:]:
        lo_peak = _peak(r["flux_low"], r["valid_low"])
        hi_peak = _peak(r["flux_high"], r["valid_high"])
        assert abs(lo_peak - hi_peak) < 2.0 / G.resolving_power


def test_noise_is_independent_between_low_and_high(pair):
    """Their measurement noise genuinely is independent."""
    rows = augment_pair(parent_id=1, config=AugmentationConfig(n_aug=1, seed=5), **pair)
    r = rows[1]
    both = r["valid_low"] & r["valid_high"]
    assert not np.allclose(r["flux_low"][both], r["flux_high"][both])


def test_augmentation_is_reproducible_and_parent_scoped(pair):
    a = augment_pair(parent_id=3, config=AugmentationConfig(n_aug=3, seed=11), **pair)
    b = augment_pair(parent_id=3, config=AugmentationConfig(n_aug=3, seed=11), **pair)
    c = augment_pair(parent_id=4, config=AugmentationConfig(n_aug=3, seed=11), **pair)
    assert [r["z"] for r in a] == [r["z"] for r in b]
    # a different galaxy draws different realizations, so adding galaxies does
    # not perturb the ones already built
    assert [r["z"] for r in a][1:] != [r["z"] for r in c][1:]


def test_added_noise_enters_the_error_budget(pair):
    rows = augment_pair(parent_id=1, config=AugmentationConfig(n_aug=1, noise_frac=0.1, seed=2),
                        **pair)
    r = rows[1]
    good = r["valid_low"] & np.isfinite(r["flux_low_err"])
    assert (r["flux_low_err"][good] >= 0.1 - 1e-9).all()


def test_redshift_never_goes_negative(pair):
    rows = augment_pair(parent_id=1,
                        config=AugmentationConfig(n_aug=50, sigma_z=5.0, seed=0), **pair)
    assert all(r["z"] >= 0.0 for r in rows)


def test_validity_is_carried_through_augmentation(pair):
    p = dict(pair)
    p["valid_high"] = pair["valid_high"].copy()
    p["valid_high"][2000:2100] = False
    rows = augment_pair(parent_id=1, config=AugmentationConfig(n_aug=3, seed=1), **p)
    for r in rows[1:]:
        assert not r["valid_high"].all(), "gap must survive augmentation"
        assert np.isnan(r["flux_high"][~r["valid_high"]]).all()
