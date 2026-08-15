"""Tests for raw JADES discovery and pairing.

Most run against synthetic filenames so they work without the 88 GB raw tree.
Tests needing the real data are marked ``needs_data`` and skip when
``SPECSR_JADES_ROOT`` is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from specsr.data.ingest import (
    GRATINGS,
    PRISM,
    SpectrumFile,
    discover_spectra,
    group_by_target,
    read_spectrum,
)

needs_data = pytest.mark.skipif(
    not os.environ.get("SPECSR_JADES_ROOT"),
    reason="raw JADES tree not configured (SPECSR_JADES_ROOT)",
)


def _make_tree(root: Path, spec: dict[tuple[str, str, str], list[str]]) -> Path:
    """Create empty files mirroring the real layout.

    ``spec`` maps ``(field, disperser, tier)`` to a list of target ids.
    """
    for (field, disperser, tier), ids in spec.items():
        d = root / field / "spectra" / disperser / tier
        d.mkdir(parents=True, exist_ok=True)
        for tid in ids:
            name = f"hlsp_jades_jwst_nirspec_{tier}-{tid}_{disperser}_v1.0_x1d.fits"
            (d / name).touch()
    return root


# --------------------------------------------------------------------------
# filename parsing and identity
# --------------------------------------------------------------------------


def test_discover_parses_field_tier_id_disperser(tmp_path):
    _make_tree(tmp_path, {("goods-s", PRISM, "goods-s-deephst"): ["00002332"]})
    files = discover_spectra(tmp_path)
    assert len(files) == 1
    f = files[0]
    assert (f.field, f.tier, f.target_id, f.disperser) == (
        "goods-s", "goods-s-deephst", "00002332", PRISM,
    )
    assert f.is_prism


def test_galaxy_identity_excludes_tier():
    """A target observed in two tiers is ONE galaxy.

    Keying on tier would split it into two, and those halves would land on
    opposite sides of a train/test split -- the same duplicate-object leak the
    DR3 sample had.
    """
    a = SpectrumFile(Path("a"), "goods-n", "goods-n-mediumhst", "00000721", PRISM)
    b = SpectrumFile(Path("b"), "goods-n", "goods-n-mediumjwst", "00000721", PRISM)
    assert a.key == b.key == ("goods-n", "00000721")


def test_multi_tier_target_groups_as_one_galaxy(tmp_path):
    _make_tree(tmp_path, {
        ("goods-n", PRISM, "goods-n-mediumhst"): ["00000721"],
        ("goods-n", PRISM, "goods-n-mediumjwst"): ["00000721"],
        ("goods-n", GRATINGS[0], "goods-n-mediumhst"): ["00000721"],
    })
    groups = group_by_target(discover_spectra(tmp_path))
    assert len(groups) == 1, "multi-tier target must not split into two galaxies"
    (key,) = groups
    assert len(groups[key][PRISM]) == 2, "both tier observations must be retained"


def test_unparseable_filenames_are_skipped(tmp_path):
    d = tmp_path / "goods-s" / "spectra" / PRISM / "tier"
    d.mkdir(parents=True)
    (d / "index.html").touch()
    (d / "random_file_x1d.fits").touch()
    assert discover_spectra(tmp_path) == []


def test_discover_ignores_unrequested_dispersers(tmp_path):
    _make_tree(tmp_path, {
        ("goods-s", PRISM, "t"): ["1"],
        ("goods-s", "f290lp-g395h", "t"): ["1"],   # high-res grating, not used
    })
    assert {f.disperser for f in discover_spectra(tmp_path)} == {PRISM}


# --------------------------------------------------------------------------
# pairing requirements
# --------------------------------------------------------------------------


def test_target_without_prism_is_dropped(tmp_path):
    _make_tree(tmp_path, {("goods-s", GRATINGS[0], "t"): ["1"]})
    assert group_by_target(discover_spectra(tmp_path), require_prism=True) == {}


def test_target_without_enough_gratings_is_dropped(tmp_path):
    _make_tree(tmp_path, {("goods-s", PRISM, "t"): ["1"]})
    files = discover_spectra(tmp_path)
    assert group_by_target(files, require_gratings=1) == {}


def test_require_all_three_gratings(tmp_path):
    spec = {("goods-s", PRISM, "t"): ["1", "2"]}
    for g in GRATINGS:
        spec[("goods-s", g, "t")] = ["1"]          # only target 1 gets all three
    spec[("goods-s", GRATINGS[0], "t")] = ["1", "2"]
    groups3 = group_by_target(discover_spectra(tmp_path := _make_tree(tmp_path, spec)),
                              require_gratings=3)
    groups1 = group_by_target(discover_spectra(tmp_path), require_gratings=1)
    assert set(k[1] for k in groups3) == {"1"}
    assert set(k[1] for k in groups1) == {"1", "2"}


def test_same_target_in_two_fields_is_two_galaxies(tmp_path):
    """Field is part of the identity; the same id in GOODS-N and GOODS-S is not
    the same object."""
    spec = {}
    for field in ("goods-n", "goods-s"):
        spec[(field, PRISM, f"{field}-t")] = ["00000001"]
        spec[(field, GRATINGS[0], f"{field}-t")] = ["00000001"]
    groups = group_by_target(discover_spectra(_make_tree(tmp_path, spec)))
    assert len(groups) == 2


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------


@needs_data
def test_real_tree_pairs_and_dedups():
    from specsr.paths import release_dir

    files = discover_spectra(release_dir("DR4"))
    assert len(files) > 10_000

    groups = group_by_target(files, require_prism=True, require_gratings=3)
    assert len(groups) > 4_000

    # every group must be a single galaxy identity
    for (field, tid), by_disp in groups.items():
        assert field in ("goods-n", "goods-s")
        assert tid.isdigit()
        assert PRISM in by_disp


@needs_data
def test_real_spectrum_read_has_expected_units_and_coords():
    from specsr.paths import release_dir

    files = [f for f in discover_spectra(release_dir("DR4")) if f.is_prism]
    s = read_spectrum(files[0].path)
    assert s["wavelength"].ndim == 1
    assert s["flux"].shape == s["wavelength"].shape
    assert (s["wavelength"] > 0).all()
    assert list(s["wavelength"]) == sorted(s["wavelength"])
    assert "AA-1" in s["bunit"]          # f_lambda
    assert -90 <= s["dec"] <= 90


@needs_data
def test_dispersers_of_one_target_agree_on_sky_position():
    """If they did not, (field, target_id) would be the wrong identity."""
    from specsr.paths import release_dir

    groups = group_by_target(discover_spectra(release_dir("DR4")), require_gratings=3)
    key = sorted(groups)[0]
    coords = [
        (read_spectrum(fs[0].path)["ra"], read_spectrum(fs[0].path)["dec"])
        for fs in groups[key].values()
    ]
    ras, decs = zip(*coords, strict=True)
    assert max(ras) - min(ras) < 1e-4
    assert max(decs) - min(decs) < 1e-4


# --------------------------------------------------------------------------
# Augmentation policy
# --------------------------------------------------------------------------


def test_augment_train_only_leaves_no_row_unused(tmp_path, monkeypatch):
    """A built product must contain no row that the split cannot use.

    Augmenting every galaxy and filtering at split time is behaviourally the
    same, but it leaves thousands of rows in the file that nothing may read —
    and a row that must never be read is a trap for the next person. After the
    build, `train + val` must account for every row in the file.
    """
    import numpy as np

    from specsr.data.augment import AugmentationConfig, augment_pair
    from specsr.data.grid import DEFAULT_GRID
    from specsr.data.splits import make_group_split_3way

    n_gal, n_aug, train_frac = 40, 20, 0.8
    ids = np.arange(n_gal)

    tr_g, va_g, _ = make_group_split_3way(
        ids, np.ones(n_gal, bool), train_frac, 1.0 - train_frac, allow_empty_test=True
    )
    train_ids = {int(ids[i]) for i in tr_g}

    L = DEFAULT_GRID.n_samples
    rows = []
    for pid in ids:
        cfg = AugmentationConfig(n_aug=n_aug if int(pid) in train_ids else 0)
        rows.extend(
            augment_pair(
                flux_low=np.ones(L), flux_low_err=np.ones(L), valid_low=np.ones(L, bool),
                flux_high=np.ones(L), flux_high_err=np.ones(L), valid_high=np.ones(L, bool),
                z=3.0, parent_id=int(pid), grid=DEFAULT_GRID, config=cfg,
            )
        )

    groups = np.array([r["parent_id"] for r in rows])
    orig = np.array([r["is_original"] for r in rows], dtype=bool)
    tr, va, _ = make_group_split_3way(
        groups, orig, train_frac, 1.0 - train_frac, allow_empty_test=True
    )

    assert len(tr) + len(va) == len(rows), (
        f"{len(rows) - len(tr) - len(va)} rows belong to neither split"
    )
    # Held-out galaxies contribute exactly their real spectrum, nothing else.
    held = set(np.unique(groups[va]).tolist())
    assert held == set(ids.tolist()) - train_ids
    for pid in held:
        assert int((groups == pid).sum()) == 1
