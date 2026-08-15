"""``SpecSRPipeline`` must reproduce the trained chain, not merely run.

The pipeline is the package's headline interface -- the thing the README tells
people to call. It was advertised for some time before it existed, so the first
job of these tests is that it exists and imports.

The second job is harder and matters more. Every failure mode here is silent:
the redshift bounds, the normalisation moments, the SR2 channel order and the
``log_var``/``log_sigma`` conversion all produce a plausible spectrum and a
plausible redshift when they are wrong. Nothing raises. So these assert on
*numbers* against the known-good run 5 chain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
RUN5 = REPO / "checkpoints/checkpoints_run5_20260728"
DATASET = REPO / "data" / "paired_DR4_logR.npz"

# Measured on the first 64 rows with the run 5 chain. The chain's own validation
# sigma_NMAD is ~0.016; anything near 0.5 means the redshift decoding is broken,
# which is exactly what a wrong z_min_n/z_max_n produces.
MAX_SIGMA_NMAD = 0.05

needs_artifacts = pytest.mark.skipif(
    not (DATASET.exists() and RUN5.exists()),
    reason="dataset or run 5 checkpoints not available",
)


def test_pipeline_is_importable_from_the_documented_path():
    """``from specsr.inference import SpecSRPipeline`` is what the README says."""
    from specsr.inference import SpecSRPipeline

    assert hasattr(SpecSRPipeline, "from_checkpoints")


@needs_artifacts
def test_pipeline_recovers_redshift():
    """End-to-end redshifts must match the truth, not merely be finite.

    Regression test for a real bug: the transform's bounds are not stored in
    older head checkpoints, and defaulting them to ``[-3, 3]`` instead of the
    training split's ``[-1.78, 6.25]`` put z~3.9 objects at z~0.7. Every value
    was finite, ordered and physically plausible.
    """
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(RUN5, dataset=DATASET)
    with np.load(DATASET, allow_pickle=True) as d:
        flux_low = np.asarray(d["flux_low"][:64], dtype=np.float32)
        z_true = np.asarray(d["z"][:64], dtype=np.float32)

    res = pipe(flux_low)
    dz = (res.z - z_true) / (1.0 + z_true)
    sigma_nmad = float(1.4826 * np.median(np.abs(dz - np.median(dz))))

    assert sigma_nmad < MAX_SIGMA_NMAD, (
        f"sigma_NMAD {sigma_nmad:.4f} exceeds {MAX_SIGMA_NMAD}. The chain runs but "
        "decodes redshift wrongly -- check the RedshiftTransform bounds against the "
        "training split the head was fitted on."
    )


@needs_artifacts
def test_pipeline_refuses_to_guess_redshift_bounds():
    """Missing bounds must raise, not fall back on a default.

    A default here does not fail; it returns confident wrong answers. Refusing is
    the only safe behaviour for a checkpoint that predates stored bounds.
    """
    from specsr.inference import SpecSRPipeline

    with np.load(DATASET, allow_pickle=True) as d:
        wave = np.asarray(d["wavelength_high"], dtype=np.float32)

    with pytest.raises(ValueError, match="bounds"):
        SpecSRPipeline.from_checkpoints(RUN5, wavelength=wave)  # no dataset=


@needs_artifacts
def test_pipeline_output_is_in_physical_units():
    """Outputs come back on the scale of the input, not in normalised units.

    The models work in per-spectrum standardised space. If de-normalisation is
    skipped the spectra look entirely reasonable -- just off by ~20 orders of
    magnitude, which is easy to miss on a plot with no axis labels.
    """
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(RUN5, dataset=DATASET)
    with np.load(DATASET, allow_pickle=True) as d:
        flux_low = np.asarray(d["flux_low"][:16], dtype=np.float32)
        flux_high = np.asarray(d["flux_high"][:16], dtype=np.float64)

    res = pipe(flux_low)
    ratio = np.median(np.abs(res.sr1)) / np.median(np.abs(flux_high))
    assert 0.1 < ratio < 10.0, (
        f"SR1 output is {ratio:.3g}x the HR flux scale; it is probably still in "
        "normalised units rather than physical ones"
    )


# --- The public entry path: no dataset, arbitrary input grid -----------------
#
# Both behaviours below exist so a user with a spectrum and nothing else can run
# the model. Both fail silently if broken: a wrong default grid still returns a
# spectrum of the right shape, and a skipped resample still returns plausible
# flux -- just misaligned with the wavelengths the caller believes it is on.

RELEASE = REPO / "checkpoints" / "release"

needs_release = pytest.mark.skipif(
    not RELEASE.exists(), reason="release checkpoints not available"
)


@needs_release
def test_from_checkpoints_defaults_to_the_project_grid_without_a_dataset():
    """A public user has no ``paired_DR4_logR.npz``; the grid must not require one."""
    from specsr.data.grid import DEFAULT_GRID
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(RELEASE)
    assert pipe.wavelength.shape == (6671,)
    np.testing.assert_allclose(pipe.wavelength, DEFAULT_GRID.centers(), rtol=1e-6)


@needs_release
@needs_artifacts
def test_default_grid_matches_the_built_dataset_exactly():
    """The fallback is only safe while it equals what the dataset ships."""
    from specsr.data.grid import DEFAULT_GRID

    with np.load(str(DATASET), allow_pickle=True) as npz:
        np.testing.assert_allclose(
            DEFAULT_GRID.centers(), np.asarray(npz["wavelength_high"]), rtol=1e-6
        )


@needs_release
def test_wavelength_argument_resamples_an_off_grid_spectrum():
    """Passing ``wavelength=`` must regrid, and agree with the on-grid answer."""
    from specsr.data.grid import resample_flux_conserving
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(RELEASE)

    # A real spectrum, not synthetic noise: the redshift head is only stable on
    # input that looks like a galaxy, so noise would make this test measure the
    # head's instability rather than the resampling.
    sample = REPO / "tutorials_for_user" / "sample_one_spectrum.npz"
    if not sample.exists():
        pytest.skip("tutorial sample spectrum not available")
    with np.load(str(sample), allow_pickle=True) as npz:
        on_grid = np.asarray(npz["flux_low"], dtype=float)

    # A plausible "my own reduction" grid: linear, coarser, narrower.
    user_w = np.linspace(1.05, 5.1, 1800)
    user_f = resample_flux_conserving(pipe.wavelength.astype(float), on_grid, user_w)

    out_native = pipe(on_grid)
    out_user = pipe(user_f, user_w)

    assert out_user.sr2.shape == out_native.sr2.shape == (1, 6671)
    # Not identical -- a round trip through a coarser grid loses information --
    # but the same object, so the redshifts must stay close. Measured difference
    # on this spectrum is ~0.014; 0.2 leaves room for hardware variation while
    # still failing loudly if the resample is skipped or misaligned.
    assert abs(float(out_user.z[0]) - float(out_native.z[0])) < 0.2


@needs_release
def test_mismatched_wavelength_length_raises_instead_of_broadcasting():
    """The silent version of this bug is a spectrum shifted against its own axis."""
    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_checkpoints(RELEASE)
    flux = np.ones(1800)
    with pytest.raises(ValueError, match="wavelength"):
        pipe(flux, np.linspace(1.05, 5.1, 1797))


@needs_release
def test_release_zhead_carries_its_redshift_bounds():
    """Without stored bounds the pipeline cannot decode z without the dataset.

    Heads trained before 2026-07-29 did not store them. The released head must,
    or every public user hits a hard error on a private file they cannot get.
    """
    ck = torch.load(RELEASE / "best_zhead.pth", map_location="cpu", weights_only=False)
    assert "z_min_n" in ck and "z_max_n" in ck
    assert ck["z_min_n"] < ck["z_max_n"]
