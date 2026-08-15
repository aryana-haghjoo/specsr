"""The published release must work for someone who has only the Hub.

Everything else in this suite tests the code against local archives, where the
weights and the SR1 config sit side by side in one directory. The Hub does not
work that way: ``hf_hub_download`` places *only the file you asked for* into the
snapshot, so any code that looks for a sibling file finds nothing.

That gap shipped. ``from_pretrained()`` derived the SR1 config path with
``Path(sr1_ckpt).with_name("config_logR.yaml")``, which resolves locally and
raises ``FileNotFoundError`` for every public user. The full suite passed the
whole time, because no test ever exercised the download path's directory layout.

So this module tests two things nothing else does:

1. **Offline** -- the resolution logic, against a simulated Hub layout where each
   file lands in its own directory. These always run.
2. **Against the real Hub** -- that the *published artifacts* are the right ones.
   This is a different failure class from "the code is wrong": the code can be
   perfect and the uploaded checkpoint still be a version that cannot decode a
   redshift. Opt in with ``SPECSR_TEST_HUB=1``.

Run the second group deliberately, before and after every weights upload::

    SPECSR_TEST_HUB=1 pytest tests/test_public_release.py -v
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "checkpoints" / "release"
SAMPLE = REPO / "tutorials_for_user" / "sample_one_spectrum.npz"

# What the pipeline must be able to fetch to build a full chain.
CHAIN_NAMES = ("sr1", "sr1_config", "zhead", "sr2")

# Hub path -> the file in checkpoints/release/ that stands in for it.
_HUB_TO_LOCAL = {
    "sr1/best_sr1.pth": "best_superres_model.pth",
    "sr1/config_logR.yaml": "config_logR.yaml",
    "zhead/best_zhead.pth": "best_zhead.pth",
    "sr2/best_sr2.pth": "best_sr2.pth",
}

needs_release = pytest.mark.skipif(
    not RELEASE.exists(), reason="checkpoints/release not available"
)


# --------------------------------------------------------------------------
# Offline: the resolution logic
# --------------------------------------------------------------------------


def test_every_artifact_the_chain_needs_is_registered():
    """A name the pipeline requests but the registry lacks is a KeyError at runtime."""
    from specsr.checkpoints import _REGISTRY, available_checkpoints

    for name in CHAIN_NAMES:
        assert name in _REGISTRY, f"{name!r} missing from the checkpoint registry"
        assert name in available_checkpoints()


@needs_release
def test_local_checkpoint_dir_resolves_the_sr1_config():
    """``SPECSR_CHECKPOINT_DIR`` must serve the config, not just the weights."""
    from specsr.checkpoints import get_checkpoint


    old = os.environ.get("SPECSR_CHECKPOINT_DIR")
    os.environ["SPECSR_CHECKPOINT_DIR"] = str(RELEASE)
    try:
        assert get_checkpoint("sr1_config").exists()
    finally:
        if old is None:
            os.environ.pop("SPECSR_CHECKPOINT_DIR", None)
        else:
            os.environ["SPECSR_CHECKPOINT_DIR"] = old


@needs_release
def test_from_pretrained_works_when_each_file_lands_in_its_own_directory(
    tmp_path, monkeypatch
):
    """The regression test for the shipped bug.

    Simulates the Hub: every download goes to a separate directory, so nothing is
    beside anything else. Code that infers the config from the weights' path
    fails here and passes against a local archive -- which is exactly how this
    reached users.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")

    from specsr.inference import SpecSRPipeline

    monkeypatch.delenv("SPECSR_CHECKPOINT_DIR", raising=False)

    def fake_download(repo_id, filename, **kwargs):
        src = RELEASE / _HUB_TO_LOCAL[filename]
        # One directory per file: the defining property of a Hub snapshot here.
        dest_dir = tmp_path / filename.replace("/", "__")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(filename).name
        if not dest.exists():
            shutil.copy2(src, dest)
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    pipe = SpecSRPipeline.from_pretrained()
    assert pipe.wavelength.shape == (6671,)

    # And it must actually run, not merely construct.
    flux = np.zeros(6671, dtype=float)
    flux[3000] = 1.0
    out = pipe(flux)
    assert out.sr2.shape == (1, 6671)


@needs_release
def test_release_head_decodes_redshift_without_the_training_dataset(monkeypatch):
    """A public user has no ``paired_DR4_logR.npz``; the head must be self-sufficient."""
    from specsr.inference import SpecSRPipeline

    monkeypatch.setenv("SPECSR_CHECKPOINT_DIR", str(RELEASE))
    pipe = SpecSRPipeline.from_checkpoints(RELEASE)   # note: no dataset=

    if not SAMPLE.exists():
        pytest.skip("tutorial sample spectrum not available")
    with np.load(str(SAMPLE), allow_pickle=True) as npz:
        flux = np.asarray(npz["flux_low"], dtype=float)
        z_true = float(npz["z"])

    z = float(pipe(flux).z[0])
    # Loose on purpose: this asserts the decode is sane, not that the model is
    # accurate. Wrong bounds put z~3 objects near z~0.7 -- a factor of five.
    assert 0.0 < z < 15.0
    assert abs(z - z_true) < 3.0


# --------------------------------------------------------------------------
# Against the real Hub: are the *published artifacts* right?
# --------------------------------------------------------------------------

# Opt-in, following `test_ingest.py`'s treatment of `SPECSR_JADES_ROOT`: the
# marker labels these, an env var actually gates them. They need network, and
# they assert on artifacts published outside this repo, so a red run here means
# "re-upload the weights", not "the code regressed" -- a distinction CI cannot
# act on. Run them deliberately, before and after publishing:
#
#     SPECSR_TEST_HUB=1 pytest tests/test_public_release.py -v
needs_hub = pytest.mark.skipif(
    not os.environ.get("SPECSR_TEST_HUB"),
    reason="Hub round-trip not requested (set SPECSR_TEST_HUB=1)",
)


@needs_hub
def test_hub_repo_is_public_and_readable_anonymously():
    """A private weights repo makes every quickstart in the docs a 401."""
    hub = pytest.importorskip("huggingface_hub")

    from specsr.checkpoints import DEFAULT_REPO

    info = hub.HfApi(token=False).model_info(DEFAULT_REPO)
    assert info.private is False, f"{DEFAULT_REPO} is private"


@needs_hub
def test_every_registered_path_exists_on_the_hub():
    """The registry and the uploaded layout drift independently; pin them together."""
    hub = pytest.importorskip("huggingface_hub")

    from specsr.checkpoints import _REGISTRY, DEFAULT_REPO

    published = {s.rfilename for s in hub.HfApi(token=False).model_info(DEFAULT_REPO).siblings}
    missing = [_REGISTRY[n] for n in CHAIN_NAMES if _REGISTRY[n] not in published]
    assert not missing, f"registered but not on the Hub: {missing}"


@needs_hub
def test_published_head_carries_its_redshift_bounds():
    """The uploaded head must be the stamped one.

    Heads trained before 2026-07-29 do not store ``z_min_n``/``z_max_n``, and the
    pipeline refuses to guess them -- correctly, since guessing returns confident
    wrong redshifts rather than an error. But that means publishing an unstamped
    head makes ``from_pretrained()`` raise for everyone without the training data.
    """
    from specsr.checkpoints import get_checkpoint

    ck = torch.load(get_checkpoint("zhead"), map_location="cpu", weights_only=False)
    assert "z_min_n" in ck and "z_max_n" in ck, (
        "the published zhead predates stored redshift bounds; upload "
        "checkpoints/release/best_zhead.pth"
    )


@needs_hub
def test_end_to_end_from_the_hub_alone():
    """The whole public path: no local weights, no dataset, no configuration."""
    from specsr.inference import SpecSRPipeline

    if not SAMPLE.exists():
        pytest.skip("tutorial sample spectrum not available")

    pipe = SpecSRPipeline.from_pretrained()
    with np.load(str(SAMPLE), allow_pickle=True) as npz:
        flux = np.asarray(npz["flux_low"], dtype=float)
        z_true = float(npz["z"])

    out = pipe(flux)
    assert out.sr2.shape == (1, 6671)
    assert np.isfinite(out.sr2).all()
    assert abs(float(out.z[0]) - z_true) < 3.0
