"""The bundled tutorial spectra must come from the held-out split.

The tutorials print real numbers -- redshift outlier rates, flux recovery -- and
tell the reader those are honest held-out performance. If a sample galaxy were
ever in training, every one of those numbers would be quietly optimistic and the
notebooks would be making a claim they cannot support.

This is not a hypothetical failure mode for this project. The submitted version
of the paper split an augmented dataset *by row*, so 99.2% of galaxies had
near-duplicate siblings on both sides and the median held-out row had 16 of them
in training. The samples were regenerated after that was fixed; this test is what
stops them drifting back, e.g. if someone rebuilds them from a different split.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "paired_DR4_logR.npz"
SAMPLES = [
    REPO / "tutorials_for_user" / "sample_one_spectrum.npz",
    REPO / "tutorials_for_user" / "sample_24_spectra.npz",
]


def _keys(field, target_id):
    """Galaxy identity as the dataset defines it: (field, target_id)."""
    return np.char.add(
        np.char.add(np.atleast_1d(field).astype(str), "|"),
        np.atleast_1d(target_id).astype(str),
    )


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_tutorial_sample_is_entirely_held_out(sample: Path):
    if not DATASET.exists():
        pytest.skip(f"{DATASET.name} not available (gitignored; rebuild to run this)")
    if not sample.exists():
        pytest.skip(f"{sample.name} not present")

    from specsr.data.splits import get_or_make_split_3way

    # The fractions are part of the split cache key, so they must match the ones
    # the released weights were trained under. Defaults here are NOT 80/20 and
    # would silently build and compare against a different split.
    train_idx, val_idx, _, _ = get_or_make_split_3way(
        str(DATASET), train_frac=0.8, val_frac=0.2
    )

    with np.load(DATASET, allow_pickle=True) as d:
        key = _keys(d["field"], d["target_id"])
    held_out = set(key[val_idx].tolist())
    training = set(key[train_idx].tolist())

    with np.load(sample, allow_pickle=True) as s:
        sample_keys = _keys(s["field"], s["target_id"])

    leaked = sorted(k for k in sample_keys.tolist() if k in training)
    assert not leaked, (
        f"{sample.name} contains {len(leaked)} galaxy(ies) from the TRAINING split "
        f"({leaked[:5]}). The tutorials present their output as held-out "
        "performance; with these included that claim is false."
    )

    missing = sorted(k for k in sample_keys.tolist() if k not in held_out)
    assert not missing, (
        f"{sample.name} contains {len(missing)} galaxy(ies) in neither partition "
        f"({missing[:5]}), so their provenance cannot be verified at all."
    )
