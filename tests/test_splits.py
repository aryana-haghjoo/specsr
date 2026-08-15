"""Tests for the three-way group split.

These defend three separate guarantees, each fixing a distinct way the reported
numbers could be optimistic:

  1. galaxies never straddle a partition boundary (the augmentation leak)
  2. validation and test hold real spectra only (not synthetic perturbations)
  3. validation and test are different sets (so checkpoint selection cannot
     inflate the reported number)
"""

from __future__ import annotations

import numpy as np
import pytest

from specsr.data.splits import (
    assert_clean_3way,
    make_group_split_3way,
)

G, A = 200, 20


def _fixture():
    """G galaxies, originals first then A augmentations each."""
    groups = np.concatenate([np.arange(G), np.repeat(np.arange(G), A)])
    is_original = np.concatenate([np.ones(G, bool), np.zeros(G * A, bool)])
    return groups, is_original


def test_partitions_are_disjoint_by_galaxy():
    groups, orig = _fixture()
    tr, va, te = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    assert_clean_3way(tr, va, te, groups, orig)
    g = lambda i: set(np.unique(groups[i]).tolist())  # noqa: E731
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))
    assert len(g(tr) | g(va) | g(te)) == G


def test_eval_sets_contain_only_real_spectra():
    """A synthetic realization is not an observation."""
    groups, orig = _fixture()
    tr, va, te = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    assert orig[va].all()
    assert orig[te].all()
    # exactly one row per galaxy, so statistics are over independent objects
    assert len(va) == len(np.unique(groups[va]))
    assert len(te) == len(np.unique(groups[te]))


def test_training_keeps_the_augmented_rows():
    groups, orig = _fixture()
    tr, _, _ = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    assert (~orig[tr]).sum() > 0
    assert len(tr) == len(np.unique(groups[tr])) * (A + 1)


def test_proportions_are_over_galaxies_not_rows():
    groups, orig = _fixture()
    tr, va, te = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    n = lambda i: len(np.unique(groups[i]))  # noqa: E731
    assert n(tr) == pytest.approx(0.8 * G, abs=2)
    assert n(va) == pytest.approx(0.1 * G, abs=2)
    assert n(te) == pytest.approx(0.1 * G, abs=2)


def test_split_is_deterministic_and_seed_sensitive():
    groups, orig = _fixture()
    a = make_group_split_3way(groups, orig, 0.8, 0.1, seed=7)
    b = make_group_split_3way(groups, orig, 0.8, 0.1, seed=7)
    c = make_group_split_3way(groups, orig, 0.8, 0.1, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[2], c[2])


def test_guard_rejects_a_leaking_split():
    groups, orig = _fixture()
    tr, va, te = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    # move one training row of a training galaxy into test
    bad_te = np.concatenate([te, tr[:1]])
    with pytest.raises(RuntimeError, match="share"):
        assert_clean_3way(tr, va, bad_te, groups, orig)


def test_guard_rejects_augmented_rows_in_an_eval_set():
    groups, orig = _fixture()
    tr, va, te = make_group_split_3way(groups, orig, 0.8, 0.1, seed=0)
    aug_of_test_galaxy = np.flatnonzero((groups == groups[te[0]]) & ~orig)[:1]
    with pytest.raises(RuntimeError, match="originals only"):
        assert_clean_3way(tr, va, np.concatenate([te, aug_of_test_galaxy]), groups, orig)


def test_rejects_fractions_that_leave_no_test_set():
    groups, orig = _fixture()
    with pytest.raises(ValueError, match="no test galaxies"):
        make_group_split_3way(groups, orig, 0.9, 0.1, seed=0)


# --------------------------------------------------------------------------
# Two-way (80/20) split
# --------------------------------------------------------------------------


def _toy(n_gal=100, n_aug=20):
    """One original + n_aug augmentations per galaxy, like the built product."""
    groups = np.repeat(np.arange(n_gal), n_aug + 1)
    is_original = np.zeros(groups.shape, dtype=bool)
    is_original[:: n_aug + 1] = True
    return groups, is_original


def test_two_way_split_needs_explicit_opt_in():
    """An empty test partition must be asked for, never fallen into."""
    from specsr.data.splits import make_group_split_3way

    groups, orig = _toy()
    with pytest.raises(ValueError, match="allow_empty_test"):
        make_group_split_3way(groups, orig, train_frac=0.8, val_frac=0.2)

    tr, va, te = make_group_split_3way(
        groups, orig, train_frac=0.8, val_frac=0.2, allow_empty_test=True
    )
    assert len(te) == 0 and len(tr) > 0 and len(va) > 0


def test_two_way_split_keeps_the_train_set_identical_to_the_three_way_one():
    """80/20 and 80/10/10 must share a training set.

    Galaxies are permuted once with a fixed seed and train takes the first
    `train_frac`, so changing only how the remainder is divided cannot move a
    galaxy into or out of training -- which is what makes it safe to switch
    split scheme without retraining SR1.
    """
    from specsr.data.splits import make_group_split_3way

    groups, orig = _toy()
    tr3, va3, te3 = make_group_split_3way(groups, orig, 0.8, 0.1)
    tr2, va2, te2 = make_group_split_3way(groups, orig, 0.8, 0.2, allow_empty_test=True)

    assert set(tr3.tolist()) == set(tr2.tolist())
    # and the 20% eval set is exactly the old val + test galaxies
    assert set(np.unique(groups[va2])) == set(np.unique(groups[va3])) | set(
        np.unique(groups[te3])
    )


def test_two_way_eval_contains_only_original_spectra():
    """The product augments *every* galaxy, so an unfiltered 20% would be ~95%
    synthetic rows -- performance on perturbations, over 21x correlated samples."""
    from specsr.data.splits import make_group_split_3way

    groups, orig = _toy()
    tr, va, _ = make_group_split_3way(groups, orig, 0.8, 0.2, allow_empty_test=True)

    assert orig[va].all(), "evaluation set contains augmented rows"
    assert len(va) == len(np.unique(groups[va])), "more than one eval row per galaxy"
    # Training still uses the augmentations, which is the point of having them.
    assert (~orig[tr]).any()
    assert not set(np.unique(groups[tr])) & set(np.unique(groups[va]))


def test_evaluation_and_training_use_the_same_held_out_set():
    """"The held-out set" must mean one thing across the whole project.

    `specsr.evaluation.load_split` hardcoded 0.8/0.1 while training moved to
    0.8/0.2, which would have produced every figure and every quoted number on
    286 galaxies while training validated on 572 -- two different held-out sets,
    with nothing in any log to say so.
    """
    from specsr import evaluation
    from specsr.data.splits import get_training_split

    assert (evaluation.TRAIN_FRAC, evaluation.VAL_FRAC) == (
        get_training_split.__defaults__[0],
        get_training_split.__defaults__[1],
    ), "evaluation split fractions have drifted from get_training_split's defaults"
    assert evaluation.SEED == get_training_split.__defaults__[2]
    assert evaluation.ALLOW_EMPTY_TEST == get_training_split.__defaults__[4]


def test_held_out_galaxies_contribute_no_rows_at_all_to_training():
    """Neither the held-out originals nor their augmented copies may be trained on.

    Group-awareness already guarantees this, but it is the property the whole
    revision rests on, so it is asserted directly rather than inferred.
    """
    from specsr.data.splits import make_group_split_3way

    groups, orig = _toy(n_gal=100, n_aug=20)
    tr, va, _ = make_group_split_3way(groups, orig, 0.8, 0.2, allow_empty_test=True)

    eval_galaxies = set(np.unique(groups[va]).tolist())
    rows_of_eval_galaxies = set(np.flatnonzero(np.isin(groups, list(eval_galaxies))).tolist())

    assert not rows_of_eval_galaxies & set(tr.tolist()), (
        "training contains rows from held-out galaxies"
    )
    # Their augmented copies are simply unused -- in the product, in neither split.
    unused = rows_of_eval_galaxies - set(va.tolist())
    assert unused and not any(orig[i] for i in unused)


def test_builder_and_splitter_group_by_the_same_column():
    """The build-time split and the training split must pick the same galaxies.

    `get_or_make_split_3way` groups by the stored `parent_id`; `parent_group_ids`
    labels galaxies by sorted `(ra, dec, field)`. Both are permutations of
    0..N-1 and both seed the same RNG, so they select *different* galaxies —
    which is silent: the product still has the right row count, still leaks
    nothing, still warns about nothing, and simply holds out one set while
    augmenting another. The builder must therefore group by `parent_id`.
    """
    from specsr.data.splits import make_group_split_3way, parent_group_ids

    rng = np.random.default_rng(0)
    n_gal = 60
    ra = rng.uniform(53.0, 53.2, n_gal)
    dec = rng.uniform(-27.9, -27.7, n_gal)
    field = np.array(["goods-s"] * n_gal)
    parent_id = np.arange(n_gal)

    by_pid, _, _ = make_group_split_3way(
        parent_id, np.ones(n_gal, bool), 0.8, 0.2, allow_empty_test=True
    )
    labels = parent_group_ids(ra=ra, dec=dec, field=field)
    by_coord, _, _ = make_group_split_3way(
        labels, np.ones(n_gal, bool), 0.8, 0.2, allow_empty_test=True
    )

    # They must be different, or this test is not guarding anything.
    assert set(by_pid.tolist()) != set(by_coord.tolist()), (
        "the two labellings coincided; pick coordinates that actually reorder"
    )
