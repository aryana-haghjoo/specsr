#!/usr/bin/env python3
"""Group-aware train/test splitting for the augmented spectra datasets.

Background
----------
The augmented ``.npz`` datasets contain 21 rows per physical galaxy: one
original followed by 20 stochastic realizations (Gaussian redshift offset plus
10%-amplitude flux noise). Every row carries the *same* sky position as its
parent, and the ``id`` column is only a running row index -- it does not encode
provenance.

The original ``get_or_make_split`` drew a flat ``rng.permutation(N)`` over all
rows, so ~16-17 augmented siblings of each test galaxy landed in the training
set. Because the augmentations are near-duplicates, this leaked the test set
into training and inflated every reported held-out metric.

This module splits on the *parent galaxy* instead, so all 21 rows of a galaxy
fall on the same side of the split.

Grouping key
------------
Parents are recovered from ``(ra, dec, field)``, which augmentation leaves
untouched. This is preferred over the positional formula

    parent(i) = i if i < G else (i - G) // n_aug

because it additionally merges genuine duplicate observations of the same
object. The DR3 file has 1,187 rows spanning only 1,163 distinct sky positions
(23 objects appear more than once); those duplicates would straddle a
positional split. The DR4 file is clean (2,507 galaxies, uniform 21x), and
there the two schemes agree exactly.
"""

import hashlib
import os
import time

import numpy as np

# Bumped whenever the splitting algorithm changes, so that split caches written
# by an older (leaky) version are never silently reused.
SPLIT_SCHEME = "groupsplit-v1"

# Sky coordinates are matched at this precision (degrees). 1e-6 deg ~ 3.6 mas,
# far below any astrometric ambiguity but exact for augmented copies, which
# carry bit-identical coordinates.
_COORD_DECIMALS = 6


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def parent_group_ids(dataset_path=None, ra=None, dec=None, field=None):
    """Return an integer parent-galaxy label per row.

    Pass either ``dataset_path`` or the ``ra``/``dec``/``field`` arrays.
    Rows sharing a sky position (and field) get the same label.
    """
    if ra is None:
        with np.load(dataset_path, allow_pickle=True) as d:
            ra, dec, field = d["ra"], d["dec"], d["field"]

    ra = np.round(np.asarray(ra, dtype=float), _COORD_DECIMALS)
    dec = np.round(np.asarray(dec, dtype=float), _COORD_DECIMALS)
    field = np.asarray(field).astype(str)

    # strict=True: mismatched lengths here would silently truncate the key and
    # produce a wrong split rather than an error.
    key = np.array([f"{a!r}_{b!r}_{c}" for a, b, c in zip(ra, dec, field, strict=True)])
    _, groups = np.unique(key, return_inverse=True)
    return groups


def make_group_split(groups, train_frac=0.8, seed=42):
    """Split galaxies (not rows) into train/test, then expand back to rows.

    ``train_frac`` is applied to the number of *galaxies*. Because every galaxy
    contributes the same number of rows in these datasets, the resulting row
    fractions match closely; they are not forced to match exactly.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_groups)
    n_train = int(round(train_frac * len(perm)))
    train_groups = set(perm[:n_train].tolist())

    is_train = np.array([g in train_groups for g in groups])
    train_idx = np.flatnonzero(is_train)
    test_idx = np.flatnonzero(~is_train)
    return train_idx, test_idx


def assert_no_group_leakage(train_idx, test_idx, groups):
    """Fail loudly if any galaxy has rows on both sides of the split."""
    shared = np.intersect1d(groups[train_idx], groups[test_idx])
    if shared.size:
        raise RuntimeError(
            f"Split leaks {shared.size} parent galaxies across train/test "
            f"(e.g. group {shared[0]}). Refusing to train."
        )
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise RuntimeError(f"train_idx and test_idx share {overlap.size} rows.")


def get_or_make_split(dataset_path, N, train_frac=0.8, seed=42, split_dir="splits"):
    """Group-aware replacement for the original flat-permutation splitter.

    Signature matches the previous helper so call sites need no changes, but
    cache files use a distinct ``groupsplit_`` prefix: old leaky ``split_*.npz``
    files are ignored rather than reused.
    """
    os.makedirs(split_dir, exist_ok=True)
    ds_hash = hash_file(dataset_path)
    split_path = os.path.join(split_dir, f"groupsplit_{ds_hash}.npz")

    groups = parent_group_ids(dataset_path)
    if len(groups) != N:
        raise RuntimeError(
            f"Dataset has {len(groups)} rows but caller passed N={N}."
        )

    if os.path.exists(split_path):
        arr = np.load(split_path)
        train_idx, test_idx = arr["train_idx"], arr["test_idx"]
        if max(train_idx.max(), test_idx.max()) >= N:
            raise RuntimeError("Saved indices exceed current dataset size.")
        assert_no_group_leakage(train_idx, test_idx, groups)
        print(f"Loaded group split from {split_path}")
        return train_idx, test_idx, split_path

    train_idx, test_idx = make_group_split(groups, train_frac=train_frac, seed=seed)
    assert_no_group_leakage(train_idx, test_idx, groups)

    n_gal = len(np.unique(groups))
    np.savez(
        split_path,
        train_idx=train_idx,
        test_idx=test_idx,
        groups=groups,
        dataset_hash=ds_hash,
        N=N,
        n_galaxies=n_gal,
        n_train_galaxies=len(np.unique(groups[train_idx])),
        n_test_galaxies=len(np.unique(groups[test_idx])),
        scheme=SPLIT_SCHEME,
        seed=seed,
        train_frac=train_frac,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(
        f"Saved new group split to {split_path}\n"
        f"  {n_gal} galaxies -> {len(np.unique(groups[train_idx]))} train / "
        f"{len(np.unique(groups[test_idx]))} test\n"
        f"  {N} rows -> {len(train_idx)} train / {len(test_idx)} test"
    )
    return train_idx, test_idx, split_path


# ---------------------------------------------------------------------------
# Three-way splitting
# ---------------------------------------------------------------------------

#: Bumped when the three-way scheme changes, so old caches are never reused.
SPLIT_SCHEME_3WAY = "groupsplit3-v1"


def make_group_split_3way(
    groups,
    is_original,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    allow_empty_test: bool = False,
):
    """Split galaxies three ways, evaluating on real spectra only.

    Returns ``(train_idx, val_idx, test_idx)`` as row indices.

    Two separate guarantees, for two separate problems:

    **Galaxies never straddle a boundary.** All 21 rows of a galaxy (1 original
    + 20 augmentations) land in the same partition. A split drawn over rows put
    ~16 near-duplicate siblings of every held-out galaxy into training.

    **Validation and test contain only originals.** Augmentation is a
    training-time technique: a synthetic realization is not an observation.
    Evaluating on augmented rows would report performance on perturbations
    rather than on real spectra, and would compute statistics over 21x
    correlated rows, understating uncertainties — the effective sample size is
    the number of galaxies, not the number of rows.

    **Why validation and test are separate.** Training saves a checkpoint
    whenever the monitored metric improves, so over N epochs the saved model is
    the best of N draws on whatever set is monitored. Reporting that same set
    conflates genuine convergence with a favourable fluctuation, and the two
    cannot be separated using the set the selection was made on. Validation
    makes decisions; test is measured once.
    """
    groups = np.asarray(groups)
    is_original = np.asarray(is_original, dtype=bool)
    if is_original.shape != groups.shape:
        raise ValueError("groups and is_original must have the same length")

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_groups)

    n = len(perm)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    if n_train + n_val > n:
        raise ValueError(
            f"train_frac={train_frac} + val_frac={val_frac} exceeds the sample"
        )
    if n_train + n_val == n and not allow_empty_test:
        raise ValueError(
            f"train_frac={train_frac} + val_frac={val_frac} leaves no test galaxies. "
            "Pass allow_empty_test=True for a deliberate two-way split -- and note "
            "that the reported number is then the best of many draws on the set "
            "being reported, because checkpoint selection monitors it."
        )

    train_g = set(perm[:n_train].tolist())
    val_g = set(perm[n_train : n_train + n_val].tolist())
    test_g = set(perm[n_train + n_val :].tolist())

    in_train = np.array([g in train_g for g in groups])
    in_val = np.array([g in val_g for g in groups])
    in_test = np.array([g in test_g for g in groups])

    train_idx = np.flatnonzero(in_train)                 # augmented rows included
    val_idx = np.flatnonzero(in_val & is_original)       # real spectra only
    test_idx = np.flatnonzero(in_test & is_original)     # real spectra only
    return train_idx, val_idx, test_idx


def assert_clean_3way(train_idx, val_idx, test_idx, groups, is_original):
    """Fail loudly if any guarantee of the three-way split is violated."""
    groups = np.asarray(groups)
    is_original = np.asarray(is_original, dtype=bool)

    g_tr, g_va, g_te = (set(np.unique(groups[i]).tolist()) for i in (train_idx, val_idx, test_idx))
    checks = (
        (g_tr, g_va, "train/val"),
        (g_tr, g_te, "train/test"),
        (g_va, g_te, "val/test"),
    )
    for a, b, name in checks:
        shared = a & b
        if shared:
            raise RuntimeError(f"{name} share {len(shared)} galaxies, e.g. {sorted(shared)[:3]}")

    for idx, name in ((val_idx, "validation"), (test_idx, "test")):
        if not is_original[idx].all():
            n_bad = int((~is_original[idx]).sum())
            raise RuntimeError(
                f"{name} set contains {n_bad} augmented rows; it must be originals only"
            )

    overlap = set(train_idx.tolist()) & (set(val_idx.tolist()) | set(test_idx.tolist()))
    if overlap:
        raise RuntimeError(f"row index appears in more than one partition ({len(overlap)})")


def get_or_make_split_3way(
    dataset_path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    split_dir: str = "splits",
    allow_empty_test: bool = False,
):
    """Cached three-way split for a built product.

    Requires the product to carry ``parent_id`` and ``is_original``; products
    built before those existed are rejected rather than guessed at.
    """
    os.makedirs(split_dir, exist_ok=True)
    ds_hash = hash_file(dataset_path)
    # The fractions are part of the cache key. Without them an 80/20 run would
    # silently load an 80/10/10 cache and validate on half the galaxies it asked
    # for, with nothing in the logs to say so.
    tag = f"{int(round(train_frac * 100))}_{int(round(val_frac * 100))}"
    split_path = os.path.join(split_dir, f"groupsplit3_{tag}_{ds_hash}.npz")

    with np.load(dataset_path, allow_pickle=True) as d:
        missing = [k for k in ("parent_id", "is_original") if k not in d]
        if missing:
            raise KeyError(
                f"{dataset_path} lacks {missing}; rebuild it with `specsr build-dataset` "
                "so provenance is explicit rather than inferred."
            )
        groups = np.asarray(d["parent_id"])
        is_original = np.asarray(d["is_original"], dtype=bool)

    if os.path.exists(split_path):
        arr = np.load(split_path)
        tr, va, te = arr["train_idx"], arr["val_idx"], arr["test_idx"]
        assert_clean_3way(tr, va, te, groups, is_original)
        print(f"Loaded 3-way split from {split_path}")
        return tr, va, te, split_path

    tr, va, te = make_group_split_3way(
        groups, is_original, train_frac, val_frac, seed, allow_empty_test=allow_empty_test
    )
    assert_clean_3way(tr, va, te, groups, is_original)

    np.savez(
        split_path,
        train_idx=tr, val_idx=va, test_idx=te,
        groups=groups, is_original=is_original,
        dataset_hash=ds_hash, scheme=SPLIT_SCHEME_3WAY,
        seed=seed, train_frac=train_frac, val_frac=val_frac,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    n_g = lambda i: len(np.unique(groups[i]))  # noqa: E731
    print(
        f"Saved 3-way split to {split_path}\n"
        f"  galaxies : {n_g(tr)} train / {n_g(va)} val / {n_g(te)} test\n"
        f"  rows     : {len(tr)} train (augmented) / {len(va)} val / "
        f"{len(te)} test (originals only)"
    )
    return tr, va, te, split_path


def get_training_split(dataset_path, train_frac=0.8, val_frac=0.2, seed=42, split_dir="splits",
                       allow_empty_test=True):
    """Return ``(train_idx, val_idx, split_path)`` for a training run.

    Deliberately does **not** return the test indices. Training monitors the
    validation set for checkpoint selection and the learning-rate schedule; the
    test set must not be loaded during training at all, so that the number
    eventually reported is not the best of many draws on the set being reported.

    Retrieve the test indices separately, once, at evaluation time:

        _, _, test_idx, _ = get_or_make_split_3way(path)
    """
    train_idx, val_idx, test_idx, split_path = get_or_make_split_3way(
        dataset_path, train_frac=train_frac, val_frac=val_frac, seed=seed,
        split_dir=split_dir, allow_empty_test=allow_empty_test
    )
    held = (f"{len(test_idx)} test spectra withheld and not loaded" if len(test_idx)
            else "no test partition (two-way split)")
    print(
        f"  training on {len(train_idx)} rows; validating on {len(val_idx)} real spectra; "
        f"{held}"
    )
    return train_idx, val_idx, split_path
