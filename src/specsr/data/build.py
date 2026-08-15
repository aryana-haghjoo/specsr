"""Build paired training products from the raw JADES release.

Pipeline::

    discover x1d files
      -> group by (field, target_id)          # one galaxy, whatever the tier
      -> attach catalogue redshift            # secure flags only
      -> resample prism onto the log grid     # flux-conserving, upsampling
      -> stitch the medium gratings           # inverse-variance, with a mask
      -> quality cuts
      -> augment                              # explicit parent_id
      -> write .npz

Every stage is deliberate about provenance and about not inventing data; the
reasoning lives in the modules that do the work (:mod:`~specsr.data.ingest`,
:mod:`~specsr.data.grid`, :mod:`~specsr.data.stitch`,
:mod:`~specsr.data.augment`).

Redshifts
---------
Taken from ``Combined_DR4_external_v1.2.1.fits`` as ``z_Spec``, keeping only
quality flags A, B and C. That selection yields 3,297 galaxies across the
catalogue, which is the "robust spectroscopic redshifts" sample the paper
already quotes, so the definition is inherited rather than invented. Flag E is
almost entirely ``z_Spec = -1`` (no redshift) and is dropped.

The redshift matters beyond bookkeeping: SR2 places its line tokens using it, so
an insecure redshift puts the physics prior in the wrong place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .augment import AugmentationConfig, augment_pair
from .grid import DEFAULT_GRID, LogWavelengthGrid, resample_flux_conserving
from .ingest import GRATINGS, PRISM, discover_spectra, group_by_target, read_spectrum
from .stitch import stitch_gratings

__all__ = ["BuildConfig", "build_dataset", "run_build"]

#: Catalogue redshift-quality flags accepted as secure.
SECURE_Z_FLAGS = ("A", "B", "C")

_FIELD_FROM_CATALOG = {"GN": "goods-n", "GS": "goods-s"}


@dataclass
class BuildConfig:
    """Options for a dataset build."""

    release: str = "DR4"
    fields: tuple[str, ...] = ("goods-n", "goods-s")
    require_gratings: int = 3
    min_coverage: float = 0.5
    #: Minimum fraction of the grid that must be genuinely measured in the
    #: reference. A galaxy covering only a sliver contributes almost nothing but
    #: still costs a full row.
    min_valid_fraction: float = 0.5
    secure_z_flags: tuple[str, ...] = SECURE_Z_FLAGS
    grid: LogWavelengthGrid = field(default_factory=lambda: DEFAULT_GRID)
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)
    #: Optional cap on the number of galaxies, for smoke tests.
    limit: int | None = None
    #: Augment only the galaxies that will be trained on. The held-out galaxies
    #: then contribute exactly one row each -- their real spectrum -- and no
    #: synthetic copy of a held-out galaxy exists anywhere in the product.
    #:
    #: Augmenting everything and filtering at split time is behaviourally
    #: identical, but it leaves ~11k rows in the file that nothing may use, and
    #: a row that must never be read is a mistake waiting for someone who does
    #: not know that.
    augment_train_only: bool = True
    #: Fraction of galaxies used for training. Must match the fraction the
    #: training split uses, or the product's augmentation and the split will
    #: disagree about which galaxies are held out.
    train_frac: float = 0.8
    split_seed: int = 42


def load_redshifts(release_dir: Path, flags: tuple[str, ...] = SECURE_Z_FLAGS) -> dict:
    """Map ``(field, target_id)`` to a secure catalogue redshift."""
    from astropy.table import Table

    cat_path = release_dir / "Combined_DR4_external_v1.2.1.fits"
    if not cat_path.exists():
        raise FileNotFoundError(f"redshift catalogue not found: {cat_path}")

    t = Table.read(cat_path, hdu=1)
    z = np.asarray(t["z_Spec"], dtype=float)
    flag = np.array([str(x).strip() for x in t["z_Spec_flag"]])
    fld = np.array([_FIELD_FROM_CATALOG.get(str(x).strip(), "") for x in t["Field"]])
    nid = np.asarray(t["NIRSpec_ID"], dtype=int)

    keep = (z > 0) & np.isin(flag, list(flags)) & (fld != "")
    return {
        (f, f"{i:08d}"): float(zz)
        for f, i, zz in zip(fld[keep], nid[keep], z[keep], strict=True)
    }


def build_dataset(
    release_dir: Path | str,
    out_path: Path | str,
    config: BuildConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Build one ``.npz`` product. Returns a summary dict."""
    config = config or BuildConfig()
    release_dir = Path(release_dir)
    out_path = Path(out_path)
    grid = config.grid
    wave = grid.centers()

    def log(msg):
        if verbose:
            print(f"[build] {msg}", flush=True)

    t0 = time.time()
    files = discover_spectra(release_dir, fields=config.fields)
    log(f"discovered {len(files)} x1d files")

    groups = group_by_target(files, require_prism=True, require_gratings=config.require_gratings)
    log(f"{len(groups)} targets with prism + >={config.require_gratings} gratings")

    redshifts = load_redshifts(release_dir, config.secure_z_flags)
    log(f"{len(redshifts)} catalogue redshifts with flags {config.secure_z_flags}")

    keys = sorted(k for k in groups if k in redshifts)
    log(f"{len(keys)} targets have both paired spectra and a secure redshift")
    if config.limit:
        keys = keys[: config.limit]
        log(f"limited to {len(keys)} targets")

    rows: list[dict] = []
    meta: list[dict] = []
    pending: list[dict] = []
    dropped = {"low_coverage": 0, "read_error": 0}

    for key in keys:
        field_name, target_id = key
        try:
            prism = read_spectrum(groups[key][PRISM][0].path)
            arms = {
                d: read_spectrum(groups[key][d][0].path)
                for d in GRATINGS
                if d in groups[key]
            }
        except Exception:
            dropped["read_error"] += 1
            continue

        # Low resolution is interpolated UP onto the high-resolution sampling.
        lo_f, lo_e = resample_flux_conserving(
            prism["wavelength"], prism["flux"], wave, err_in=prism["flux_err"]
        )
        lo_v = np.isfinite(lo_f)

        hi = stitch_gratings(arms, grid=grid, min_coverage=config.min_coverage)

        if hi.coverage < config.min_valid_fraction or lo_v.mean() < config.min_valid_fraction:
            dropped["low_coverage"] += 1
            continue

        z = redshifts[key]
        # Augmentation is deferred: which galaxies may be augmented is decided
        # below, once the full galaxy list is known and can be split.
        pending.append(
            dict(
                flux_low=lo_f, flux_low_err=lo_e, valid_low=lo_v,
                flux_high=hi.flux, flux_high_err=hi.flux_err, valid_high=hi.valid,
                z=z, parent_id=len(meta),
            )
        )
        meta.append(
            {
                "parent_id": len(meta),
                "field": field_name,
                "target_id": target_id,
                "ra": float(prism["ra"]),
                "dec": float(prism["dec"]),
                "z": z,
                "coverage": hi.coverage,
            }
        )

        if verbose and (len(meta) % 250 == 0):
            # `rows` is empty until augmentation runs, which is now deferred
            # until the split is known -- report what has actually been read.
            log(f"  {len(meta)} galaxies read  ({time.time()-t0:.0f}s)")

    log(f"kept {len(meta)} galaxies, dropped {dropped}")
    if not pending:
        raise RuntimeError("no galaxies survived the cuts")

    # ------------------------------------------------------------------
    # Decide the split *before* augmenting, so no synthetic copy of a
    # held-out galaxy is ever written.
    #
    # The split must be computed exactly as `specsr.data.splits` computes it on
    # the finished file, or the product and the splitter would disagree about
    # which galaxies are held out -- silently, and in the worst possible way.
    # So the same two functions are called here, on one row per galaxy: labels
    # come from (ra, dec, field), not from `parent_id`, because that is what
    # `parent_group_ids` uses and the two orderings are different.
    # ------------------------------------------------------------------
    train_ids = None
    if config.augment_train_only and config.augment.n_aug > 0:
        from .splits import make_group_split_3way

        # Group by `parent_id`, because that is the column
        # `get_or_make_split_3way` groups by when it reads the finished file.
        # It is *not* the same labelling as `parent_group_ids`, which sorts by
        # (ra, dec, field): both are permutations of 0..N-1, they seed the same
        # RNG, and they therefore select different galaxies. Using the wrong one
        # here leaves the product's unaugmented galaxies and the splitter's
        # held-out galaxies disjoint -- caught only because this is asserted
        # below rather than assumed.
        ids = np.array([m["parent_id"] for m in meta])
        tr, va, _ = make_group_split_3way(
            ids, np.ones(len(ids), dtype=bool),
            train_frac=config.train_frac, val_frac=1.0 - config.train_frac,
            seed=config.split_seed, allow_empty_test=True,
        )
        train_ids = {int(ids[i]) for i in tr}
        holdout_ids = {int(ids[i]) for i in va}
        log(f"augmenting {len(train_ids)} training galaxies; "
            f"{len(va)} held-out galaxies stay unaugmented")

    no_aug = AugmentationConfig(
        n_aug=0, sigma_z=config.augment.sigma_z,
        noise_frac=config.augment.noise_frac, seed=config.augment.seed,
    )
    for g in pending:
        # Per-galaxy seeding (see AugmentationConfig.seed) means a training
        # galaxy's realizations do not depend on which other galaxies were
        # augmented -- so this reproduces the previous product's training rows
        # exactly, rather than merely equivalently.
        use = config.augment if (train_ids is None or g["parent_id"] in train_ids) else no_aug
        rows.extend(augment_pair(grid=grid, config=use, **g))
    del pending

    if not rows:
        raise RuntimeError("no galaxies survived the cuts")

    # Neutral fill for the numerical value; the mask is what the loss consumes.
    # See specsr.training.losses for why both are required.
    def stack(name, fill=0.0):
        return np.stack([np.nan_to_num(r[name], nan=fill) for r in rows]).astype(np.float32)

    parent = np.array([r["parent_id"] for r in rows], dtype=np.int32)
    by_parent = {m["parent_id"]: m for m in meta}

    np.savez_compressed(
        out_path,
        flux_low=stack("flux_low"),
        flux_low_err=stack("flux_low_err", fill=1.0),
        valid_low=np.stack([r["valid_low"] for r in rows]),
        flux_high=stack("flux_high"),
        flux_high_err=stack("flux_high_err", fill=1.0),
        valid_high=np.stack([r["valid_high"] for r in rows]),
        z=np.array([r["z"] for r in rows], dtype=np.float64),
        parent_id=parent,
        is_original=np.array([r["is_original"] for r in rows]),
        ra=np.array([by_parent[p]["ra"] for p in parent]),
        dec=np.array([by_parent[p]["dec"] for p in parent]),
        field=np.array([by_parent[p]["field"] for p in parent]),
        target_id=np.array([by_parent[p]["target_id"] for p in parent]),
        wavelength_low=wave,
        wavelength_high=wave,
        # Provenance of the augmentation policy. A consumer can check that the
        # split it is about to draw is the one the file was built for, rather
        # than assuming it.
        augment_train_only=config.augment_train_only,
        split_train_frac=config.train_frac,
        split_seed=config.split_seed,
        n_aug=config.augment.n_aug,
        grid_lambda_min=grid.lambda_min,
        grid_lambda_max=grid.lambda_max,
        grid_resolving_power=grid.resolving_power,
    )

    summary = {
        "galaxies": len(meta),
        "rows": len(rows),
        "n_samples": int(wave.size),
        "dropped": dropped,
        "seconds": round(time.time() - t0, 1),
        "out": str(out_path),
        "median_coverage": float(np.median([m["coverage"] for m in meta])),
    }
    log(f"wrote {out_path} : {summary}")

    # Verify, on the file as written, that the galaxies left unaugmented are
    # exactly the ones the splitter will hold out. These are computed by
    # different code paths from different columns, and when they disagreed the
    # product looked perfectly healthy: right row count, no leak, no warning --
    # just augmented copies of held-out galaxies that nothing could use, which
    # is the very thing this build exists to eliminate.
    if train_ids is not None:
        from .splits import make_group_split_3way as _split

        with np.load(out_path, allow_pickle=True) as _d:
            _g = np.asarray(_d["parent_id"])
            _o = np.asarray(_d["is_original"], dtype=bool)
        _tr, _va, _ = _split(_g, _o, config.train_frac, 1.0 - config.train_frac,
                             config.split_seed, allow_empty_test=True)
        split_holdout = set(np.unique(_g[_va]).tolist())
        unaugmented = {int(k) for k in np.unique(_g) if int((_g == k).sum()) == 1}
        if split_holdout != unaugmented or split_holdout != holdout_ids:
            raise RuntimeError(
                "augmentation policy and split disagree about the held-out galaxies:\n"
                f"  builder left unaugmented : {len(unaugmented)} galaxies\n"
                f"  splitter holds out       : {len(split_holdout)} galaxies\n"
                f"  symmetric difference     : "
                f"{len(split_holdout ^ unaugmented)} galaxies"
            )
        unused = len(_g) - len(_tr) - len(_va)
        if unused:
            raise RuntimeError(f"{unused} rows belong to neither split; expected 0")
        log(f"verified: {len(split_holdout)} held-out galaxies unaugmented, "
            f"{len(_tr)} training rows, 0 rows unused")

    return summary


def run_build(args) -> int:
    """CLI entry point for ``specsr build-dataset``."""
    from ..paths import data_dir, release_dir

    cfg = BuildConfig(release=args.release)
    if getattr(args, "no_augment", False):
        cfg.augment = AugmentationConfig(n_aug=0)
    elif getattr(args, "n_aug", None) is not None:
        cfg.augment = AugmentationConfig(n_aug=args.n_aug)
    if getattr(args, "limit", None):
        cfg.limit = args.limit

    out = Path(args.out) if args.out else data_dir() / f"paired_{args.release}_logR.npz"
    build_dataset(release_dir(args.release), out, cfg)
    return 0
