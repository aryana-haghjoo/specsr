#!/usr/bin/env python3
"""Blind redshift recovery with redmonster, on all four arms.

Are the redshift conclusions a property of the data, or of this particular
neural redshift head?  This runs an independent, classical estimator --
redmonster's chi-squared scan over template and redshift -- on the same held-out
galaxies, with the same settings for every arm, so the four are comparable to
each other and to the head.

The comparison that matters is *between arms*, not the absolute numbers.  The
templates cover rest-frame 1525-10852 A, so how much of a spectrum they can use
depends on redshift; but that limitation applies identically to the prism, SR1,
SR2 and the grating, so the ordering between them stays meaningful even where
the absolute performance is depressed.

Setup (redmonster is not on PyPI and is Python-2 era):

    git clone https://github.com/timahutchinson/redmonster.git
    # one compatibility fix: NumPy >= 1.24 refuses float arrays as indices, and
    # zfitter.z_refine uses rounded spline minima as indices. In
    # python/redmonster/physics/zfitter.py change
    #     zminlocs = n.round(zspline.get_min())
    # to
    #     zminlocs = n.round(zspline.get_min()).astype(int)
    # Values are unchanged; this is a compatibility cast, not an algorithm change.
    export REDMONSTER_DIR=/path/to/redmonster

Then:

    python scripts/redmonster_arms.py --out cache/redmonster_arms.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from specsr.data.grid import resample_flux_conserving  # noqa: E402

#: Redmonster's galaxy templates are sampled at 1e-4 dex/pixel and it requires
#: the data on the same scale. Ours is 1.086e-4 dex (the R = 4000 grid), so the
#: spectra are resampled *up* onto 1e-4 -- finer, so nothing is downsampled,
#: consistent with the rule that the reference is never degraded to fit a grid.
DEX_PER_PIX = 1e-4

#: 300 galaxy templates (15 x 4 x 5), rest-frame 1525-10852 A, with emission.
TEMPLATE = "ndArch-ssp_em_galaxy-v000.fits"

#: Steps of 4 pixels = 9.2e-4 in dz/(1+z). Coarser than the precision we are
#: comparing, which is why ZFitter's parabolic refinement of the chi2 minimum
#: matters: on the smoke test it recovered redshifts to ~1e-4, well inside one
#: step. Finer stepping costs runtime proportionally and buys nothing the
#: refinement does not already give.
NPIXSTEP = 4

ARMS = ("prism", "sr1", "sr2", "grating")


def load_redmonster():
    """Import redmonster, with an actionable error if it is not set up."""
    root = os.environ.get("REDMONSTER_DIR")
    if not root:
        raise SystemExit(
            "REDMONSTER_DIR is not set. See this script's docstring for the "
            "three setup steps (clone, one numpy-compat cast, export)."
        )
    sys.path.insert(0, str(Path(root) / "python"))
    os.environ.setdefault("REDMONSTER_TEMPLATES_DIR", str(Path(root) / "templates"))
    from redmonster.physics.zfinder import ZFinder
    from redmonster.physics.zfitter import ZFitter
    return ZFinder, ZFitter


def load_arms(cache, dataset):
    """The four spectra sets and their uncertainties, on our native grid."""
    d = np.load(cache, allow_pickle=True)
    wave_A = d["wave"].astype(float) * 1e4
    z_true = d["z_true"].astype(float)

    # The prism uncertainty is not carried in the prediction cache; take it from
    # the dataset by row, so every arm has a real error array rather than an
    # assumed one.
    with np.load(dataset, allow_pickle=True) as ds:
        lo_err = ds["flux_low_err"][d["row_index"]].astype(float)

    flux = {
        "prism": d["flux_low"].astype(float),
        "sr1": d["sr1"].astype(float),
        "sr2": d["sr2"].astype(float),
        "grating": d["flux_high"].astype(float),
    }
    err = {
        "prism": lo_err,
        "sr1": d["sr1_sigma"].astype(float),
        "sr2": d["sr2_sigma"].astype(float),
        "grating": d["flux_high_err"].astype(float),
    }
    return wave_A, z_true, flux, err


def to_redmonster_grid(wave_A, flux, err):
    """Resample onto redmonster's 1e-4 dex grid; return (specs, ivar, loglam)."""
    loglam = np.arange(np.log10(wave_A[0]), np.log10(wave_A[-1]), DEX_PER_PIX)
    grid = 10 ** loglam
    specs = np.zeros((len(flux), len(grid)))
    ivar = np.zeros_like(specs)
    for i in range(len(flux)):
        f, e = resample_flux_conserving(wave_A, flux[i], grid, err_in=err[i])
        # ivar 0 marks "no information here", which is how redmonster masks.
        # Non-finite flux or a non-positive error both mean exactly that.
        good = np.isfinite(f) & np.isfinite(e) & (e > 0)
        specs[i][good] = f[good]
        ivar[i][good] = 1.0 / e[good] ** 2
    return specs, ivar, loglam


def metrics(z_rm, z_true):
    """The two numbers Section 4.4 compares arms on."""
    ok = np.isfinite(z_rm) & (z_rm > 0)     # ZFitter flags failures with -1
    dz = np.full(len(z_true), np.nan)
    dz[ok] = (z_rm[ok] - z_true[ok]) / (1.0 + z_true[ok])
    out = np.abs(dz) > 0.15
    return dict(
        n=int(ok.sum()),
        n_failed=int((~ok).sum()),
        med_abs_dz=float(np.nanmedian(np.abs(dz))),
        outlier_frac=float(np.nansum(out) / max(ok.sum(), 1)),
        dz=dz,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(REPO / "cache" / "predictions_val.npz"))
    ap.add_argument("--dataset", default=str(REPO / "data" / "paired_DR4_logR.npz"))
    ap.add_argument("--out", default=str(REPO / "cache" / "redmonster_arms.npz"))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--restart", action="store_true",
                    help="recompute every arm instead of resuming from --out")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N galaxies (smoke testing)")
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=15.0)
    args = ap.parse_args()

    ZFinder, ZFitter = load_redmonster()
    wave_A, z_true, flux, err = load_arms(args.cache, args.dataset)
    if args.limit:
        z_true = z_true[: args.limit]
        flux = {k: v[: args.limit] for k, v in flux.items()}
        err = {k: v[: args.limit] for k, v in err.items()}
    print(f"{len(z_true)} galaxies, arms: {', '.join(ARMS)}")
    print(f"template {TEMPLATE}, npixstep {NPIXSTEP}, z in [{args.zmin}, {args.zmax}], "
          f"{args.nproc} procs\n")

    # Each arm takes ~40 minutes, so the output is written after every arm
    # rather than only at the end: an interrupted run then costs one arm
    # instead of all four, and a rerun skips what is already done. The first
    # attempt at this was killed 568 spectra into arm 1 and saved nothing.
    out = Path(args.out)
    results = {}
    if out.exists() and not args.restart:
        with np.load(out, allow_pickle=True) as prev:
            for a in ARMS:
                if f"z_{a}" in prev.files:
                    z_prev = prev[f"z_{a}"]
                    if len(z_prev) == len(z_true):
                        results[a] = (z_prev, metrics(z_prev, z_true))
                        print(f"{a:8s}  loaded from {out.name}, skipping")

    def save():
        done = [a for a in ARMS if a in results]
        np.savez(out, z_true=z_true, arms_done=np.array(done),
                 **{f"z_{a}": results[a][0] for a in done},
                 **{f"dz_{a}": results[a][1]["dz"] for a in done},
                 template=TEMPLATE, npixstep=NPIXSTEP)

    for arm in ARMS:
        if arm in results:
            continue
        t0 = time.time()
        specs, ivar, loglam = to_redmonster_grid(wave_A, flux[arm], err[arm])
        zf = ZFinder(fname=TEMPLATE, npoly=4, zmin=args.zmin, zmax=args.zmax,
                     nproc=args.nproc)
        zf.zchi2(specs, loglam, ivar, npixstep=NPIXSTEP)
        zfit = ZFitter(zf.zchi2arr, zf.zbase)
        zfit.z_refine()
        z_rm = np.asarray(zfit.z)[:, 0]
        m = metrics(z_rm, z_true)
        results[arm] = (z_rm, m)
        save()
        print(f"{arm:8s}  med|dz|/(1+z) = {m['med_abs_dz']:.5f}   "
              f"outliers = {100 * m['outlier_frac']:5.2f}%   "
              f"(n={m['n']}, failed={m['n_failed']})   "
              f"[{(time.time() - t0) / 60:.1f} min]  -> saved", flush=True)

    save()
    print(f"\nwrote {out}")

    print("\n--- for comparison, our redshift head on the same galaxies ---")
    print("   prism 0.00151 / 6.1%   SR1 0.00177 / 10.8%   "
          "SR2 0.00145 / 12.1%   grating 0.00083 / 11.0%")


if __name__ == "__main__":
    main()
