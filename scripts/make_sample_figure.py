#!/usr/bin/env python3
"""Sample-characterisation figure: what population this work is trained on.

Two panels:

  (a) redshift against apparent F444W magnitude, for the JADES NIRSpec parent
      sample and for the paired sample this work uses;
  (b) stellar mass against star-formation rate for the CANDELS-matched subset.

Panel (a) is built entirely from JADES: the DR4 combined target catalogue gives
the parent sample and the tiering, the DR5 NIRCam photometry gives the
magnitudes, and 97% of the sample has one.  Panel (b) has no JADES source --
neither the DR5 photometric catalogues nor the DR4 target catalogue carries a
stellar mass -- so it is the one place an external survey is load-bearing, and
it inherits that survey's incompleteness.  Both facts are annotated on the
figure rather than left for the reader to discover.

Two things this script handles that a naive cross-match does not:

1. **The GOODS-S CANDELS astrometry is offset from JADES** by about -0.25" in
   declination.  Matching at 0.1" without correcting it recovers 10 of 1,799
   GOODS-S galaxies instead of ~970, and matching at a loosened 0.3" pairs some
   objects with their neighbours instead.  The offset is measured from the data
   and removed before matching.

2. **The two CANDELS fields use different SED fits** -- FAST for GOODS-N
   (Barro et al. 2019), Santini et al. (2015) median-of-methods masses with the
   6a_tau star-formation rates for GOODS-S.  They are plotted with distinct
   markers so a reader is not invited to read a trend across the two systems.

Run from the repository root:

    python scripts/make_sample_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from specsr.paths import require_jades_root  # noqa: E402
from specsr.plotting import COLOR_HR, COLOR_SR, PAPER_RC, save_figure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Resolved from SPECSR_JADES_ROOT like every other consumer of the raw tree.
# This was an absolute path into one developer's home directory, so the script
# ran nowhere else and would have shipped that way.
JADES = require_jades_root()
DR4_CAT = JADES / "DR4" / "Combined_DR4_external_v1.2.1.fits"
DR5_PHOT = {
    "goods-n": JADES / "DR5/GOODS-N/hlsp/catalogs/hlsp_jades_jwst_nircam_goods-n_photometry_v5.0_catalog.fits",
    "goods-s": JADES / "DR5/GOODS-S/hlsp/catalogs/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits",
}
CANDELS = REPO / "data" / "candels"

#: nJy -> AB.  The DR5 catalogues are in nJy, for which the AB zero point is
#: 31.4 mag.
NJY_ZP = 31.4

#: Matching radius, arcsec.  JADES-internal matches land at ~0.02", so this is
#: loose for those; for CANDELS it is applied *after* the field offset is
#: removed, so it is a true separation rather than a systematic.
TOL = 0.3

#: Panel (a) reuses the paper's HR and SR colours rather than inventing a pair,
#: so a reader who has seen the spectra figures recognises the palette. Imported
#: rather than restated: the paper's colours have one definition.
COL_PARENT = COLOR_HR   # the survey's own sample, as HR is the reference data
COL_SAMPLE = COLOR_SR   # the subset this work super-resolves


def _native(a):
    """FITS tables come back big-endian; pandas and some numpy paths refuse them."""
    a = np.asarray(a)
    return a.astype(a.dtype.newbyteorder("=")) if a.dtype.kind in "iufc" else a


def match(ra_a, dec_a, ra_b, dec_b, tol=TOL):
    """Nearest-neighbour match. Returns ``(index_into_b, keep_mask)``."""
    ca = SkyCoord(np.asarray(ra_a, float) * u.deg, np.asarray(dec_a, float) * u.deg)
    cb = SkyCoord(np.asarray(ra_b, float) * u.deg, np.asarray(dec_b, float) * u.deg)
    idx, sep, _ = ca.match_to_catalog_sky(cb)
    return idx, sep.arcsec <= tol


def median_offset(ra_a, dec_a, ra_b, dec_b, search=1.0):
    """Median (dRA*cos(dec), dDec) of a-b in arcsec, over pairs within ``search``."""
    idx, keep = match(ra_a, dec_a, ra_b, dec_b, tol=search)
    if keep.sum() < 20:
        return 0.0, 0.0, int(keep.sum())
    cosd = np.cos(np.radians(np.asarray(dec_a, float)[keep]))
    dra = (np.asarray(ra_a, float)[keep] - np.asarray(ra_b, float)[idx[keep]]) * 3600 * cosd
    ddec = (np.asarray(dec_a, float)[keep] - np.asarray(dec_b, float)[idx[keep]]) * 3600
    return float(np.median(dra)), float(np.median(ddec)), int(keep.sum())


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------
def load_sample(dataset):
    """The 2,858 real galaxies.

    Deliberately *not* split into training and held-out partitions. The figure
    sits in Section 2, where the sample is introduced and no training has been
    described yet; marking the partitions there would forward-reference the
    split, the augmentation that motivates it and the evaluation that uses it,
    none of which the reader has met at that point. What the figure
    characterises is the dataset, which is one population.
    """
    with np.load(dataset, allow_pickle=True) as d:
        orig = np.asarray(d["is_original"], bool)
        ra, dec = d["ra"][orig], d["dec"][orig]
        field = np.asarray(d["field"]).astype(str)[orig]
        z = d["z"][orig]

    print(f"sample: {orig.sum()} galaxies, "
          f"z {z.min():.2f}-{z.max():.2f} (median {np.median(z):.2f})")
    return dict(ra=ra, dec=dec, field=field, z=z)


def load_parent():
    """The JADES NIRSpec target catalogue: every galaxy we could have used."""
    with fits.open(DR4_CAT) as h:
        d = h[1].data
        out = dict(
            ra=_native(d["RA_TARG"]).astype(float),
            dec=_native(d["Dec_TARG"]).astype(float),
            field=np.array(["goods-n" if str(f).strip().upper() == "GN" else "goods-s"
                            for f in d["Field"]]),
            z_spec=_native(d["z_Spec"]).astype(float),
            z_phot=_native(d["z_phot"]).astype(float),
            tier=np.array([str(t).strip() for t in d["TIER"]]),
        )
    out["z"] = np.where(np.isfinite(out["z_spec"]) & (out["z_spec"] > 0),
                        out["z_spec"], out["z_phot"])
    print(f"parent: {len(out['ra'])} JADES NIRSpec targets, "
          f"{np.isfinite(out['z']).sum()} with a redshift")
    return out


def add_f444w(cat):
    """Attach F444W KRON AB magnitudes from the DR5 NIRCam catalogues."""
    n = len(cat["ra"])
    mag = np.full(n, np.nan)
    for fld, path in DR5_PHOT.items():
        sel = np.where(cat["field"] == fld)[0]
        if not sel.size:
            continue
        with fits.open(path, memmap=True) as h:
            k = h["KRON"].data
            idx, keep = match(cat["ra"][sel], cat["dec"][sel],
                              _native(k["RA"]), _native(k["DEC"]))
            f = _native(k["F444W_KRON"]).astype(float)[idx]
        good = keep & np.isfinite(f) & (f > 0)
        mag[sel[good]] = NJY_ZP - 2.5 * np.log10(f[good])
        print(f"  {fld}: F444W for {good.sum()}/{len(sel)} ({100 * good.mean():.1f}%)")
    cat["m444"] = mag
    return cat


# --------------------------------------------------------------------------
# CANDELS
# --------------------------------------------------------------------------
def load_candels():
    """Harmonised (RA, Dec, log M*, log SFR) per field, on each field's own scale."""
    gn_phot = Table.read(CANDELS / "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1-1_photometry-cat.fits", hdu=1)
    gn_fast = Table.read(CANDELS / "hlsp_candels_hst_wfc3_goodsn-barro19_multi_v1_mass-fast-cat.fits", hdu=1)
    order = np.argsort(_native(gn_fast["ID"]))
    pos = np.searchsorted(_native(gn_fast["ID"])[order], _native(gn_phot["ID"]))
    row = order[np.clip(pos, 0, len(order) - 1)]
    ok = _native(gn_fast["ID"])[row] == _native(gn_phot["ID"])
    gn = dict(
        ra=_native(gn_phot["RA"]).astype(float),
        dec=_native(gn_phot["DEC"]).astype(float),
        mass=np.where(ok, _native(gn_fast["lmass"]).astype(float)[row], np.nan),
        sfr=np.where(ok, _native(gn_fast["lsfr"]).astype(float)[row], np.nan),
    )

    gs_mass = Table.read(CANDELS / "hlsp_candels_hst_wfc3_goodss_santini_v1_mass-cat.fits", hdu=1)
    gs_phys = Table.read(CANDELS / "hlsp_candels_hst_wfc3_goodss_santini_v1_physpar-cat.fits", hdu=1)
    assert np.array_equal(_native(gs_mass["Seq"]), _native(gs_phys["Seq"])), \
        "Santini mass and physpar catalogues are not row-aligned"
    m_lin = _native(gs_mass["M_med"]).astype(float)
    s_lin = _native(gs_phys["SFR_6a_tau"]).astype(float)
    gs = dict(
        ra=_native(gs_mass["RAdeg"]).astype(float),
        dec=_native(gs_mass["DECdeg"]).astype(float),
        mass=np.where(m_lin > 0, np.log10(np.where(m_lin > 0, m_lin, 1)), np.nan),
        sfr=np.where(s_lin > 0, np.log10(np.where(s_lin > 0, s_lin, 1)), np.nan),
    )
    print(f"CANDELS: {len(gn['ra'])} GOODS-N (FAST), {len(gs['ra'])} GOODS-S (Santini)")
    return {"goods-n": gn, "goods-s": gs}


def add_candels(cat):
    """Attach CANDELS M* and SFR, correcting each field's astrometric offset."""
    cd = load_candels()
    n = len(cat["ra"])
    cat["mass"] = np.full(n, np.nan)
    cat["sfr"] = np.full(n, np.nan)
    for fld, tab in cd.items():
        sel = np.where(cat["field"] == fld)[0]
        if not sel.size:
            continue
        dra, ddec, npair = median_offset(cat["ra"][sel], cat["dec"][sel],
                                         tab["ra"], tab["dec"])
        cosd = np.cos(np.radians(np.median(cat["dec"][sel])))
        ra_c = tab["ra"] + dra / 3600.0 / cosd
        dec_c = tab["dec"] + ddec / 3600.0
        idx, keep = match(cat["ra"][sel], cat["dec"][sel], ra_c, dec_c)
        cat["mass"][sel[keep]] = tab["mass"][idx[keep]]
        cat["sfr"][sel[keep]] = tab["sfr"][idx[keep]]
        print(f"  {fld}: offset dRA {dra:+.3f}\" dDec {ddec:+.3f}\" from {npair} pairs "
              f"-> {keep.sum()}/{len(sel)} matched ({100 * keep.mean():.1f}%)")
    both = np.isfinite(cat["mass"]) & np.isfinite(cat["sfr"])
    print(f"  M* and SFR for {both.sum()}/{n} ({100 * both.mean():.1f}%)")
    return cat


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(sample, parent, out):
    """Panel (a) carries marginal histograms; the distributions are the point of
    the panel, and a scatter alone hides where the sample actually sits."""
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(11.4, 4.9))
        gs = fig.add_gridspec(
            2, 4, width_ratios=[4.0, 0.85, 0.55, 4.4], height_ratios=[0.8, 4.0],
            wspace=0.06, hspace=0.06, left=0.06, right=0.985, top=0.93, bottom=0.115)
        ax1 = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=ax1)
        axr = fig.add_subplot(gs[1, 1], sharey=ax1)
        ax2 = fig.add_subplot(gs[:, 3])

        # ---- (a) redshift vs apparent magnitude ----
        pm = np.isfinite(parent["m444"]) & np.isfinite(parent["z"])
        sm = np.isfinite(sample["m444"])

        # The parent is the backdrop and outnumbers the sample, so it is drawn
        # smaller and more transparent. Same hue, less weight: the eye should
        # land on the sample without the palette changing.
        ax1.scatter(parent["z"][pm], parent["m444"][pm], s=3.5, c=[COL_PARENT],
                    edgecolors="none", alpha=0.45, rasterized=True, zorder=1)
        ax1.scatter(sample["z"][sm], sample["m444"][sm], s=6, c=[COL_SAMPLE],
                    edgecolors="none", alpha=0.8, rasterized=True, zorder=2)

        mlo, mhi = np.nanpercentile(sample["m444"], [0.2, 99.8])
        ax1.set_ylim(mhi + 0.9, mlo - 0.9)          # inverted: bright at the top
        ax1.set_xlim(-0.4, 14.6)
        ax1.set_xlabel("Redshift")
        ax1.set_ylabel(r"$m_{\rm F444W}$ (AB)")
        ax1.grid(alpha=0.25, lw=0.5)
        ax1.legend(handles=[
            Line2D([], [], marker="o", ls="", ms=4.5, color=COL_PARENT,
                   label=f"JADES NIRSpec targets ({pm.sum():,})"),
            Line2D([], [], marker="o", ls="", ms=4.5, color=COL_SAMPLE,
                   label=f"paired sample used here ({sm.sum():,})"),
        ], loc="lower right", fontsize=7.6, framealpha=0.93, handletextpad=0.4)

        zbins = np.arange(0, 15.0, 0.5)
        mbins = np.arange(np.floor(mlo) - 1, np.ceil(mhi) + 1, 0.35)
        for ax, data, bins, orient in ((axt, "z", zbins, "v"), (axr, "m444", mbins, "h")):
            kw = dict(bins=bins, histtype="step", lw=0.9, density=True)
            if orient == "h":
                kw["orientation"] = "horizontal"
            ax.hist(parent[data][pm], color=COL_PARENT, **kw)
            ax.hist(sample[data][sm], color=COL_SAMPLE, **kw)
            ax.set_axis_off()
        axt.set_title("(a) redshift and apparent magnitude", fontsize=10, pad=4)

        # ---- (b) stellar mass vs star-formation rate ----
        both = np.isfinite(sample["mass"]) & np.isfinite(sample["sfr"])
        zc = sample["z"][both]
        vmax = float(np.nanpercentile(zc, 98))
        # Santini et al. floor SFR_6a_tau at 0.01 Msun/yr, so a run of GOODS-S
        # galaxies lands exactly on log SFR = -2. Left in the main scatter that
        # floor draws a straight line a reader would take for a real feature, so
        # those points are marked as the upper limits they are.
        floored = both & np.isclose(sample["sfr"], -2.0, atol=1e-3) & (sample["field"] == "goods-s")
        for fld, marker, label in (("goods-s", "o", "GOODS-S (Santini et al. 2015)"),
                                   ("goods-n", "^", "GOODS-N (Barro et al. 2019)")):
            m = both & (sample["field"] == fld) & ~floored
            sc = ax2.scatter(sample["mass"][m], sample["sfr"][m], c=sample["z"][m],
                             s=13, marker=marker, cmap="plasma", vmin=0, vmax=vmax,
                             edgecolors="none", alpha=0.85, rasterized=True,
                             label=f"{label}, {m.sum():,}")
        if floored.any():
            ax2.scatter(sample["mass"][floored], sample["sfr"][floored],
                        c=sample["z"][floored], s=22, marker="v", cmap="plasma",
                        vmin=0, vmax=vmax, alpha=0.85, linewidths=0.4,
                        edgecolors="0.35", rasterized=True,
                        label=f"SFR at the GOODS-S fitting floor, {floored.sum():,}")
            print(f"  {floored.sum()} GOODS-S galaxies sit on the SFR floor (0.01 Msun/yr)")
        cb = fig.colorbar(sc, ax=ax2, pad=0.015)
        cb.set_label("Redshift", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        # A handful of SED fits run away to unphysical corners; the axes are set
        # from the bulk so the main sequence is legible, and the number left
        # outside is stated rather than quietly dropped.
        xlo, xhi = np.nanpercentile(sample["mass"][both], [0.5, 99.7])
        ylo, yhi = np.nanpercentile(sample["sfr"][both], [0.5, 99.7])
        xlo, xhi = xlo - 0.35, xhi + 0.35
        ylo, yhi = ylo - 0.35, yhi + 0.35
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(ylo, yhi)
        outside = int(np.sum(both & ((sample["mass"] < xlo) | (sample["mass"] > xhi) |
                                     (sample["sfr"] < ylo) | (sample["sfr"] > yhi))))
        ax2.set_xlabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
        ax2.set_ylabel(r"$\log_{10}({\rm SFR}\,/\,M_{\odot}\,{\rm yr}^{-1})$")
        ax2.grid(alpha=0.25, lw=0.5)
        ax2.legend(loc="upper left", fontsize=7.6, framealpha=0.93, handletextpad=0.4)
        ax2.set_title("(b) stellar mass and star-formation rate", fontsize=10, pad=4)
        # The matched fraction and the clipped count belong in the caption, not
        # printed onto the panel; they are reported here so the caption can be
        # kept honest when the figure is regenerated.
        print(f"  caption numbers: {100 * both.mean():.0f}% matched, "
              f"{outside} galaxies outside the panel axes")

        path = save_figure(fig, out)
        plt.close(fig)
    print(f"wrote {path}")
    return path


def report(sample):
    """The numbers the manuscript text quotes."""
    z, m = sample["z"], sample["m444"]
    both = np.isfinite(sample["mass"]) & np.isfinite(sample["sfr"])
    print("\n--- numbers for the text ---")
    print(f"galaxies                : {len(z)}")
    print("z   p5/50/95            : " + " / ".join(f"{v:.2f}" for v in np.nanpercentile(z, [5, 50, 95])))
    print(f"z range                 : {z.min():.2f} - {z.max():.2f}")
    print(f"F444W available         : {np.isfinite(m).sum()} ({100 * np.isfinite(m).mean():.1f}%)")
    print("F444W p5/50/95          : " + " / ".join(f"{v:.2f}" for v in np.nanpercentile(m, [5, 50, 95])))
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 15)]:
        s = (z >= lo) & (z < hi)
        print(f"  z {lo:>2}-{hi:<2}: {s.sum():5d} galaxies, median F444W {np.nanmedian(m[s]):5.2f}, "
              f"{100 * both[s].mean():5.1f}% with CANDELS M*+SFR")
    print(f"CANDELS M*+SFR          : {both.sum()} ({100 * both.mean():.1f}%)")
    print("log M* p5/50/95         : " + " / ".join(
        f"{v:.2f}" for v in np.nanpercentile(sample["mass"][both], [5, 50, 95])))
    print("log SFR p5/50/95        : " + " / ".join(
        f"{v:.2f}" for v in np.nanpercentile(sample["sfr"][both], [5, 50, 95])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(REPO / "data" / "paired_DR4_logR.npz"))
    ap.add_argument("--out", default=None,
                    help="output path; defaults to $SPECSR_OUTPUT_DIR/figures/fig_sample.pdf")
    args = ap.parse_args()

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
    else:
        from specsr.paths import output_dir

        out = output_dir("figures") / "fig_sample.pdf"

    print("== sample ==")
    sample = load_sample(args.dataset)
    print("== JADES DR5 photometry ==")
    sample = add_f444w(sample)
    parent = add_f444w(load_parent())
    print("== CANDELS ==")
    sample = add_candels(sample)

    make_figure(sample, parent, out)
    report(sample)


if __name__ == "__main__":
    main()
