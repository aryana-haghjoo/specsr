#!/usr/bin/env python
"""The controlled redshift comparison: one z-head, four inputs.

Reads each arm's ``predictions_*.npz`` -- which stores ``z_true``/``z_pred`` for
the held-out galaxies -- and draws one panel per arm, so the only thing differing
between panels is the spectrum representation the head was trained on.

Rendered by :func:`specsr.plotting.plot_redshift_comparison`, the same function
that draws the published three-arm figure, so the two read in one visual
language: hexbin density on a shared LogNorm, the 68% HDR contour, and one
colourbar meaning the same thing in every panel.

An earlier version of this script drew its own scatter, colouring points by
whether they were catastrophic outliers. That was a second visual language for
the same quantity, and the outlier fraction is already in each metrics box. The
shared function also adapts to sample size on its own -- with 572 galaxies it
drops the 95% contour, which at this count traces individual objects rather than
a tail, and overplots the galaxies instead of implying a smooth density.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from specsr.plotting import PAPER_RC, plot_redshift_comparison, save_figure

REPO = Path(__file__).resolve().parents[1]


def main():
    # This figure is destined for the manuscript, so it is drawn in the paper's
    # font like every other one. See `specsr.plotting.PAPER_RC`.
    plt.rcParams.update(PAPER_RC)

    ap = argparse.ArgumentParser()
    ap.add_argument("--lowres",
                    default="runs/zarms_8020_20260731_010103/lowres/predictions_lowres.npz")
    ap.add_argument("--sr1", default="runs/zhead_pdf_8020/sr1/predictions_sr1.npz")
    # The sr2 arm is deliberately NOT the copy sitting in the zarms run
    # directory beside the other three. That one was trained against a
    # superseded SR2 and scores 11.71% / 0.00167 against this one's
    # 12.06% / 0.00145. Defaulting to the zarms path is exactly how this figure
    # got rebuilt from the wrong arm on 2026-08-10.
    ap.add_argument("--sr2", default="runs/zsr2_final_20260810_101714/predictions_sr2.npz")
    ap.add_argument("--hires",
                    default="runs/zarms_8020_20260731_010103/hires/predictions_hires.npz")
    # Written to the package output directory by default; pass --out (or set
    # SPECSR_OUTPUT_DIR) to regenerate the manuscript figure in place.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Ordered LR -> SR1 -> SR2 -> HR: the pipeline's own progression, so the
    # figure reads left to right as "how much of the way to HR do we get".
    arms = [("Low-res input (prism)", args.lowres),
            ("SR1 output", args.sr1),
            ("SR2 output", args.sr2),
            ("High-res reference (grating)", args.hires)]

    loaded = []
    for title, path in arms:
        p = REPO / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            raise SystemExit(
                f"missing predictions for '{title}': {p}\n"
                "Every arm must be present -- a figure with a placeholder panel "
                "reads as a result rather than as a gap."
            )
        loaded.append((title, np.load(p, allow_pickle=True)))

    # Range from the *true* redshifts, not from the predictions: a handful of
    # catastrophic outliers predict far outside the catalogue's range, and
    # letting them set the limits shrinks the locus every panel exists to show.
    # A margin is added so those points stay visible rather than being clipped
    # silently against the frame.
    z_true_all = np.concatenate([d["z_true"] for _, d in loaded])
    lim = (0.0, float(np.ceil(z_true_all.max() + 0.5)))

    # Rendered by the shared figure function rather than by a local scatter, so
    # this reads in the same visual language as the published three-arm figure:
    # hexbin density on one shared LogNorm, the 68% HDR contour, individual
    # galaxies overplotted because 572 points cannot support a smooth density,
    # and one colourbar meaning the same thing in every panel.
    fig, stats = plot_redshift_comparison(
        [(np.asarray(d["z_true"], float), np.asarray(d["z_pred"], float))
         for _, d in loaded],
        titles=[t for t, _ in loaded],
        limits=lim,
        # A single row, matching the line-flux figure it is read against. The
        # earlier 2x2 wrap existed because the panels were sized for two
        # columns; they are now sized for four, so the metrics box survives the
        # reduction to text width.
        ncols=4,
    )

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
    else:
        from specsr.paths import output_dir

        out = output_dir("figures") / "redshift_arms.png"
    save_figure(fig, out, dpi=170)
    n = len(loaded[0][1]["z_true"])
    print(f"saved -> {out}  ({n} held-out galaxies)")
    for (title, _), m in zip(loaded, stats, strict=True):
        print(f"  {title:<30} outliers {m['outlier_frac']:.2%}  "
              f"med|dz|/(1+z) {m['med_abs_dz_over_1pz']:.5f}")


if __name__ == "__main__":
    main()
