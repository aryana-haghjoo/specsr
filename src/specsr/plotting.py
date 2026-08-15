"""Figure functions for the paper.

Plotting is library code, not script code: each function takes arrays and returns
a :class:`~matplotlib.figure.Figure`, so it can be imported, tested and
documented. The scripts in ``scripts/`` do argument parsing and file IO only.

These were extracted from the evaluation notebooks. The notebook versions called
``plt.show()`` and could only run inside a session; these return the figure and
take an optional ``output_path``, which is what makes them usable from a script,
from a test, or from another figure.

Nothing here loads data. Feed it the arrays from a prediction cache built by
``scripts/make_predictions.py`` so every panel in the paper demonstrably comes
from the same predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .metrics import (
    compute_psd_stats,
    redshift_metrics,
    robust_sigma_lambda,
    robust_symmetric_vlim,
)

__all__ = [
    "PAPER_RC",
    "COLOR_HR",
    "COLOR_LR",
    "COLOR_SR",
    "EM_LINE_TRACKS",
    "LINE_GROUPS",
    "COVERAGE_LINES",
    "DISPERSER_COLORS",
    "REDSHIFT_PANELS",
    "plot_augmentation_family",
    "plot_disperser_coverage",
    "plot_line_flux_comparison",
    "plot_redshift_comparison",
    "plot_redshift_panel",
    "plot_residual_psd",
    "redshift_metrics",
    "plot_snr_comparison",
    "plot_residual_maps",
    "plot_spectra_with_inset",
    "robust_ylims",
    "save_figure",
    "soft_seismic",
]

#: Matplotlib settings every figure in the paper is drawn under.
#:
#: The manuscript is set in a serif face, so the figures are too — mixing a
#: sans-serif panel into a serif paper is immediately visible on the page. Wrap
#: any new paper figure in ``plt.rc_context(PAPER_RC)`` rather than restating
#: the family locally, so "the paper's font" has one definition and a change of
#: face does not have to be chased through every plotting function.
#:
#: ``mathtext.fontset`` matters as much as ``font.family``: leaving it at the
#: default renders every ``$\lambda$`` and ``$\hat{z}$`` in DejaVu Sans beside
#: serif body text, which is the inconsistency this constant exists to prevent.
PAPER_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
}

#: Named groups of rest-frame lines (microns) for the zoom inset, with the line
#: the window centres on. Adding a group is how you add a new inset panel.
#: Lines traced across the residual maps, rest wavelength in microns. A line at
#: rest lambda appears at lambda*(1+z), so on a redshift-sorted map it draws a
#: curve -- which is how you tell a real feature from an instrumental one.
EM_LINE_TRACKS: dict[str, float] = {
    r"Ly$\alpha$": 0.1216,
    r"[OII]": 0.3727,
    r"H$\beta$": 0.4861,
    r"[OIII]": 0.5007,
    r"H$\alpha$": 0.6563,
    r"HeI": 1.0830,
    r"Pa$\beta$": 1.2818,
    r"Pa$\alpha$": 1.8751,
}

#: Where along each track to put its label, as a fraction of the visible span.
#: Hbeta and [OIII] are close enough together that the default 0.5 overlaps them.
_TRACK_LABEL_FRAC = {r"H$\beta$": 0.38, r"[OIII]": 0.62}

LINE_GROUPS: dict[str, dict] = {
    "o3hb": {
        "center": 0.5007,
        "lines": [("Hβ", 0.4861), ("[OIII]4959", 0.4959), ("[OIII]5007", 0.5007)],
    },
    "halpha_nii": {
        "center": 0.6563,
        "lines": [("Hα", 0.6563), ("[NII]6548", 0.6548), ("[NII]6583", 0.6583)],
    },
}


def robust_ylims(y, p_lo: float = 1, p_hi: float = 99, pad_frac: float = 0.08):
    """Percentile-based y limits that a single bad pixel cannot blow out.

    Emission-line spectra have a large dynamic range and occasional spikes, so
    ``min``/``max`` limits make the continuum an invisible flat line.

    The padding is **relative**, deliberately. The notebook version used
    ``pad_frac * (hi - lo + 1e-12)``; that absolute epsilon is negligible for
    normalised flux (O(1)) but these spectra are ~1e-21 in physical units, where
    the epsilon *is* the span — it produced limits of +/-6e-14 on data spanning
    3e-20, i.e. a flat line at zero. Same failure as an absolute threshold
    anywhere else in this codebase: fine at one scale, silently wrong at another.
    """
    y = np.asarray(y)
    m = np.isfinite(y)
    if not np.any(m):
        return (-1.0, 1.0)
    lo = float(np.nanpercentile(y[m], p_lo))
    hi = float(np.nanpercentile(y[m], p_hi))
    span = hi - lo
    if span <= 0:  # constant data: fall back to a scale set by the value itself
        span = abs(hi) if hi != 0 else 1.0
    pad = pad_frac * span
    return lo - pad, hi + pad


def save_figure(fig, output_path: str | Path, dpi: int = 450):
    """Write a figure, creating the directory. Matches the paper's export settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    return output_path


#: Palette, sampled from ``plasma`` at fixed points. Fixed rather than the
#: default cycle so every figure in the paper uses the same three colours for
#: HR / LR / SR, and so greyscale printing keeps them ordered by lightness.
_PLASMA = plt.get_cmap("plasma")
COLOR_HR = _PLASMA(0.18)
COLOR_LR = _PLASMA(0.52)
COLOR_SR = _PLASMA(0.72)

#: The redshift and line-flux comparison figures share this ramp, so a reader
#: moving between them is not asked to relearn what a colour means.
_VIRIDIS = plt.get_cmap("viridis")


def plot_spectra_with_inset(
    spectra,
    wave_um,
    *,
    inset_rest_range_um: tuple[float, float] = (0.4890, 0.5120),
    lines=None,
    fig_width: float = 12.0,
    output_path: str | Path | None = None,
):
    """Stacked LR / SR / HR panels, each with a rest-frame zoom inset.

    The inset window is specified in the **rest frame** and redshifted per
    spectrum, so the same physical wavelength range is shown whatever the
    redshift — which is what makes panels at different z comparable. A rectangle
    on the main axis marks where the inset is looking.

    Parameters
    ----------
    spectra
        Sequence of dicts, one per panel, with keys ``hr``, ``lr``, ``sr``,
        ``z_true``, and optionally ``z_pred`` and ``z_sigma``. Fluxes should be
        **normalised**, matching the paper's ``F_lambda (normalized)`` axis.
    wave_um
        Shared observed-frame wavelength grid, microns.
    lines
        ``(name, rest_wavelength_um)`` pairs to mark. Defaults to Hβ + [O III].

    Returns
    -------
    matplotlib.figure.Figure
    """
    spectra = list(spectra)
    if not spectra:
        raise ValueError("pass at least one spectrum")
    if lines is None:
        lines = LINE_GROUPS["o3hb"]["lines"]

    x = np.asarray(wave_um)
    nrows = len(spectra)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_width, 3.6 * nrows),
                             constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, out in zip(axes, spectra, strict=True):
        hr = np.asarray(out["hr"])
        lr = np.asarray(out["lr"])
        sr = np.asarray(out["sr"])
        z_true = float(out["z_true"])

        rest_lo, rest_hi = inset_rest_range_um
        x0 = max(rest_lo * (1.0 + z_true), float(x.min()))
        x1 = min(rest_hi * (1.0 + z_true), float(x.max()))
        if x1 <= x0:  # window redshifted off the grid: fall back to centring
            center = 0.5007 * (1.0 + z_true)
            half = 0.5 * (rest_hi - rest_lo) * (1.0 + z_true)
            x0 = max(center - half, float(x.min()))
            x1 = min(center + half, float(x.max()))
        mzoom = (x >= x0) & (x <= x1)

        y_main = robust_ylims(np.concatenate([hr, lr, sr]), p_lo=1, p_hi=99, pad_frac=0.06)
        # The inset's top comes from the data maximum, not from a robust
        # percentile. A p_hi of 99 is right for the main panel, which spans the
        # whole grid and is dominated by noise, but in the zoom it cut the top
        # off the brightest line -- the [O III] 5007 peak the figure exists to
        # show. The window is narrow and centred on the doublet, so its maximum
        # *is* the line peak. The lower limit stays robust, because that end is
        # continuum and noise.
        if mzoom.sum() > 10:
            zoom_all = np.concatenate([hr[mzoom], lr[mzoom], sr[mzoom]])
            y_lo = robust_ylims(zoom_all, p_lo=1, p_hi=99, pad_frac=0.10)[0]
            y_peak = float(np.nanmax(zoom_all))
            y_in = (y_lo, y_peak + 0.06 * (y_peak - y_lo))
        else:
            y_in = y_main

        ax.plot(x, hr, lw=1.0, alpha=0.55, color=COLOR_HR, label="HR", zorder=1)
        ax.plot(x, lr, lw=1.15, ls="--", alpha=0.85, color=COLOR_LR,
                label="LR (interp)", zorder=2)
        ax.plot(x, sr, lw=1.45, alpha=0.92, color=COLOR_SR, label="SR", zorder=3)
        ax.set_ylim(*y_main)
        ax.set_ylabel(r"$F_\lambda$ (normalized)")

        label = rf"z_true={z_true:.4f}"
        if out.get("z_pred") is not None:
            label += rf" | $\hat{{z}}$={float(out['z_pred']):.4f}"
            if out.get("z_sigma") is not None:
                label += rf" ± {float(out['z_sigma']):.4f}"
        ax.text(0.01, 0.90, label, transform=ax.transAxes, fontsize=12,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.7", alpha=0.85, linewidth=0.8))

        xmin, xmax = float(x.min()), float(x.max())
        for _name, lam_rest in lines:
            lam_obs = lam_rest * (1.0 + z_true)
            if xmin <= lam_obs <= xmax:
                ax.axvline(lam_obs, color="0.6", linestyle=":", linewidth=0.9, alpha=0.20)

        ax.add_patch(Rectangle((x0, y_in[0]), x1 - x0, y_in[1] - y_in[0], fill=False,
                               ec="0.6", lw=0.9, alpha=0.9, zorder=0))

        axins = inset_axes(ax, width="44%", height="44%", loc="upper right", borderpad=1.0)
        axins.plot(x[mzoom], hr[mzoom], lw=1.0, alpha=0.62, color=COLOR_HR,
                   label="HR", zorder=1)
        axins.plot(x[mzoom], lr[mzoom], lw=1.05, ls="--", alpha=0.82, color=COLOR_LR,
                   label="LR", zorder=2)
        axins.plot(x[mzoom], sr[mzoom], lw=1.30, alpha=0.92, color=COLOR_SR,
                   label="SR", zorder=3)
        in_window = [(name, lam_rest * (1.0 + z_true)) for name, lam_rest in lines
                     if x0 <= lam_rest * (1.0 + z_true) <= x1]

        # The line labels used to be drawn rotated and centred on the line they
        # name, which ran the text straight down through the peak and blended
        # it into the spectra. No horizontal shift fixes that: rotated text is
        # tall enough to reach well down the panel, and [O III] 5007's profile
        # is wider than the room between it and its neighbour. So reserve a
        # clear band along the top of the inset and set the labels horizontally
        # inside it -- nothing is plotted there, so there is nothing to collide
        # with, and horizontal text is easier to read at this size. The dotted
        # markers still sit exactly on the lines.
        head = 0.20 if in_window else 0.0
        band = head * (y_in[1] - y_in[0])
        axins.set_xlim(x0, x1)
        axins.set_ylim(y_in[0], y_in[1] + band)
        axins.tick_params(labelsize=8)

        if in_window:
            # The y-limits are robust percentiles, so the brightest line runs
            # off the top of the data region and would otherwise climb into the
            # band. Mask it: the band is for labels only.
            axins.axhspan(y_in[1], y_in[1] + band, facecolor="white",
                          edgecolor="none", zorder=4)
            # The band's top edge lies exactly on the top spine, and spines
            # default to zorder 2.5 -- so the white patch painted over it and
            # the frame came out with three black sides and one faint one.
            for spine in axins.spines.values():
                spine.set_zorder(7)

        for name, lam_t in in_window:
            # Stop the marker at the foot of the band rather than running it
            # through the text it points at.
            axins.axvline(lam_t, ymax=1.0 / (1.0 + head), color="0.30",
                          linestyle=":", linewidth=1.0, alpha=0.85, zorder=5)
            axins.annotate(name, xy=(lam_t, 1.0), xycoords=("data", "axes fraction"),
                           xytext=(0, -2), textcoords="offset points",
                           va="top", ha="center", fontsize=8, color="0.25", zorder=6)

        # Below the label band, so a long label and the legend cannot overlap.
        axins.legend(loc="upper right", fontsize=7, frameon=False,
                     handlelength=1.6, borderpad=0.2,
                     bbox_to_anchor=(1.0, 1.0 - head), bbox_transform=axins.transAxes)

    axes[-1].set_xlabel("Observed wavelength (μm)")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def soft_seismic():
    """A desaturated diverging colormap.

    Plain ``seismic`` is too saturated for large residual maps — it reads as
    alarming everywhere and hides structure. This mixes it halfway to white.
    """
    base = plt.cm.seismic(np.linspace(0, 1, 256))
    base = base * 0.5 + 0.5
    return mcolors.LinearSegmentedColormap.from_list("soft_seismic", base)


def plot_residual_maps(
    wavelength,
    flux_lr,
    flux_hr,
    flux_sr,
    z_true=None,
    *,
    sigma_lr_med=None,
    sigma_hr_med=None,
    sort_by_z: bool = True,
    vmax_abs_pct: float = 97.0,
    sigma_ymax_pct: float = 99.0,
    resid_min_count: int = 30,
    resid_clip_sigma: float = 6.0,
    residual_units: str = "physical units",
    show_tracks: bool = True,
    output_path: str | Path | None = None,
):
    """Three residual maps (LR−HR, LR−SR, SR−HR) over a per-wavelength scatter panel.

    Rows are spectra, sorted by redshift so that features tracking a line move
    diagonally and instrumental features stay vertical — that distinction is the
    whole reason the figure is sorted rather than left in file order.

    ``sigma_lr_med`` / ``sigma_hr_med`` are optional reference noise curves from
    :func:`specsr.metrics.estimate_median_noise_lambda`; the scatter panel is
    interpretable only against them, since the question is whether residuals sit
    at the noise level or above it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    wl = np.asarray(wavelength)
    lr, hr, sr = np.asarray(flux_lr), np.asarray(flux_hr), np.asarray(flux_sr)

    r_lr_hr, r_lr_sr, r_sr_hr = lr - hr, lr - sr, sr - hr

    z_sorted = None
    if sort_by_z and z_true is not None:
        order = np.argsort(np.asarray(z_true))
        z_sorted = np.asarray(z_true)[order]
        r_lr_hr, r_lr_sr, r_sr_hr = r_lr_hr[order], r_lr_sr[order], r_sr_hr[order]

    vmin, vmax = robust_symmetric_vlim([r_lr_hr, r_lr_sr, r_sr_hr], vmax_abs_pct)
    cmap = soft_seismic()

    sigs = [robust_sigma_lambda(r, resid_min_count, resid_clip_sigma)
            for r in (r_lr_hr, r_lr_sr, r_sr_hr)]

    pool = [s[np.isfinite(s)] for s in sigs]
    for extra in (sigma_lr_med, sigma_hr_med):
        if extra is not None:
            extra = np.asarray(extra)
            pool.append(extra[np.isfinite(extra)])
    allsig = np.concatenate(pool) if pool else np.array([1.0])
    # Relative, not floored at an absolute value. The notebook used
    # `max(1e-3, percentile)`, which is harmless for normalised flux but pins the
    # axis at 1e-3 for physical spectra ~1e-20 -- an empty panel. Fourth absolute
    # constant in this codebase to break at a different scale; prefer relative.
    ymax = float(np.percentile(allsig, sigma_ymax_pct)) if allsig.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    with plt.rc_context({
        **PAPER_RC,
        "font.size": 10, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    }):
        fig = plt.figure(figsize=(14.8, 8.6))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05],
                              height_ratios=[1.0, 0.55], wspace=0.25, hspace=0.28)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(gs[0, 3])

        extent = [wl[0], wl[-1], 0, r_lr_hr.shape[0]]
        ims = [ax.imshow(r, aspect="auto", origin="lower", extent=extent,
                         cmap=cmap, vmin=vmin, vmax=vmax)
               for ax, r in zip(axes, (r_lr_hr, r_lr_sr, r_sr_hr), strict=True)]

        for ax, title in zip(axes, ("LR − HR", "LR − SR", "SR − HR"), strict=True):
            ax.set_title(title)
            ax.set_xlabel(r"Observed wavelength [$\mu$m]")
        axes[0].set_ylabel(r"Redshift $z$")

        cb = fig.colorbar(ims[0], cax=cax)
        cb.set_label(f"Residual ({residual_units})")

        # Label rows by redshift rather than row number: the row index is an
        # artefact of sorting, the redshift is the physical axis.
        if z_sorted is not None:
            n = r_lr_hr.shape[0]
            pos, labs = [], []
            for zval in (0, 1, 2, 3, 4, 5, 10):
                if z_sorted[0] <= zval <= z_sorted[-1]:
                    pos.append(min(int(np.searchsorted(z_sorted, zval)), n - 1))
                    labs.append(str(zval))
            for ax in axes:
                ax.set_yticks(pos)
            axes[0].set_yticklabels(labs)
            for ax in axes[1:]:
                ax.set_yticklabels([])

            # Emission-line tracks. Without these the maps are hard to read:
            # every diagonal ridge is unlabelled and could be anything. With
            # them, a ridge either follows a track (a real line the model is
            # getting wrong) or it does not (something instrumental).
            #
            # ``show_tracks=False`` exists because the overlay hides what it
            # annotates: a dashed line drawn along a ridge covers the pixels a
            # reader most wants to see. Building an unobstructed version
            # alongside lets the figure annotate the ridges *and* be checked
            # without the annotation.
            if show_tracks:
                wl_min, wl_max = float(wl[0]), float(wl[-1])
                z_min, z_max = float(z_sorted[0]), float(z_sorted[-1])
                wl_track = np.linspace(wl_min, wl_max, 300)
                for name, lam_rest in EM_LINE_TRACKS.items():
                    z_track = wl_track / lam_rest - 1.0
                    idx_track = np.interp(z_track, z_sorted, np.arange(n),
                                          left=-1, right=n + 1)
                    vis = (z_track >= z_min) & (z_track <= z_max)
                    if vis.sum() < 2:
                        continue
                    for ax in axes:
                        ax.plot(wl_track[vis], idx_track[vis], color="k", lw=0.7,
                                ls="--", alpha=0.5)
                    frac = _TRACK_LABEL_FRAC.get(name, 0.5)
                    j = int(frac * (vis.sum() - 1))
                    for ax in axes:
                        ax.text(wl_track[vis][j], idx_track[vis][j], name,
                                fontsize=7.5, ha="center", va="bottom",
                                color="k", alpha=0.9)

        # One scatter panel under each map, each against the noise reference
        # that matters for it: LR noise for the two panels involving LR, HR
        # noise for SR-HR. Keeping them separate is the point -- the question
        # per panel is "is this residual at the noise level", which a shared
        # panel makes harder to read.
        lr_lab = r"Median LR noise: $\tilde{\sigma}_{\rm LR,data}(\lambda)$"
        hr_lab = r"Median HR noise: $\tilde{\sigma}_{\rm HR,data}(\lambda)$"
        panels = [
            (sigs[0], sigma_lr_med, lr_lab, False),
            (sigs[1], sigma_lr_med, lr_lab, False),
            (sigs[2], sigma_hr_med, hr_lab, True),
        ]
        for col, (sig, noise, noise_label, add_legend) in enumerate(panels):
            axS = fig.add_subplot(gs[1, col])
            axS.plot(wl, sig, lw=2.3,
                     label=r"Typical error: $\sigma_{\rm resid}(\lambda)$")
            if noise is not None:
                axS.plot(wl, noise, lw=2.0, ls="--", color="0.35", label=noise_label)
            axS.set_xlim(float(wl[0]), float(wl[-1]))
            axS.set_ylim(0.0, ymax)
            axS.set_xlabel(r"Wavelength [$\mu$m]")
            if col == 0:
                axS.set_ylabel("Robust scatter / noise amplitude")
            if add_legend:
                axS.legend(loc="upper right", frameon=False)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def plot_residual_psd(
    wavelength,
    residual_maps,
    labels=None,
    *,
    signal_maps=None,
    signal_labels=None,
    detrend_win: int = 101,
    use_hann: bool = True,
    show_band: bool = True,
    logy: bool = True,
    output_path: str | Path | None = None,
):
    """Median power spectra, with a 16-84 spread band, on one or two panels.

    The right-hand panel is the residuals -- what structure is *left over*, and
    at which scales. Pass ``signal_maps`` to get a left-hand panel showing the
    high-pass power of the products themselves (LR, SR, HR), which answers a
    different question: how much fine-scale structure each one carries in the
    first place. The two together separate "the model removed noise" from "the
    model removed everything", which neither shows alone.

    Detrending subtracts a running mean per spectrum, so both panels are the
    power in the *high-pass* component; without it the PSD is dominated by
    continuum shape. The dashed line marks 0.3 x Nyquist, the cut behind the
    high-frequency fraction in each legend entry -- higher means grainier.

    Parameters
    ----------
    residual_maps
        Sequence of ``(n_spectra, n_lambda)`` arrays, e.g. LR-HR, LR-SR, SR-HR.
    signal_maps
        Optional sequence of the products themselves, e.g. LR, SR, HR. Feed the
        same normalisation used for the residuals, or the two panels are not
        on comparable axes.

    Returns
    -------
    ``(fig, hf_fracs)`` where ``hf_fracs`` pairs each residual label with its
    fraction, so the numbers can go in the text without being read off the
    figure.
    """
    if labels is None:
        labels = [f"R{i + 1}" for i in range(len(residual_maps))]

    two = signal_maps is not None
    if two and signal_labels is None:
        signal_labels = [f"S{i + 1}" for i in range(len(signal_maps))]

    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(14.0, 5.4) if two else (9.2, 6.0))
    axes = np.atleast_1d(axes)
    ax_sig = axes[0] if two else None
    ax_res = axes[-1]

    def _draw(ax, maps, labs, *, with_hf):
        collected, freq = [], None
        for M, lab in zip(maps, labs, strict=True):
            freq, p50, p16, p84, hf = compute_psd_stats(
                wavelength, M, detrend_win=detrend_win, use_hann=use_hann)
            collected.append((lab, hf))
            ax.plot(freq[1:], p50[1:], lw=2.0,
                    label=f"{lab}  (HF frac={hf:.2f})" if with_hf else lab)
            if show_band:
                ax.fill_between(freq[1:], p16[1:], p84[1:], alpha=0.18)

        ax.set_xlabel(r"Frequency $k$ (cycles / $\mu$m)")
        ax.set_xlim(freq[1], freq.max())
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")

        nyq = float(freq.max())
        ax.axvline(0.3 * nyq, ls="--", lw=1.2)
        ax.text(0.305 * nyq, ax.get_ylim()[1] / (3.0 if logy else 1.2),
                r"$k > 0.3\,k_{\rm Nyq}$", rotation=90, va="top")
        return collected

    if two:
        _draw(ax_sig, signal_maps, signal_labels, with_hf=True)
        ax_sig.set_ylabel("High-pass power spectral density (arb.)")
        ax_sig.set_title("Signal (median across objects)")

    hf_fracs = _draw(ax_res, residual_maps, labels, with_hf=True)
    ax_res.set_ylabel("Residual power spectral density (arb.)")
    ax_res.set_title("Residuals (median across objects)"
                     if two else "Residual power spectrum (median across objects)")

    fig.tight_layout()
    if output_path is not None:
        save_figure(fig, output_path, dpi=300)
    return fig, hf_fracs


def plot_line_flux_comparison(
    rows,
    line_names,
    *,
    line_indices=None,
    pred_col: int = 6,
    sr1_col: int = 5,
    ncols: int = 4,
    outlier_factor: float = 2.0,
    axis_range: str = "all",
    hex_gridsize: int = 18,
    figsize=None,
    output_path: str | Path | None = None,
):
    """One-to-one SR vs HR integrated line flux, per line.

    This is the measurement the S/N figure cannot supply: S/N never references
    the HR truth, so it cannot show the lines are *correctly* modelled.
    Integrated flux against the HR reference can.

    Log-log, because line fluxes span orders of magnitude and the interesting
    failure is multiplicative -- a constant fractional loss is a parallel
    offset, legible here and not on linear axes. Both axes span the same
    decades, so a vertical displacement reads directly as a flux ratio.

    Laid out as a single row to match the S/N figure beside it in the paper, and
    coloured from the shared palette, with a black dashed one-to-one line as in
    the S/N figure.

    Only SR2 is drawn. SR1 is an intermediate stage rather than the product,
    and plotting it required three sentences of caption to stop it being
    misread: a log axis omits a much larger fraction of SR1 than of SR2, so the
    eye flatters the baseline. The SR1 comparison is quoted numerically in the
    text instead, where it cannot be misjudged.

    .. warning::

       A log axis silently drops non-positive values, and here they are not a
       rounding artefact: for 10--14% of lines SR2 emits no flux at all. Those
       are the failures the figure is for, so each panel reports the fraction
       numerically. Do not read the plotted cloud as the whole sample.

    ``axis_range`` selects between the two ways of framing the panels, which
    trade against each other and cannot both be had:

    ``"all"``
        Span every positive point, so nothing is pinned or clipped and the low
        tail is shown where it actually falls. SR2's tail runs far below
        anything the reference contains, so this costs 4--6 decades per panel
        against roughly 2.5 decades of reference, and since both axes must span
        the same decades the informative cloud is compressed into a corner.
    ``"reference"``
        Span the reference flux alone, so the cloud fills the panel. Predictions
        below the resulting floor are drawn pinned to the bottom edge as open
        downward markers and counted, so a severe under-prediction stays
        visible, but its true depth does not.

    Parameters
    ----------
    rows
        Array from ``scripts/flux_conservation.py``: columns are
        ``z, center_um, line_index, f_hr, sigma_hr, f_sr1, f_sr2, f_lines,
        f_cnn, presence, dv_kms``. New columns are appended, never inserted --
        ``pred_col``/``sr1_col`` and the hard-coded ``3`` below index by
        position, so an insertion would silently re-point them.
    line_names
        ``{line_index: display name}`` for the panels to draw.
    pred_col, sr1_col
        Which columns hold the prediction and the SR1 baseline.
    outlier_factor
        A line whose recovered flux is wrong by more than this factor either way
        counts as an outlier. The default of 2 is the flux counterpart of the
        redshift figure's catastrophic threshold: not a precision statement, but
        the point past which the measurement cannot carry a diagnostic.

    Returns
    -------
    ``(fig, stats)`` where ``stats`` maps each line to its median ratio and the
    total-flux ratio, so the numbers quoted in the text come from the same call
    that drew the figure.
    """
    rows = np.asarray(rows)
    idx = rows[:, 2].astype(int)
    if line_indices is None:
        line_indices = list(line_names)

    k = len(line_indices)
    ncols = int(ncols or k)
    nrows = -(-k // ncols)
    if figsize is None:
        # Matches plot_snr_comparison, so the two figures reduce to the text
        # width by the same factor and their type comes out the same size.
        figsize = (4.0 * ncols, 4.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for extra in axes[k:]:
        extra.set_visible(False)
    axes = axes[:k]
    stats = {}

    c_sr = COLOR_HR   # the off-scale markers, over the viridis density
    hb_last = None
    n_below_total = 0

    for i, (ax, li) in enumerate(zip(axes, line_indices, strict=True)):
        m = idx == li
        name = line_names[li]
        f_hr, f_sr, f_sr1 = rows[m, 3], rows[m, pred_col], rows[m, sr1_col]

        good = np.isfinite(f_hr) & np.isfinite(f_sr) & (f_hr > 0)

        if good.any():
            pos = good & (f_sr > 0)
            if axis_range == "reference":
                # See the note in the docstring: fills the panel, at the cost of
                # pinning the low tail to the floor.
                axlo = float(np.nanpercentile(f_hr[good], 1.0)) / 2.5
                axhi = float(np.nanpercentile(f_hr[good], 99.5)) * 2.5
            elif axis_range == "all":
                # Every positive point inside the frame. The margin is only
                # enough to keep the extreme markers off the spines.
                lo = float(np.min(f_hr[good]))
                hi = float(np.max(f_hr[good]))
                if pos.any():
                    lo = min(lo, float(np.min(f_sr[pos])))
                    hi = max(hi, float(np.max(f_sr[pos])))
                axlo, axhi = lo / 1.4, hi * 1.4
            else:
                raise ValueError(
                    f"axis_range must be 'all' or 'reference', got {axis_range!r}")

            above = pos & (f_sr >= axlo)
            below = pos & (f_sr < axlo)
            floor = axlo * 1.08

            # Density as a hexbin with a log count scale, styled and coloured as
            # the redshift figure is, so the two one-to-one comparisons in the
            # paper read the same way. Binned in log space because the axes are
            # logarithmic: hexagons of constant width in dex.
            lx, ly = np.log10(f_hr[above]), np.log10(f_sr[above])
            ext = (np.log10(axlo), np.log10(axhi), np.log10(axlo), np.log10(axhi))
            hb = ax.hexbin(lx, ly, gridsize=hex_gridsize, extent=ext, mincnt=1,
                           norm=mcolors.LogNorm(vmin=1), linewidths=0.0,
                           edgecolors="none", alpha=0.92, rasterized=True)
            hb_last = hb
            # The 68% highest-density region, as the redshift figure draws it.
            # Computed in log10 flux, the coordinates the panel is drawn in: the
            # same contour taken in linear flux would be a thin sliver hugging
            # the origin and would describe nothing.
            _draw_hdr_contours(ax, lx, ly, np.log10(axlo), np.log10(axhi))
            # Pinned to the floor rather than dropped -- these are the severe
            # under-predictions the figure exists to show, and a log axis would
            # otherwise carry them off the bottom without trace.
            n_below_total += int(below.sum())
            if below.any():
                ax.plot(np.log10(f_hr[below]),
                        np.full(int(below.sum()), np.log10(floor)),
                        ls="none", marker="v", ms=4.0, mfc="none", mec=c_sr,
                        mew=0.7, alpha=0.80, zorder=5)

            # Everything is drawn in log10 of the flux, so the axes stay linear
            # and the tick labels are formatted back into powers of ten. A log
            # *scale* cannot be combined with hexbin binning in log space.
            llo, lhi = np.log10(axlo), np.log10(axhi)
            ax.plot([llo, lhi], [llo, lhi], "k--", lw=1.0, alpha=0.5, zorder=6)
            ax.set_xlim(llo, lhi)
            ax.set_ylim(llo, lhi)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mticker.MultipleLocator(1))
                axis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"$10^{{{v:.0f}}}$"))

            n_np = int(np.sum(f_sr[good] <= 0))
            # SR1 is no longer drawn: plotting the intermediate stage forced
            # three sentences of caption to stop it being misread. Its counts
            # are still returned, because the text quotes them.
            n_np1 = int(np.sum(f_sr1[good] <= 0)) if np.isfinite(f_sr1[good]).any() else 0

            ratio = f_sr[good] / f_hr[good]
            total = float(f_sr[good].sum() / f_hr[good].sum())
            # The flux counterpart of the redshift figure's catastrophic-outlier
            # rate. There |dz|/(1+z) > 0.15 separates a usable redshift from a
            # misidentification; here a flux wrong by more than `outlier_factor`
            # is the measurement no diagnostic can be built on. Lines the model
            # did not emit at all have ratio 0 and are outliers by construction,
            # the same convention the doublet test uses for a lost line.
            f_out = float(outlier_factor)
            outlier = float(np.mean((ratio < 1.0 / f_out) | (ratio > f_out)))
            stats[name] = {"median_ratio": float(np.median(ratio)),
                           "total_ratio": total, "n": int(good.sum()),
                           "n_nonpositive_sr2": n_np, "n_nonpositive_sr1": n_np1,
                           "n_below_axis_sr2": int(below.sum()),
                           "outlier_rate": outlier, "outlier_factor": f_out}
            label = (f"n = {good.sum()}\n"
                     + r"median = " + f"{np.median(ratio):.2f}\n"
                     + r"$\Sigma$SR2/$\Sigma$HR = " + f"{total:.2f}\n"
                     # Named, not defined, in the box: the factor-of-two
                     # threshold is spelled out in the caption, as the redshift
                     # figure's |dz|/(1+z) cut is in its own.
                     + f"outlier rate: {100 * outlier:.0f}%\n"
                     + f"no flux emitted: {100 * n_np / good.sum():.0f}%")
            ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top",
                    ha="left", fontsize=9.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.80))
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.12)
        ax.set_xlabel("HR integrated flux")
        if i % ncols == 0:
            ax.set_ylabel("SR integrated flux")

    # Proxy handles rather than the drawn artists. The points are small and
    # semi-transparent so several hundred of them stay readable, and a legend
    # inheriting that came out as bare text with no visible marker at the size
    # this figure reduces to. These are opaque and larger, and they include the
    # off-scale marker, which is the one a reader cannot guess.
    from matplotlib.lines import Line2D

    handles = []
    # Only when something is actually pinned. Under ``axis_range="all"`` nothing
    # is, and a legend entry for an absent marker sends the reader hunting.
    if n_below_total:
        handles.append(
            Line2D([], [], ls="none", marker="v", markersize=6.5,
                   markerfacecolor="none", markeredgecolor=c_sr,
                   markeredgewidth=0.9, label="below axis"))
    handles.append(
        Line2D([], [], ls="--", color="k", alpha=0.5, lw=1.0, label="one-to-one"))
    axes[0].legend(handles=handles, loc="lower right", fontsize=9,
                   frameon=True, framealpha=0.85, edgecolor="0.7",
                   handlelength=1.3, handletextpad=0.6, borderpad=0.4,
                   labelspacing=0.35)

    fig.tight_layout()
    if hb_last is not None:
        cb = fig.colorbar(hb_last, ax=list(axes), fraction=0.020, pad=0.01)
        cb.set_label("Count per hex (log scale)")
    if output_path is not None:
        save_figure(fig, output_path)
    return fig, stats


def plot_snr_comparison(
    snr,
    lines=("OII3727", "Hbeta", "OIII5007", "Halpha"),
    titles=None,
    *,
    x_kind: str = "LR",
    y_kind: str = "SR",
    maxsn: float = 100.0,
    axis_max: float = 70.0,
    gridsize: int = 55,
    mincnt: int = 2,
    figsize=(16, 4),
    colorbar_fraction: float = 0.025,
    colorbar_pad: float = -0.17,
    output_path: str | Path | None = None,
):
    """Per-line S/N of one reconstruction against another, as a hexbin density.

    Hexbin rather than a scatter because there are tens of thousands of points
    and a scatter saturates into a black blob; the log colour scale is what
    makes both the dense locus and the sparse tail readable at once.

    Points above the one-to-one line are lines detected more confidently after
    super-resolution. Note the caveat in :mod:`specsr.linefit`: this S/N never
    references the HR truth, so it cannot show the lines are *correct* — pair it
    with :func:`plot_line_flux_comparison`, which does.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if titles is None:
        titles = {"Halpha": r"H$\alpha$", "OII3727": r"[O II] $\lambda3727$",
                  "OIII5007": r"[O III] $\lambda5007$", "Hbeta": r"H$\beta$"}

    fig, axes = plt.subplots(1, len(lines), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    hb_last = None

    for ax, ln in zip(axes, lines, strict=True):
        x = np.asarray(snr[f"{ln}_sn_{x_kind}"], float)
        y = np.asarray(snr[f"{ln}_sn_{y_kind}"], float)
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x0, y0 = x[good], y[good]

        inr = (x0 <= maxsn) & (y0 <= maxsn)
        if inr.any():
            hb = ax.hexbin(x0[inr], y0[inr], gridsize=gridsize, mincnt=mincnt, bins="log")
            if hb.get_array() is not None and hb.get_array().size:
                hb_last = hb
            # No 68% contour here, unlike the flux and redshift comparisons.
            # This density piles against S/N(LR) = 0 with a long tail in
            # S/N(SR), so the region is a ridge banked against the axis rather
            # than a closed core: the contour cannot close inside the frame, and
            # it reads as a boundary the data respect rather than as the two
            # thirds it encloses.

        # A count box, as the line-flux and redshift figures carry, so all three
        # comparisons state their sample size on the figure itself.
        #
        # The second line is not decoration. The axes stop at ``axis_max`` while
        # the data run to ``maxsn``, and the overflow is almost entirely
        # one-sided -- points where the reconstruction scores far higher than
        # its input, which is to say this figure's own result. For H-alpha that
        # is half the sample, and the median SR value quoted in the text sits
        # outside the frame. A panel reporting ``n`` without it would be
        # counting objects the reader cannot see.
        #
        # Phrased as the detection it is rather than as a plotting casualty:
        # these lines are not missing, they are measured above the strongest S/N
        # the frame can show. Deliberately not called "up-scaled" -- upscaling is
        # the resampling operation in this paper, and every line underwent it.
        off = int(np.sum((x0 > axis_max) | (y0 > axis_max)))
        label = f"n = {x0.size}"
        if off:
            label += f"\nS/N > {axis_max:g}: {100 * off / x0.size:.0f}%"
        # Upper *right*, unlike the other two figures, which put the box upper
        # left. Here that corner holds the tall left-hand column of large gains
        # -- 28 to 52 visible points per panel -- while the upper right holds
        # none in any panel. Matching the other figures' corner would mean
        # covering data to do it.
        ax.text(0.96, 0.96, label, transform=ax.transAxes, va="top", ha="right",
                fontsize=9.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80))

        ax.plot([0, maxsn], [0, maxsn], "k--", lw=1.0, alpha=0.5)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_title(titles.get(ln, ln), fontsize=12)
        ax.set_xlabel(f"S/N ({x_kind})")

    axes[0].set_ylabel(f"S/N ({y_kind})")
    fig.tight_layout()
    if hb_last is not None:
        # Attach to the list of axes so the bar sits outside the last panel;
        # a negative pad here overlaps it onto the data.
        cbar = fig.colorbar(hb_last, ax=list(axes), fraction=colorbar_fraction,
                            pad=max(colorbar_pad, 0.01))
        cbar.set_label(r"$\log_{10}(\mathrm{count})$")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def plot_augmentation_family(
    wavelength_low,
    wavelength_high,
    flux_low,
    flux_high,
    z,
    *,
    z_original: float | None = None,
    figsize=(12, 5),
    output_path: str | Path | None = None,
):
    """One galaxy's original spectrum and its redshift-shifted augmentations.

    Shows what augmentation actually does to the data, and by extension why the
    split must be drawn over parent galaxies: these are near-duplicates of one
    another, so an augmented sibling in training makes a held-out original
    trivially predictable.

    Augmentations are coloured by redshift offset through a shared colourbar
    rather than a legend entry each — with twenty shifts a legend would cover
    the data it is labelling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    z = np.asarray(z, dtype=float)
    z0 = float(z[0]) if z_original is None else float(z_original)
    dz = z[1:] - z0
    order = np.argsort(dz) + 1

    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=float(dz.min()), vmax=float(dz.max())) \
        if dz.size else mcolors.Normalize(0, 1)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, wl, flux, title in (
        (axs[0], wavelength_low, flux_low, "Low-Resolution"),
        (axs[1], wavelength_high, flux_high, "High-Resolution"),
    ):
        for i in order:
            ax.plot(wl, flux[i], color=cmap(norm(z[i] - z0)), ls="--",
                    alpha=0.7, lw=0.8, zorder=2)
        ax.plot(wl, flux[0], color="k", lw=1.4, zorder=3,
                label=f"original z = {z0:.4f}")
        ax.text(0.012, 0.94, title, transform=ax.transAxes, va="top", ha="left",
                fontsize=11)
        ax.set_ylabel(r"$F_\lambda$ (normalized)")

    axs[0].legend(frameon=False, loc="upper right", fontsize=11)
    axs[1].set_xlabel("Wavelength (μm)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axs), fraction=0.02, pad=0.01)
    cb.set_label(r"$\Delta z$")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig


#: Disperser colours for the coverage figure. Chosen to stay distinguishable in
#: greyscale and for common colour-vision deficiencies, which a default cycle of
#: four does not guarantee.
DISPERSER_COLORS = {
    "g140m": "#388E8E",
    "g235m": "#007FFF",
    "g395m": "#1F4045",
    # Sampled from the published figure. The preprocessing notebook still says
    # "#601936"; the figure that shipped uses this coral, so match the figure.
    "prism": "#FB5252",
}


def plot_disperser_coverage(
    spectra,
    z: float,
    lines=None,
    *,
    ylim=(-1, 6),
    figsize=(12, 4),
    label_min_separation_um: float = 0.04,
    output_path: str | Path | None = None,
):
    """The three medium gratings and the prism for one galaxy, on native grids.

    Motivates the whole setup: the prism covers 1--5 um in one shot but blends
    lines, while the gratings resolve them and each covers only a slice. The
    reference spectrum is these three stitched together.

    Plotted on their **own** wavelength arrays, not resampled — the point is
    what each disperser natively delivers.

    Parameters
    ----------
    spectra
        ``{"g140m": (wave, flux), ..., "prism": (wave, flux)}``. Flux should be
        normalised; missing keys are skipped.
    lines
        ``{name: rest_wavelength_um}`` to mark, redshifted by ``z``. Crowded
        labels are thinned by ``label_min_separation_um`` so they stay readable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for key in ("g140m", "g235m", "g395m", "prism"):
        if key not in spectra:
            continue
        w, f = spectra[key]
        ax.plot(np.asarray(w), np.asarray(f), label=key.upper() if key != "prism" else "Prism",
                color=DISPERSER_COLORS[key], lw=2.0 if key == "prism" else 1.2,
                zorder=3 if key == "prism" else 2)

    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel(r"$F_\lambda$ (normalized)")
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper right")

    all_w = np.concatenate([np.asarray(w) for w, _ in spectra.values()])
    x_min, x_max = float(np.nanmin(all_w)), float(np.nanmax(all_w))
    ax.set_xlim(x_min, x_max)

    if lines:
        obs = sorted(((lam * (1 + z), name) for name, lam in lines.items()),
                     key=lambda t: t[0])
        obs = [(o, n) for o, n in obs if x_min <= o <= x_max]
        last_labelled = -np.inf
        for o, name in obs:
            ax.axvline(o, color="0.80", ls="--", lw=0.9, zorder=1)
            # Label only when there is room; in crowded groups the marker still
            # appears, so nothing is hidden, but the text does not overlap.
            if o - last_labelled >= label_min_separation_um:
                ax.text(o, ylim[1] * 0.98, name, rotation=90, va="top", ha="center",
                        fontsize=8, color="0.25")
                last_labelled = o

    fig.tight_layout()
    if output_path is not None:
        save_figure(fig, output_path)
    return fig


#: Lines marked on the disperser-coverage figure, rest wavelength in microns.
COVERAGE_LINES = {
    "HeII": 0.1640, "CIII]": 0.1909, "CII]": 0.2326, "OII]": 0.2471,
    "MgII": 0.2799, "OIII": 0.3346, "[NeV]": 0.3426, "[OII]": 0.3727,
    r"H$\delta$": 0.4102, "[OIII]": 0.4363, r"H$\beta$": 0.4861,
    "[OIII] ": 0.5007, "HeI": 0.5876, r"H$\alpha$": 0.6563, "[SII]": 0.6716,
    "[SIII]": 0.9069, "[SIII] ": 0.9531,
}


# ---------------------------------------------------------------------------
# Redshift comparison (paper Fig. redshift_comparison)
# ---------------------------------------------------------------------------

#: Panel order and titles, matching the published figure.
REDSHIFT_PANELS = ("Low-res baseline", "SR2 z-head", "High-res oracle")


def _metrics_box_text(m: dict) -> str:
    """The five lines in each panel's box, in the published order.

    ``NMAD`` here is the raw-residual variant, matching the published figure.
    See :func:`specsr.metrics.redshift_metrics` for why that is not the same as
    ``sigma_nmad``.
    """
    return "\n".join([
        # First line, as in the line-flux and S/N figures: how many objects the
        # rest of the box is computed from, before any of the summaries.
        f"n = {m['n']}",
        f"MAE: {m['mae']:.4f}",
        f"RMSE: {m['rmse']:.4f}",
        f"NMAD: {m['nmad']:.4f}",
        f"Med |dz|/(1+z): {m['med_abs_dz_over_1pz']:.4f}",
        f"Outlier rate: {m['outlier_frac'] * 100:.2f}%",
    ])


def _hdr_threshold(density, mass: float = 0.68):
    """Highest-density-region threshold: the level enclosing ``mass`` of the total.

    Returns ``None`` for an empty or degenerate grid, so the caller can skip the
    contour rather than draw a meaningless one at zero.
    """
    d = np.asarray(density, dtype=np.float64)
    total = float(d.sum())
    if total <= 0:
        return None
    flat = d.ravel()
    order = np.argsort(flat)[::-1]
    k = int(np.searchsorted(np.cumsum(flat[order]), mass * total))
    t = float(flat[order][min(max(k, 0), flat.size - 1)])
    return t if np.isfinite(t) and t > 0 else None


def _density_grid(z_true, z_pred, bins, zmin, zmax, smooth_sigma):
    hist, xedges, yedges = np.histogram2d(
        z_true, z_pred, bins=bins, range=[[zmin, zmax], [zmin, zmax]]
    )
    hist = hist.T
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            hist = gaussian_filter(hist, sigma=float(smooth_sigma))
        except ImportError:
            pass  # contours get blockier without scipy, but still render
    return 0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]), hist


def _draw_hdr_contours(ax, x, y, lo, hi, *, bins: int = 40,
                       smooth_sigma: float = 1.15, masses=(0.68,)):
    """Highest-density-region contours over a hexbin panel.

    Shared by all three one-to-one comparisons -- S/N, integrated flux and
    redshift -- so the line means the same thing in each: the smallest region
    containing ``mass`` of the plotted sample. Read off a smoothed 2D histogram
    rather than fitted, because none of these distributions is an ellipse. Each
    is a ridge along the one-to-one line plus a scatter of failures off it, and
    a moment-based ellipse would be pulled off the ridge by the failures and
    then describe neither population.

    ``lo``/``hi`` bound both axes: these panels are square by construction, and
    the contour must be computed in the coordinates the panel is drawn in --
    log10 flux for the flux figure, not flux.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return
    xc, yc, dens = _density_grid(x, y, bins, lo, hi, smooth_sigma)
    for mass in masses:
        level = _hdr_threshold(dens, mass=mass)
        if level is None:
            continue
        # 68% thick, tail contours thin -- the eye should read the core first.
        lw, alpha = (2.2, 0.95) if mass <= 0.70 else (1.3, 0.80)
        ax.contour(xc, yc, dens, levels=[level], linewidths=lw, alpha=alpha)


def plot_redshift_panel(
    z_true,
    z_pred,
    *,
    title: str = "z-head",
    ax=None,
    output_path=None,
    outlier_thresh: float = 0.15,
    limits=None,
    smooth_sigma: float = 1.15,
):
    """One predicted-vs-true redshift panel, with the published metrics box.

    Same styling and the same :func:`_metrics_box_text` as the three-panel paper
    figure, so a plot watched during training and a plot printed in the paper
    cannot disagree about what "outlier rate" means.

    This exists to be logged every epoch: a redshift run's scalars do not show
    *how* it is failing, and the difference between "scatter about the diagonal"
    and "a second locus off the diagonal from line misidentification" is exactly
    what the scatter shows and no single number does. Returns ``(fig, metrics)``;
    ``fig`` is ``None`` when drawing into a caller-supplied ``ax``.
    """
    z_true = np.asarray(z_true, dtype=np.float64).ravel()
    z_pred = np.asarray(z_pred, dtype=np.float64).ravel()
    if z_true.size != z_pred.size:
        raise ValueError(f"z_true has {z_true.size} values, z_pred has {z_pred.size}")

    n = z_true.size
    hex_gridsize = 85 if n >= 10_000 else (40 if n >= 2_000 else 22)
    mincnt = 3 if n >= 10_000 else 1
    contour_bins = 170 if n >= 10_000 else (80 if n >= 2_000 else 40)
    hdr_masses = (0.68, 0.95) if n >= 2_000 else (0.68,)
    show_points = n < 2_000

    if limits is None:
        pooled = np.concatenate([z_true, z_pred])
        zmin, zmax = np.percentile(pooled, [0.5, 99.5])
        zmin = min(float(zmin), 0.0)
        zmax = float(zmax)
        if zmax <= zmin:
            zmax = zmin + 1.0
    else:
        zmin, zmax = (float(v) for v in limits)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)

    ax.hexbin(z_true, z_pred, gridsize=hex_gridsize,
              extent=(zmin, zmax, zmin, zmax), mincnt=mincnt,
              norm=mcolors.LogNorm(vmin=max(mincnt, 1)),
              linewidths=0.0, edgecolors="none", alpha=0.92, rasterized=True)

    _draw_hdr_contours(ax, z_true, z_pred, zmin, zmax, bins=contour_bins,
                       smooth_sigma=smooth_sigma, masses=hdr_masses)

    if show_points:
        ax.plot(z_true, z_pred, ".", ms=2.6, mew=0, color="0.15",
                alpha=0.55, rasterized=True, zorder=3)

    ax.plot([zmin, zmax], [zmin, zmax], lw=1.8, alpha=0.95, zorder=4)

    m = redshift_metrics(z_true, z_pred, outlier_thresh=outlier_thresh)
    ax.text(0.04, 0.96, _metrics_box_text(m), transform=ax.transAxes,
            va="top", fontsize=11.0,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.80))

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("True Redshift")
    ax.set_ylabel("Predicted Redshift")
    ax.grid(True, alpha=0.12)

    if output_path is not None and fig is not None:
        save_figure(fig, output_path)
    return fig, m


def plot_redshift_comparison(
    panels,
    *,
    titles=REDSHIFT_PANELS,
    output_path=None,
    hex_gridsize: int | None = None,
    mincnt: int | None = None,
    contour_bins: int | None = None,
    smooth_sigma: float = 1.15,
    hdr_masses=None,
    outlier_thresh: float = 0.15,
    limits=None,
    ncols: int | None = None,
    panel_size=None,
):
    """Predicted vs true redshift for the LR / SR / HR arms, side by side.

    ``panels`` is a sequence of ``(z_true, z_pred)`` pairs -- three for the
    published LR/SR/HR figure, four once the raw prism and the grating reference
    are shown alongside SR1 and SR2. Every panel shares axis limits and colour
    normalisation, which is the whole point: the panels are meant to be compared
    by eye, and per-panel scaling would hide the effect. ``titles`` must have one
    entry per panel.

    **The binning adapts to sample size, and it has to.** The published version
    of this figure was made on the augmented evaluation set — tens of thousands
    of rows — where an 85x85 hex grid with ``mincnt=3`` gives a smooth density.
    The leak-free split has 286 real spectra, which over that same grid is 0.04
    points per cell: with ``mincnt=3`` essentially nothing renders, and the HDR
    contours would be fitted to noise. Defaults are therefore chosen from ``n``
    unless given explicitly. Pass ``hex_gridsize``/``mincnt``/``contour_bins`` to
    reproduce the old figure exactly.
    """
    panels = [(np.asarray(t, dtype=np.float64).ravel(),
               np.asarray(p, dtype=np.float64).ravel()) for t, p in panels]
    titles = list(titles)
    if not panels:
        raise ValueError("no panels given")
    # Titles are zipped against panels below. Zipping non-strictly would drop
    # the trailing panel whenever more panels than titles are passed -- which is
    # exactly what happens when the default three-arm titles meet a four-arm
    # comparison, and it fails silently by rendering a figure that is simply
    # missing an arm.
    if len(titles) != len(panels):
        raise ValueError(
            f"got {len(panels)} panels but {len(titles)} titles; pass one title "
            "per panel (the default REDSHIFT_PANELS describes three arms)"
        )
    for i, (t, p) in enumerate(panels):
        if t.size != p.size:
            raise ValueError(f"panel {i}: z_true has {t.size} values, z_pred has {p.size}")

    n = min(t.size for t, _ in panels)
    # Small samples need coarse cells and mincnt=1, or the panel is blank. The
    # thresholds are judgement, not theory; they are here so the figure degrades
    # visibly rather than silently rendering almost nothing.
    if hex_gridsize is None:
        hex_gridsize = 85 if n >= 10_000 else (40 if n >= 2_000 else 22)
    if mincnt is None:
        mincnt = 3 if n >= 10_000 else 1
    if contour_bins is None:
        contour_bins = 170 if n >= 10_000 else (80 if n >= 2_000 else 40)
    if hdr_masses is None:
        # The 95% contour traces the tails, and with a few hundred points the
        # tails are individual galaxies: each isolated outlier acquires its own
        # closed loop, which reads as a density feature and is one object. Keep
        # the 68% core contour, which is supported, and drop the tail one.
        hdr_masses = (0.68, 0.95) if n >= 2_000 else (0.68,)

    if limits is None:
        pooled = np.concatenate([a for pair in panels for a in pair])
        zmin, zmax = np.percentile(pooled, [0.5, 99.5])
        zmin = min(float(zmin), 0.0)
        zmax = float(zmax)
        if zmax <= zmin:
            zmax = zmin + 1.0
    else:
        zmin, zmax = (float(v) for v in limits)

    # One shared LogNorm across every panel, estimated from the pooled counts so
    # the colourbar means the same thing in each.
    vmax = mincnt + 1.0
    for t, p in panels:
        h, _, _ = np.histogram2d(t, p, bins=hex_gridsize,
                                 range=[[zmin, zmax], [zmin, zmax]])
        vmax = max(vmax, float(h.max()))
    norm = mcolors.LogNorm(vmin=max(mincnt, 1), vmax=max(vmax, mincnt + 1.0))

    k = len(panels)
    ncols = int(ncols or k)
    nrows = -(-k // ncols)
    # Per-panel geometry matching plot_line_flux_comparison, so a row of four
    # reduces to the manuscript's text width by the same factor as the flux
    # figure and the two one-to-one comparisons come out with type of the same
    # size on the page. An earlier version used 6.34 in panels, at which width a
    # row of four does put the metrics box at about 3 pt; the fix is a panel
    # sized for the layout, not a wrapped layout. Box text is tied to the panel
    # width for the same reason -- change one and the other follows.
    pw, ph = panel_size or (4.0, 4.4)
    box_fs = 9.5 * pw / 4.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for extra in axes[k:]:
        extra.set_visible(False)
    axes = axes[:k]
    hb = None
    stats = []
    for i, (ax, (z_true, z_pred), title) in enumerate(zip(axes, panels, titles, strict=True)):
        hb = ax.hexbin(
            z_true, z_pred, gridsize=hex_gridsize,
            extent=(zmin, zmax, zmin, zmax), mincnt=mincnt, norm=norm,
            linewidths=0.0, edgecolors="none", alpha=0.92, rasterized=True,
        )

        _draw_hdr_contours(ax, z_true, z_pred, zmin, zmax, bins=contour_bins,
                           smooth_sigma=smooth_sigma, masses=hdr_masses)

        # Black dashed, as plot_line_flux_comparison draws it. It was a solid
        # C0 line, which put the one reference the reader is measuring against
        # in a colour the density map also uses, and made it the most prominent
        # thing in the panel rather than the quietest.
        ax.plot([zmin, zmax], [zmin, zmax], "k--", lw=1.0, alpha=0.5, zorder=4)

        m = redshift_metrics(z_true, z_pred, outlier_thresh=outlier_thresh)
        stats.append(m)
        ax.text(0.04, 0.96, _metrics_box_text(m), transform=ax.transAxes,
                va="top", fontsize=box_fs,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80))

        ax.set_xlim(zmin, zmax)
        ax.set_ylim(zmin, zmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("True Redshift")
        # Left column only, which is not "panel 0" once the panels wrap.
        ax.set_ylabel("Predicted Redshift" if i % ncols == 0 else "")
        ax.grid(True, alpha=0.12)

    cb = fig.colorbar(hb, ax=list(axes), fraction=0.020, pad=0.01)
    cb.set_label("Count per hex (log scale)")

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, stats
