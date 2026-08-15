"""Residual and power-spectrum statistics used by the paper figures.

Computation lives here; :mod:`specsr.plotting` draws it. That split is the point:
a number that goes in the text should be obtainable without importing matplotlib
or rendering anything, and it should be testable.

Extracted from the evaluation notebooks, where several of these were redefined
in three separate cells with slightly different names
(``_nan_moving_average_1d`` / ``nan_moving_average_1d``) — one of the ways a
notebook quietly ends up running a different function than the one you read.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "redshift_metrics",
    "compute_psd_stats",
    "doublet_dip_depth",
    "estimate_median_noise_lambda",
    "moving_average_rows",
    "robust_sigma_lambda",
    "rank_doublet_examples",
    "robust_symmetric_vlim",
]


def moving_average_rows(X, win: int):
    """NaN-aware moving average along each row.

    Gaps are skipped rather than treated as zero: the denominator counts only
    finite samples, so a masked detector region does not drag the local mean
    toward zero and invent a fake absorption trough.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
        squeeze = True
    else:
        squeeze = False
    win = int(win) | 1  # force odd so the window is symmetric
    k = np.ones(win, dtype=np.float64)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        y = X[i]
        finite = np.isfinite(y)
        num = np.convolve(np.where(finite, y, 0.0), k, mode="same")
        den = np.convolve(finite.astype(np.float64), k, mode="same")
        out[i] = np.where(den > 0, num / den, np.nan)
    return out[0] if squeeze else out


def robust_symmetric_vlim(residual_arrays, vmax_abs_pct: float = 97.0):
    """Symmetric colour limits at a percentile of ``|residual|``.

    Symmetric because these are signed residuals and a diverging colormap must
    put zero at the centre; percentile-based because a handful of bad pixels
    would otherwise set the scale and flatten everything else to white.
    """
    R = np.concatenate([np.asarray(r).reshape(-1) for r in residual_arrays])
    R = R[np.isfinite(R)]
    mx = float(np.percentile(np.abs(R), vmax_abs_pct)) if R.size else 1.0
    return -mx, mx


def robust_sigma_lambda(R, min_count: int = 30, clip_sigma: float = 6.0):
    """Per-wavelength robust scatter of a residual map, via clipped MAD.

    MAD rather than standard deviation because emission-line residuals are
    heavy-tailed: a few galaxies with a strong line at one wavelength would
    dominate an unclipped estimate and read as an instrumental feature.

    Returns NaN at wavelengths with fewer than ``min_count`` finite samples,
    rather than a number computed from too little data.
    """
    R = np.asarray(R, dtype=np.float64)
    n_lambda = R.shape[1]
    sig = np.full(n_lambda, np.nan, dtype=np.float64)

    # Degeneracy guard, set from the data rather than as an absolute constant.
    # The notebook added a flat 1e-12 to every MAD; residuals here are ~1e-21 in
    # physical units, so that term *was* the answer -- every wavelength reported
    # sigma = 1e-12 and the panel plotted a flat line off-scale.
    finite = R[np.isfinite(R)]
    tiny = 1e-12 * float(np.median(np.abs(finite))) if finite.size else 0.0

    for j in range(n_lambda):
        col = R[:, j]
        col = col[np.isfinite(col)]
        if col.size < min_count:
            continue
        med = np.median(col)
        mad = 1.4826 * np.median(np.abs(col - med)) + tiny
        if clip_sigma > 0 and mad > 0:
            keep = (col >= med - clip_sigma * mad) & (col <= med + clip_sigma * mad)
            col = col[keep]
            if col.size < min_count:
                continue
            mad = 1.4826 * np.median(np.abs(col - np.median(col))) + tiny
        sig[j] = mad
    return sig.astype(np.float32)


def estimate_median_noise_lambda(X, smooth_win: int = 71, local_win: int = 31,
                                 min_count_per_lambda: int = 80):
    """Median per-wavelength noise, estimated from high-frequency scatter.

    Smooths each spectrum, takes the residual against its own smooth version as
    a noise proxy, and reports the median across spectra. This measures the
    *data's* noise without needing the reported uncertainties, which is what
    makes it a usable reference line on a residual plot.
    """
    X = np.asarray(X, dtype=np.float64)
    resid = X - moving_average_rows(X, smooth_win)
    sigma_i = np.sqrt(moving_average_rows(resid * resid, local_win))

    n_lambda = X.shape[1]
    out = np.full(n_lambda, np.nan, dtype=np.float64)
    for j in range(n_lambda):
        col = sigma_i[:, j]
        col = col[np.isfinite(col)]
        if col.size >= min_count_per_lambda:
            out[j] = np.median(col)
    return out.astype(np.float32)


def compute_psd_stats(wavelength, R, detrend_win: int = 101, use_hann: bool = True):
    """PSD of a residual map: median, 16-84 band, and high-frequency fraction.

    Parameters
    ----------
    wavelength
        Uniformly spaced grid; only the spacing is used, so the frequency axis
        is cycles per unit of whatever coordinate this is.
    R
        ``(n_spectra, n_lambda)`` residual map.
    detrend_win
        Remove slow structure per spectrum first. Without it the PSD is
        dominated by the continuum shape and says nothing about the fine
        structure the figure is about.

    Returns
    -------
    ``(freq, p50, p16, p84, hf_frac)``. ``hf_frac`` is the median fraction of
    power above 0.3 x Nyquist -- a single number for "how grainy", which is what
    lets three curves be compared in a legend.
    """
    wl = np.asarray(wavelength, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_lambda = R.shape[1]
    if n_lambda != wl.size:
        raise ValueError(f"R has {n_lambda} columns but wavelength has {wl.size}")

    freq = np.fft.rfftfreq(n_lambda, d=float(np.median(np.diff(wl))))

    X = R.copy()
    if detrend_win is not None and int(detrend_win) >= 3:
        X = X - moving_average_rows(X, detrend_win)

    # A row that is mostly gaps gives a meaningless spectrum; drop it rather
    # than let zero-filled NaNs masquerade as signal.
    keep = np.isfinite(X).sum(axis=1) > (0.8 * n_lambda)
    X = X[keep]
    if X.shape[0] == 0:
        raise ValueError("no spectra with enough finite samples for a PSD")

    window = np.hanning(n_lambda) if use_hann else np.ones(n_lambda)
    wnorm = float(np.mean(window * window))
    X = np.where(np.isfinite(X), X, 0.0) * window[None, :]

    power = np.abs(np.fft.rfft(X, axis=1)) ** 2 / (n_lambda * wnorm)
    p50 = np.nanmedian(power, axis=0)
    p16 = np.nanpercentile(power, 16, axis=0)
    p84 = np.nanpercentile(power, 84, axis=0)

    hf_mask = freq >= 0.3 * freq.max()
    total = np.sum(power, axis=1)
    good = total > 0
    hf_frac = float(np.median(np.sum(power[:, hf_mask], axis=1)[good] / total[good])) \
        if good.any() else float("nan")

    return freq, p50, p16, p84, hf_frac


def doublet_dip_depth(flux, wavelength, z, lam1: float = 0.4959, lam2: float = 0.5007):
    """How resolved a close doublet is: 1 = two clean peaks, 0 = one blended bump.

    Measures the valley between the two lines relative to the weaker peak, above
    a local continuum taken from sidebands outside the pair. This is the
    quantity the paper's spectrum figure is chosen to display — whether the
    [O III] doublet, unresolved at prism resolution, is separated after
    super-resolution.

    Returns ``(depth, peak1, peak2)``, or ``None`` if the pair is off the grid
    or there is no positive signal to measure.
    """
    y = np.asarray(flux, dtype=np.float64)
    w = np.asarray(wavelength, dtype=np.float64)
    a, b = lam1 * (1 + z), lam2 * (1 + z)
    if a < w[0] or b > w[-1]:
        return None

    span = b - a
    sb = (((w > a - 3 * span) & (w < a - 1.2 * span))
          | ((w > b + 1.2 * span) & (w < b + 3 * span)))
    if sb.sum() < 10:
        return None
    cont = float(np.median(y[sb]))

    ia, ib = int(np.argmin(np.abs(w - a))), int(np.argmin(np.abs(w - b)))
    win = max(2, int(0.25 * (ib - ia)))
    p1 = float(np.max(y[max(0, ia - win):ia + win + 1])) - cont
    p2 = float(np.max(y[ib - win:min(y.size, ib + win + 1)])) - cont
    valley = (float(np.min(y[ia + win:ib - win])) - cont
              if (ib - win) > (ia + win) else min(p1, p2))
    peak = min(p1, p2)
    if peak <= 0:
        return None
    return float(np.clip((peak - valley) / peak, 0.0, 1.0)), p1, p2


def _peak_offset(flux, wavelength, z, lam_rest, search_frac: float = 0.4):
    """Distance from a line's expected position to the nearest local maximum.

    Returned in units of the doublet separation, so it is comparable across
    redshift.
    """
    w = np.asarray(wavelength, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    sep = (0.5007 - 0.4959) * (1 + z)
    target = lam_rest * (1 + z)
    half = search_frac * sep
    m = (w > target - half) & (w < target + half)
    if m.sum() < 3:
        return np.inf
    return float(abs(w[m][int(np.argmax(y[m]))] - target) / sep)


def rank_doublet_examples(wavelength, flux_lr, flux_sr, flux_hr, z, *,
                          z_pred=None, hr_min: float = 0.35, lr_max: float = 0.20,
                          sr_min: float = 0.40, amp_percentile: float = 40.0,
                          max_peak_offset: float = 0.25,
                          max_dz_over_1pz: float = 0.02,
                          amp_ratio_range: tuple[float, float] = (0.4, 2.5)):
    """Rank spectra by how well they demonstrate the doublet being resolved.

    Selects objects where the doublet is genuinely separated in the HR reference,
    blended in the LR input, and separated again in the super-resolved output —
    so the figure shows the model recovering structure that is really there,
    rather than a case where the reference was already ambiguous.

    **Position is checked, not just separation.** SR2 conditions its line branch
    on the predicted redshift, so when that prediction is wrong it emits a
    well-separated doublet at the wrong wavelength. Ranking on dip depth alone
    selects exactly those cases: two crisp peaks that are not where the lines
    are. ``max_peak_offset`` requires each SR peak to land within a fraction of
    the doublet separation of its true position, and ``max_dz_over_1pz``
    optionally rejects objects whose redshift is badly predicted.

    **Amplitude is checked for the same reason.** This function used to score on
    the *SR* peak height, which rewards emitting hard: the top-ranked example on
    the released model over-emitted [O III] 5007 by 18.8x, the 99.7th percentile
    of the held-out set, where the median galaxy sits at 0.55. It made a figure
    whose HR reference looked like a flat line beside the model's output — an
    unrepresentative worst case presented as the headline illustration.
    ``amp_ratio_range`` bounds the SR/HR peak ratio, and the score now ranks on
    the *HR* line brightness, which is what makes an example legible on the page
    without rewarding the model for overshooting.

    Note this selects examples that are faithful in amplitude; it is an
    illustration, not a measurement. The distribution of SR/HR line flux is the
    quantity to quote, not these panels.

    Returns indices, best first.
    """
    z = np.asarray(z)
    z_pred = None if z_pred is None else np.asarray(z_pred)
    idx, d_hr, d_lr, d_sr, amp, off, dz, ratio = [], [], [], [], [], [], [], []
    for i in range(len(z)):
        rh = doublet_dip_depth(flux_hr[i], wavelength, z[i])
        rl = doublet_dip_depth(flux_lr[i], wavelength, z[i])
        rs = doublet_dip_depth(flux_sr[i], wavelength, z[i])
        if rh is None or rl is None or rs is None:
            continue
        idx.append(i)
        d_hr.append(rh[0])
        d_lr.append(rl[0])
        d_sr.append(rs[0])
        # Brightness is taken from the HR reference, not from SR: this ranks on
        # how visible the real line is, rather than on how hard the model emitted.
        amp.append(min(rh[1], rh[2]))
        # Amplitude fidelity at the stronger component (5007).
        ratio.append(rs[2] / rh[2] if rh[2] > 0 else np.inf)
        off.append(max(_peak_offset(flux_sr[i], wavelength, z[i], 0.4959),
                       _peak_offset(flux_sr[i], wavelength, z[i], 0.5007)))
        dz.append(abs(z_pred[i] - z[i]) / (1 + z[i]) if z_pred is not None else 0.0)
    if not idx:
        return np.array([], dtype=int)

    idx = np.asarray(idx)
    d_hr, d_lr, d_sr, amp, off, dz, ratio = map(
        np.asarray, (d_hr, d_lr, d_sr, amp, off, dz, ratio))
    lo, hi = amp_ratio_range
    ok = ((d_hr > hr_min) & (d_lr < lr_max) & (d_sr > sr_min)
          & (amp > np.percentile(amp, amp_percentile))
          & (off <= max_peak_offset) & (dz <= max_dz_over_1pz)
          & (ratio >= lo) & (ratio <= hi))
    # Prefer a large LR->SR change, weighted by line strength so the chosen
    # example is visible on the page rather than a marginal detection.
    score = (d_sr - d_lr) * np.sqrt(np.clip(amp, 0, None) / max(amp.max(), 1e-300))
    score = np.where(ok, score, -np.inf)
    order = np.argsort(-score)
    return idx[order][np.isfinite(score[order])]


def redshift_metrics(z_true, z_pred, outlier_thresh: float = 0.15) -> dict[str, float]:
    """Redshift-recovery summaries, defined once for the whole codebase.

    Used by the redshift head's training loop, by the W&B sweeps that rank its
    hyperparameters, and by the paper figure. That matters: both zhead sweeps
    previously optimised ``val_med_abs_dz_over_1pz``, a key the training loop
    never logged, so 75 Bayes trials ran with nothing to rank them by. A metric
    the paper quotes and a metric the sweep optimises must be the same function.

    Two robust scatters are returned and they are **not** interchangeable:

    ``nmad``
        ``1.4826 * median(|dz|)`` on the raw residual. This is what the
        published figure's "NMAD" box reports.
    ``sigma_nmad``
        the same statistic on ``dz/(1+z)``. This is the usual photo-z
        convention and is a smaller number.

    Quote whichever you like, but say which one — they differ by roughly a
    factor of ``1+z``, which is a factor of a few at these redshifts.
    """
    import numpy as _np

    z_true = _np.asarray(z_true, dtype=_np.float64).ravel()
    z_pred = _np.asarray(z_pred, dtype=_np.float64).ravel()
    dz = z_pred - z_true
    rel = dz / (1.0 + z_true)
    abs_rel = _np.abs(rel)
    return {
        "n": int(z_true.size),
        "mae": float(_np.mean(_np.abs(dz))),
        "rmse": float(_np.sqrt(_np.mean(dz**2))),
        "nmad": float(1.4826 * _np.median(_np.abs(dz))),
        "sigma_nmad": float(1.4826 * _np.median(_np.abs(rel - _np.median(rel)))),
        "med_abs_dz_over_1pz": float(_np.median(abs_rel)),
        "bias": float(_np.mean(rel)),
        "outlier_frac": float(_np.mean(abs_rel > outlier_thresh)),
    }
