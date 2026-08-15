"""Validation figures logged to W&B during training.

Every run logs pictures, not only scalars. A scalar says a run got worse; only
the spectrum says it started manufacturing lines that are not there, and only
the redshift scatter distinguishes ordinary dispersion about the diagonal from
a second locus off it caused by a misidentified line. Both failures have been
missed here before by watching loss curves alone.

Keys are prefixed ``val/0_`` so the panels sort to the top of the W&B run page,
ahead of the scalar charts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "spectrum_panel",
    "redshift_panel",
    "SPECTRUM_KEY",
    "REDSHIFT_KEY",
    "FINAL_SPECTRUM_KEY",
    "FINAL_REDSHIFT_KEY",
]

SPECTRUM_KEY = "val/0_spectra_LR_SR_HR"
REDSHIFT_KEY = "val/0_z_pred_vs_true"

# The per-epoch panels above show whatever epoch happened to land on
# ``plot_every``. These two are logged once, at the end of a run, from the
# *selected* checkpoint -- which is the model that gets used, and on an
# early-stopped run need not be an epoch that was ever plotted.
FINAL_SPECTRUM_KEY = "val/00_final_spectra_LR_SR_HR"
FINAL_REDSHIFT_KEY = "val/00_final_z_pred_vs_true"


def _as_np(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def spectrum_panel(
    wave_um,
    lr,
    sr,
    hr,
    *,
    z_true=None,
    z_pred=None,
    z_sigma=None,
    title: str = "",
    key: str | None = None,
) -> dict[str, Any]:
    """``{key: wandb.Image}`` showing one validation spectrum: LR, SR, HR.

    ``key`` overrides :data:`SPECTRUM_KEY`, so the end-of-run panel can be
    logged under :data:`FINAL_SPECTRUM_KEY` without overwriting the per-epoch
    series.

    Returns an empty dict on any failure. A plotting bug must never end a
    training run -- this is a diagnostic, not the science.
    """
    out: dict[str, Any] = {}
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import wandb

        wave = _as_np(wave_um).ravel()
        lr_a, sr_a, hr_a = (_as_np(v).ravel() for v in (lr, sr, hr))

        fig, ax = plt.subplots(figsize=(11.0, 4.2), constrained_layout=True)
        ax.plot(wave, hr_a, lw=1.1, alpha=0.85, label="HR target")
        ax.plot(wave, lr_a, lw=1.0, alpha=0.70, label="LR input")
        ax.plot(wave, sr_a, lw=1.1, alpha=0.95, label="SR prediction")
        ax.set_xlabel(r"Observed wavelength [$\mu$m]")
        ax.set_ylabel(r"$F_\lambda$ (normalized)")

        bits = [title] if title else []
        if z_true is not None:
            bits.append(f"z_true={float(_as_np(z_true)):.4f}")
        if z_pred is not None:
            s = f"z_pred={float(_as_np(z_pred)):.4f}"
            if z_sigma is not None:
                s += f" ± {float(_as_np(z_sigma)):.4f}"
            bits.append(s)
        if bits:
            ax.set_title("   ".join(bits), fontsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.12)

        out[key or SPECTRUM_KEY] = wandb.Image(fig)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not kill a run
        print(f"  (spectrum plot skipped: {exc})", flush=True)
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return out


def redshift_panel(
    z_true, z_pred, *, title: str = "z-head", key: str | None = None
) -> dict[str, Any]:
    """``{key: wandb.Image}`` of predicted vs true redshift with the metrics box.

    Uses the same :func:`specsr.plotting.plot_redshift_panel` as the paper
    figure, so the outlier rate watched during training and the one printed in
    the paper are the same statistic.

    ``key`` overrides :data:`REDSHIFT_KEY`, for the end-of-run panel logged
    under :data:`FINAL_REDSHIFT_KEY`.
    """
    out: dict[str, Any] = {}
    fig = None
    try:
        import wandb

        from .plotting import plot_redshift_panel

        fig, _ = plot_redshift_panel(_as_np(z_true), _as_np(z_pred), title=title)
        out[key or REDSHIFT_KEY] = wandb.Image(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  (redshift plot skipped: {exc})", flush=True)
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return out
