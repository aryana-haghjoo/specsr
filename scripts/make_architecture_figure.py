#!/usr/bin/env python
"""Draw the three-stage pipeline schematic (paper Figure 3).

    python scripts/make_architecture_figure.py

Writes ``fig_architecture.pdf`` and ``.png`` into the package output
directory (``SPECSR_OUTPUT_DIR``, default ``./outputs/figures``).

Why this is a script and not TikZ
---------------------------------
The three inset panels are **real data**, not sketches: the low-resolution input,
the super-resolved output and the redshift PDF are read from the prediction cache
and recomputed with the released chain, for one galaxy selected by the same
``rank_doublet_examples`` criterion the other example figures use. A diagram whose
numbers and curves come from the model cannot drift away from the model, which is
exactly what the previous hand-maintained TikZ version did -- it still advertised
teacher forcing, a Gaussian redshift head and a 2-channel SR1 input, none of which
survive in the code.

Every architectural number printed on the figure is derived from the loaded
checkpoints at draw time (see :func:`architecture_facts`), so a retrain that
changes a width or a depth changes the figure rather than silently invalidating
it.
"""
from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from specsr import metrics  # noqa: E402
from specsr.evaluation import load_pipeline  # noqa: E402
from specsr.plotting import PAPER_RC  # noqa: E402

# The released v4 chain. Same three checkpoints the prediction cache was built
# from; `provenance` in the cache is asserted against them below so the figure
# cannot illustrate a different model from the one the paper measures.
SR1_CKPT = "runs/finetune_20260730_003724/sr1/best_superres_model.pth"
SR1_CONFIG = "checkpoints/checkpoints_baseline_20260726/config_logR.yaml"
ZHEAD_CKPT = "runs/zhead_pdf_8020/sr1/best_zhead_sr1.pth"
SR2_CKPT = "runs/sr2_maskfix_20260803_170711/best_sr2.pth"
CACHE = "cache/predictions_val.npz"

# `zhead_unfreeze_last_n` from the SR2 run config. Not readable from the ZHead
# checkpoint -- it is a property of how SR2 trained, not of the head -- so it is
# named here and cross-checked against the SR2 checkpoint's stored config.
SR2_UNFREEZE_LAST_N = 2
SR2_ZHEAD_LR_MULT = 0.1

# Readable names for the ZHead leaf modules SR2 reopens. Anything not listed
# falls through as its raw attribute name, which is deliberate: an unfamiliar
# identifier appearing on the figure is the signal that the unfrozen set changed.
_LEAF_LABELS = {
    "attn_score": "attention pooling",
    "z_logits": "the output layer",
}

# ---------------------------------------------------------------- palette ---
PANEL_SR = "#E8EDF4"
PANEL_Z = "#F2ECE0"
SLATE = "#38455A"
SLATE_TOP = "#4C5B72"
SLATE_SIDE = "#2A3446"
EDGE = "#2B3547"
GREEN = "#2E9E4F"
GREEN_EDGE = "#1F7238"
PURPLE = "#7B3FA0"
INK = "#1D2530"
MUTED = "#5B6675"
ARROW = "#3A4A5A"
LR_COL = "#CE3F3A"
SR_COL = "#E4842C"
FRAME = "#8A93A0"

# Canvas. Isotropic: the figure is exactly W_UNITS x H_UNITS in data
# coordinates, so a rounded corner drawn as a circle stays a circle.
W_UNITS, H_UNITS = 100.0, 47.0
FIG_W = 7.2


# ------------------------------------------------------------- primitives ---
def rounded(ax, x0, y0, x1, y1, *, r=1.2, fc="none", ec="none", lw=1.0,
            ls="solid", z=1, alpha=1.0):
    """Rounded rectangle spanning the corners given, in data units."""
    p = FancyBboxPatch(
        (x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=z, alpha=alpha, mutation_aspect=1.0,
    )
    ax.add_patch(p)
    return p


def arrow(ax, xy_from, xy_to, *, color=ARROW, lw=1.15, z=6, rad=0.0,
          head=6.0, ls="solid"):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=f"-|>,head_length={head},head_width={head * 0.55}",
        connectionstyle=f"arc3,rad={rad}", mutation_scale=1.0,
        color=color, linewidth=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0, joinstyle="miter",
    ))


def slab(ax, x0, y0, w, h, *, depth=1.1, face=SLATE, top=SLATE_TOP,
         side=SLATE_SIDE, z=5, lw=0.0):
    """One extruded block: front face plus a lit top and a shaded right side."""
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=face, edgecolor=face,
                           linewidth=lw, zorder=z))
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + depth, y0 + h + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=top, edgecolor=top,
                         linewidth=lw, zorder=z))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + depth, y0 + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=side, edgecolor=side,
                         linewidth=lw, zorder=z))


def text(ax, x, y, s, *, size=6.2, color=INK, weight="normal", style="normal",
         ha="center", va="center", z=10, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   fontstyle=style, ha=ha, va=va, zorder=z, **kw)


def inset(ax, x0, y0, x1, y1, *, z=8):
    """A framed sub-axes placed in data coordinates of the schematic axes."""
    fig = ax.figure
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    (fx0, fy0), (fx1, fy1) = fig.transFigure.inverted().transform([p0, p1])
    sub = fig.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0), zorder=z)
    sub.set_xticks([])
    sub.set_yticks([])
    for spine in sub.spines.values():
        spine.set_edgecolor(FRAME)
        spine.set_linewidth(0.6)
    sub.set_facecolor("white")
    return sub


# ------------------------------------------------------------------- data ---
def architecture_facts(pipe) -> dict:
    """Read every number the figure prints off the loaded models.

    Nothing here is hand-copied from a config file. ``configs/sr2.yaml`` is a
    stale W&B export carrying pre-log-grid values, and the SR1 config shipped
    beside the baseline checkpoint is a different run's export again; the
    weights are the only description of the network that cannot be out of date.
    """
    sr1, zhead, sr2 = pipe.sr1, pipe.zhead, pipe.sr2
    wave = np.asarray(pipe.wave, dtype=np.float64)
    dlnl = float(np.mean(np.diff(np.log(wave))))

    n_sr1 = sum(p.numel() for p in sr1.parameters())
    n_z = sum(p.numel() for p in zhead.parameters())
    n_sr2 = sum(p.numel() for p in sr2.parameters())

    zgrid = zhead.z_grid_n.detach().cpu().numpy() * pipe.z_std + pipe.z_mean
    # Dilation of block i is growth**i; kernel 7 throughout.
    dils = [int(m.dilation[0]) for m in zhead.flux_net if isinstance(m, torch.nn.Conv1d)]
    rf = 1 + sum((7 - 1) * d for d in dils)

    # Which parts of ZHead SR2 reopens. `unfreeze_last_n` counts *parameterised
    # leaf modules*, not blocks -- at n=2 that is the attention-pooling score and
    # the 1,024-way output layer, ~19% of the head, not "the last two conv
    # blocks". Derived here rather than described from memory, because the
    # obvious reading of the config key is wrong and both the manuscript and the
    # model card had it wrong from that reading.
    leaves = [n for n, m in zhead.named_modules() if n and list(m.parameters(recurse=False))]
    n_unfreeze = int(SR2_UNFREEZE_LAST_N)
    unfrozen = leaves[-n_unfreeze:] if n_unfreeze else []
    unfrozen_par = sum(
        p.numel() for n, m in zhead.named_modules() if n in unfrozen
        for p in m.parameters(recurse=False)
    )

    return {
        "L": int(wave.size),
        "lam_lo": float(wave[0]),
        "lam_hi": float(wave[-1]),
        "R_pix": 1.0 / dlnl,
        "sr1_in": int(sr1.initial[0].in_channels),
        "sr1_dim": int(sr1.initial[0].out_channels),
        "sr1_blocks": len(sr1.resblocks),
        "z_in": int(zhead.flux_net[0].in_channels),
        "z_dim": int(zhead.flux_net[0].out_channels),
        "z_blocks": len(dils),
        "z_rf": rf,
        "z_bins": int(zhead.n_z_bins),
        "z_lo": float(zgrid.min()),
        "z_hi": float(zgrid.max()),
        "z_half": int(zhead.soft_argmax_half),
        "z_unfrozen": " and ".join(_LEAF_LABELS.get(n, n) for n in unfrozen),
        "z_unfrozen_par": unfrozen_par,
        "z_unfrozen_frac": unfrozen_par / max(n_z, 1),
        "sr2_in": int(sr2.line_encoder[0].in_channels),
        "K": int(sr2.K),
        "line_dim": int(sr2.line_embed.embedding_dim),
        "attn_layers": len(sr2.line_attn.layers),
        "attn_heads": int(sr2.line_attn.layers[0].self_attn.num_heads),
        "window": int(sr2.W),
        "window_kms": 2 * sr2.window_half * dlnl * 299792.458,
        "cnn_dim": int(sr2.cnn_in[0].out_channels),
        "cnn_blocks": len(sr2.cnn_blocks),
        "delta_cap": float(pipe.cfg.get("delta_cap", 30.0)),
        "n_sr1": n_sr1,
        "n_z": n_z,
        "n_sr2": n_sr2,
        "n_total": n_sr1 + n_z + n_sr2,
    }


def example_data(pipe, cache_path: Path, index: int | None) -> dict:
    """One held-out galaxy carried through the chain, plus its redshift PDF."""
    with np.load(cache_path, allow_pickle=True) as d:
        # `np.savez` stringifies the provenance dict, so it comes back as a repr.
        prov = d["provenance"].item()
        if isinstance(prov, str):
            prov = ast.literal_eval(prov)
        for key, want in (("sr1_ckpt", SR1_CKPT), ("zhead_ckpt", ZHEAD_CKPT),
                          ("sr2_ckpt", SR2_CKPT)):
            got = str(prov.get(key, ""))
            if not got.endswith(str(want)):
                raise SystemExit(
                    f"cache was built with {key}={got!r}, not {want!r}. The figure would "
                    "illustrate a different model from the one the paper measures."
                )
        wave = np.asarray(d["wave"])
        lr, sr2, hr = d["flux_low"], d["sr2"], d["flux_high"]
        z_true, z_pred, z_sig = d["z_true"], d["z_pred"], d["z_sigma"]
        row = d["row_index"]

    if index is None:
        ranked = metrics.rank_doublet_examples(wave, lr, sr2, hr, z_true, z_pred=z_pred)
        if len(ranked) == 0:
            raise SystemExit("no spectra with the [O III] doublet in range")
        index = int(ranked[0])

    x = np.asarray(lr[index], dtype=np.float32)
    mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
    xn = np.nan_to_num((x - mu) / (sd if sd > 0 else 1.0))[None, None, :]
    with torch.no_grad():
        t = torch.tensor(xn, device=pipe.device)
        sr1_mean, sr1_logvar = pipe.sr1(t)
        z_in = torch.cat([sr1_mean, 0.5 * sr1_logvar], dim=1) if pipe.use_sigma else sr1_mean
        pdf = torch.softmax(pipe.zhead.logits(z_in), dim=-1)[0].cpu().numpy()
        zgrid = pipe.zhead.z_grid_n.cpu().numpy() * pipe.z_std + pipe.z_mean

    return {
        "index": index,
        "row": int(row[index]),
        "wave": wave,
        "lr": np.asarray(lr[index]),
        "sr": np.asarray(sr2[index]),
        "z_true": float(z_true[index]),
        "z_pred": float(z_pred[index]),
        "z_sigma": float(z_sig[index]),
        "zgrid": zgrid,
        "pdf": pdf,
    }


# ------------------------------------------------------------------ panels ---
def diagnostic_window(wave, z):
    """The observed-frame span from [O II] to H-alpha at redshift ``z``.

    Both spectrum insets are drawn over this window rather than the whole
    1-5.3 um grid. Over the full grid a single bright line sets the vertical
    scale and flattens everything else, so the super-resolved panel reads as an
    empty box with one spike -- the opposite of what the figure is claiming.
    The strong-line region is where the LR-to-SR difference actually lives.
    Falls back to the full grid if the window would fall off the detector.
    """
    lo = 0.3727 * (1.0 + z) * 0.94
    hi = 0.6563 * (1.0 + z) * 1.06
    lo, hi = max(lo, wave[0]), min(hi, wave[-1])
    if hi - lo < 0.25:
        return float(wave[0]), float(wave[-1])
    return float(lo), float(hi)


def draw_spectrum(sub, wave, flux, color, window):
    """A spectrum scaled to its own peak, so LR and SR are shape-comparable.

    The absolute flux scales differ by orders of magnitude between the panels
    and neither is the point here; what the figure has to show is that the
    blended humps on the left become resolved lines on the right.
    """
    f = np.asarray(flux, dtype=float)
    m = (wave >= window[0]) & (wave <= window[1])
    w, f = wave[m], f[m]
    peak = np.nanmax(np.abs(f))
    f = f / (peak if peak > 0 else 1.0)
    sub.plot(w, f, lw=0.45, color=color, solid_joinstyle="miter")
    sub.set_xlim(w[0], w[-1])
    sub.set_ylim(-0.12, 1.16)
    sub.text(0.035, 0.93, f"{w[0]:.2f}–{w[-1]:.2f} µm", transform=sub.transAxes,
             fontsize=4.6, color=MUTED, va="top", ha="left")


def draw_pdf(sub, zgrid, pdf, z_true, z_pred):
    sub.fill_between(zgrid, 0, pdf, color=GREEN, alpha=0.85, lw=0)
    sub.plot(zgrid, pdf, lw=0.7, color=GREEN_EDGE)
    sub.set_xlim(zgrid.min(), zgrid.max())
    sub.set_ylim(0, pdf.max() * 1.55)
    sub.axvline(z_true, color=MUTED, lw=0.5, ls=(0, (2.2, 1.6)))
    sub.annotate(f"$\\hat{{z}}={z_pred:.2f}$",
                 xy=(z_pred, pdf.max()), xytext=(9, 1),
                 textcoords="offset points", fontsize=5.2, color=GREEN_EDGE,
                 va="center", ha="left")


# ------------------------------------------------------------------- draw ---
def build(facts: dict, ex: dict, out_pdf: Path, out_png: Path) -> None:
    plt.rcParams.update({
        **PAPER_RC,
        # Type 42 embeds a TrueType subset rather than a Type 3 bitmap, which is
        # what ApJ's production system and Overleaf both want.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    fig = plt.figure(figsize=(FIG_W, FIG_W * H_UNITS / W_UNITS))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_UNITS)
    ax.set_ylim(0, H_UNITS)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    f = facts

    # ============================ panel backgrounds ==========================
    rounded(ax, 1.5, 23.4, 98.5, 45.9, r=1.4, fc=PANEL_SR, z=0)
    rounded(ax, 1.5, 1.0, 98.5, 21.8, r=1.4, fc=PANEL_Z, z=0)

    text(ax, 4.4, 44.1, "Spectral super-resolution", size=8.6, weight="bold",
         color=INK, ha="left")
    text(ax, 96.6, 44.1, "physics-informed losses", size=5.6, style="italic",
         color=MUTED, ha="right")
    text(ax, 4.4, 20.0, "Redshift branch", size=8.6, weight="bold", color=INK,
         ha="left")
    text(ax, 96.6, 20.0, "tells SR2 where the lines belong", size=5.6,
         style="italic", color=MUTED, ha="right")

    # ============================ stage 1: SR1 ===============================
    win = diagnostic_window(ex["wave"], ex["z_true"])
    lr_ax = inset(ax, 4.2, 32.0, 18.2, 40.9)
    draw_spectrum(lr_ax, ex["wave"], ex["lr"], LR_COL, win)
    text(ax, 10.2, 30.6, "prism spectrum, $R\\approx100$", size=6.2, weight="bold")
    text(ax, 10.2, 29.1, "one flux channel, resampled", size=5.0, color=MUTED)
    text(ax, 10.2, 27.9,
         f"onto the model grid: {f['L']:,} px,", size=5.0, color=MUTED)
    text(ax, 10.2, 26.7,
         f"$R\\approx${f['R_pix']:,.0f}, {f['lam_lo']:.1f}–{f['lam_hi']:.1f} µm",
         size=5.0, color=MUTED)

    arrow(ax, (18.9, 36.2), (21.6, 36.2))

    for i in range(4):
        slab(ax, 22.2 + i * 2.45, 32.0, 1.7, 8.0)
    text(ax, 28.0, 30.6, "SR1 — conservative CNN", size=6.2, weight="bold")
    text(ax, 28.0, 29.1,
         f"{f['sr1_blocks']} residual blocks × {f['sr1_dim']} ch",
         size=5.0, color=MUTED)
    text(ax, 28.0, 27.9, "flux + log-variance out", size=5.0, color=MUTED)
    text(ax, 28.0, 26.7, f"{f['n_sr1']:,} parameters", size=5.0, color=MUTED)

    # ---- SR1 output: the 5-channel stack SR2 consumes ----------------------
    arrow(ax, (33.6, 34.8), (49.4, 34.8))
    text(ax, 40.9, 39.0, "SR2 input stack, 5 channels", size=5.4, weight="bold",
         color="#46515F")
    text(ax, 40.9, 37.6,
         "$x_{\\rm LR}\\,\\cdot\\,x_{\\rm SR1}\\,\\cdot\\,\\sigma_{\\rm SR1}$",
         size=5.2, color=MUTED)
    text(ax, 40.9, 36.4, "$m(\\lambda;\\hat{z})\\,\\cdot\\,\\hat{z}$",
         size=5.2, color=MUTED)

    # ============================ stage 3: SR2 ===============================
    # Offset cards behind the main box stand for the K line tokens.
    for i, dx in enumerate((1.5, 1.0, 0.5)):
        rounded(ax, 50.0 + dx, 28.4 + dx, 78.0 + dx, 40.6 + dx, r=0.9,
                fc="#FFFFFF", ec="#C3CBD8", lw=0.6, z=2 + i)
    rounded(ax, 50.0, 28.4, 78.0, 40.6, r=0.9, fc=SLATE, ec=EDGE, lw=0.8, z=5)

    text(ax, 64.0, 38.0, "SR2 — attention refiner", size=7.8, weight="bold",
         color="white", z=11)
    text(ax, 64.0, 35.7,
         f"{f['K']} emission-line tokens, self-attention",
         size=5.6, color="#DDE4EF", z=11)
    text(ax, 64.0, 34.3,
         f"{f['attn_layers']} layers × {f['attn_heads']} heads  ·  "
         f"{f['window']}-px window per line",
         size=5.6, color="#DDE4EF", z=11)
    text(ax, 64.0, 32.9,
         "each token emits amplitude, width, offset",
         size=5.6, color="#DDE4EF", z=11)
    text(ax, 64.0, 31.5, "and a supervised presence gate",
         size=5.6, color="#DDE4EF", z=11)
    text(ax, 64.0, 29.7,
         f"+ CNN continuum branch, {f['cnn_blocks']} blocks × {f['cnn_dim']} ch",
         size=5.6, color="#AEBACB", z=11)

    text(ax, 62.5, 26.9,
         "$\\Delta x=\\Delta x_{\\rm line}+\\Delta x_{\\rm CNN}$, "
         f"$\\tanh$-capped at $\\pm${f['delta_cap']:.0f}",
         size=5.0, color=MUTED)
    text(ax, 62.5, 25.7,
         f"and added to the SR1 output  ·  {f['n_sr2']:,} parameters",
         size=5.0, color=MUTED)

    arrow(ax, (79.9, 36.2), (82.4, 36.2))

    sr_ax = inset(ax, 83.0, 32.0, 96.4, 40.9)
    draw_spectrum(sr_ax, ex["wave"], ex["sr"], SR_COL, win)
    text(ax, 89.7, 30.6, "super-resolved spectrum", size=6.2, weight="bold")
    text(ax, 89.7, 29.1, "same grid, matched to the", size=5.0, color=MUTED)
    text(ax, 89.7, 27.9, "$R\\approx1000$ grating reference,", size=5.0, color=MUTED)
    text(ax, 89.7, 26.7, "with a per-pixel $\\sigma$", size=5.0, color=MUTED)

    # ============================ stage 2: ZHead =============================
    ax.plot([37.2], [34.8], marker="o", ms=2.2, color=ARROW, zorder=8)
    arrow(ax, (37.2, 34.8), (36.6, 15.1))
    text(ax, 38.3, 25.3, "coarse spectrum", size=5.4, style="italic", color=MUTED,
         ha="left")
    text(ax, 38.3, 24.0, "and its $\\sigma$", size=5.4, style="italic", color=MUTED,
         ha="left")

    rounded(ax, 3.2, 5.8, 22.8, 14.7, r=0.9, fc="#E9E1D2", ec="none", z=3)
    rounded(ax, 3.2, 5.8, 22.8, 14.7, r=0.9, fc="none", ec="#A79C87", lw=0.7,
            ls=(0, (3, 2)), z=4)
    text(ax, 13.0, 13.4, "During SR2 training", size=5.6, weight="bold", color="#4A4335")
    # Wrapped rather than hand-broken: the unfrozen-layer names are derived, so a
    # future change to what SR2 reopens must not silently overflow the box.
    note = (f"SR1 is frozen. ZHead is frozen except {f['z_unfrozen']} "
            f"({f['z_unfrozen_frac'] * 100:.0f}% of its weights), which train at "
            f"{SR2_ZHEAD_LR_MULT:g}× the rate. No teacher forcing anywhere.")
    for i, line in enumerate(textwrap.wrap(note, width=34)):
        text(ax, 13.0, 11.9 - 1.25 * i, line, size=5.2, color="#4A4335")

    rounded(ax, 27.2, 6.0, 46.0, 14.7, r=0.9, fc=GREEN, ec=GREEN_EDGE, lw=0.8, z=5)
    text(ax, 36.6, 12.2, "ZHead — $P(z)$", size=7.8, weight="bold", color="white", z=11)
    text(ax, 36.6, 10.0, f"{f['z_blocks']} dilated conv blocks × {f['z_dim']} ch",
         size=5.6, color="#E6F4EA", z=11)
    text(ax, 36.6, 8.6, f"{f['z_rf']}-px receptive field", size=5.6, color="#E6F4EA", z=11)
    text(ax, 36.6, 7.2, "attention pooling over $\\lambda$", size=5.6, color="#E6F4EA", z=11)
    text(ax, 36.6, 4.5, f"{f['n_z']:,} parameters", size=5.0, color=MUTED)

    arrow(ax, (46.8, 10.3), (49.4, 10.3))

    pz_ax = inset(ax, 50.0, 6.2, 62.4, 14.4)
    draw_pdf(pz_ax, ex["zgrid"], ex["pdf"], ex["z_true"], ex["z_pred"])
    text(ax, 56.2, 4.5,
         f"$P(z)$ over {f['z_bins']:,} bins, "
         f"$z\\in[{f['z_lo']:.0f},\\,{f['z_hi']:.0f}]$",
         size=5.8, weight="bold")
    text(ax, 56.2, 3.1,
         f"$\\hat{{z}}$ from a soft-argmax over ±{f['z_half']} bins  ·  "
         "dashed: catalogue $z$",
         size=5.0, color=MUTED)

    arrow(ax, (63.1, 10.3), (72.4, 10.3))
    text(ax, 67.7, 12.8, "$\\hat{z}$ = soft-argmax", size=5.0, color=MUTED)
    text(ax, 67.7, 11.6, "$\\sigma_z$ = PDF width", size=5.0, color=MUTED)

    rounded(ax, 73.0, 6.0, 96.6, 14.7, r=0.9, fc="#FFFFFF", ec=PURPLE, lw=1.0,
            ls=(0, (3.4, 2.2)), z=5)
    text(ax, 84.8, 12.4, "line mask  $m(\\lambda;\\hat{z},\\sigma_z)$", size=7.0,
         weight="bold", color=PURPLE, z=11)
    text(ax, 84.8, 10.2, f"{f['K']} rest-frame features placed at $\\hat{{z}}$;",
         size=5.6, color=PURPLE, z=11)
    text(ax, 84.8, 8.8, "Gaussian widths grow with $\\sigma_z$,", size=5.6,
         color=PURPLE, z=11)
    text(ax, 84.8, 7.4, "capped at 0.05 µm", size=5.6, color=PURPLE, z=11)

    # The mask is a conditioning input to SR2, not a forward step in the chain:
    # drawn in the accent colour so the loop back into stage 3 is legible.
    arrow(ax, (81.0, 15.4), (76.0, 28.2), color=PURPLE, lw=1.2, rad=-0.16, z=7)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(REPO / CACHE))
    ap.add_argument("--index", type=int, default=None,
                    help="cache row to illustrate; default is the top-ranked "
                         "[O III] doublet example, as in Figure 4")
    # Pass --out, or set SPECSR_OUTPUT_DIR, to write into a manuscript
    # directory instead.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pipe = load_pipeline(
        sr2_ckpt=str(REPO / SR2_CKPT), sr1_ckpt=str(REPO / SR1_CKPT),
        sr1_config=str(REPO / SR1_CONFIG), zhead_ckpt=str(REPO / ZHEAD_CKPT),
    )
    # The two SR2-side constants are named in this file, so check them against
    # what the SR2 checkpoint records rather than trusting the copy.
    sr2_cfg = torch.load(str(REPO / SR2_CKPT), map_location="cpu",
                         weights_only=False).get("config", {})
    for key, want in (("zhead_unfreeze_last_n", SR2_UNFREEZE_LAST_N),
                      ("zhead_lr_mult", SR2_ZHEAD_LR_MULT)):
        got = sr2_cfg.get(key)
        if got is not None and float(got) != float(want):
            raise SystemExit(
                f"SR2 was trained with {key}={got}, but this script says {want}. "
                "Update the constant; the figure would misstate the freeze."
            )

    facts = architecture_facts(pipe)
    ex = example_data(pipe, Path(args.cache), args.index)

    if args.out:
        out = Path(args.out)
    else:
        from specsr.paths import output_dir

        out = output_dir("figures")
    out.mkdir(parents=True, exist_ok=True)
    build(facts, ex, out / "fig_architecture.pdf", out / "fig_architecture.png")

    print(f"\nexample: cache row {ex['index']} (dataset row {ex['row']}), "
          f"z_true={ex['z_true']:.4f}  z_pred={ex['z_pred']:.4f} "
          f"± {ex['z_sigma']:.4f}")
    print("architecture facts read from the checkpoints:")
    for k, v in facts.items():
        if isinstance(v, float):
            shown = f"{v:,.4g}"
        elif isinstance(v, int):
            shown = f"{v:,}"
        else:
            shown = str(v)
        print(f"  {k:16s} {shown}")
    print(f"\nwrote {out / 'fig_architecture.pdf'} and {out / 'fig_architecture.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
