#!/usr/bin/env python
"""Draw the per-stage hyperparameter-importance figure (paper Appendix A).

    python scripts/make_sweep_figure.py             # draw from the tracked trial tables
    python scripts/make_sweep_figure.py --refresh   # re-export from W&B, then draw

Writes ``sweep_importance.png``/``.pdf`` into the package output directory
(``SPECSR_OUTPUT_DIR``, default ``./outputs/figures``); pass ``--out`` to write
into a manuscript directory instead. The trial tables it reads live in
``sweeps/`` and are tracked, so the figure is reproducible from a clean checkout
with no W&B credentials.

This replaces the W&B UI screenshot the submitted paper carried. That export was
taken 2026-07-25 from a retired Bayesian sweep predating both the train/test leak
fix and the log-constant-R wavelength grid, so it ranked parameters for a
configuration that no longer exists, and nothing in the repository regenerated
it. Everything here comes from the three post-fix random-search sweeps: SR1
``c9tct37z``, ZHead ``kdtc1kr5``, SR2 ``y2zvwvkf``.

Sign convention: every outcome is oriented so that **lower is better**, so a
*positive* correlation means raising that parameter makes the stage worse.

Why the three panels are not computed the same way
--------------------------------------------------

All three sweeps run hyperband, so most trials are killed at a rung and never
reach a comparable epoch. Two estimates are therefore available, and which one
is trustworthy differs by stage.

*All-trials, composite outcome* (ZHead and SR2). Trials are ordered by how far
hyperband let them run, with the final metric breaking ties inside a survival
tier. This uses every trial and does not select on the outcome, so it is not
range-restricted -- but it inherits whatever hyperband ranked on.

*Survivors only* (SR1). Restricted to trials that passed every rung, ranked on
their final metric. Clean but small, and range-restricted by construction.

**SR1 must use the second, because its sweep metric is not comparable across
trials.** ``avg_val_loss_smooth`` is the value of :func:`sr1_deblend_loss`, and
five of that loss's own terms are searched axes of the same sweep --
``score_w_line`` multiplies the sharpness term outright. Measured over the 40
trials: ``score_w_line`` correlates +0.91 with the loss hyperband ranks on and
+0.01 with the model's actual ``val_mse``, and -0.74 with how long a trial was
allowed to run. Hyperband was therefore killing trials substantially on which
loss weight they happened to draw rather than on how good the model was, and the
composite outcome inherits that. The SR1 panel is consequently ranked on
``val_mse`` -- plain error against the reference, independent of every searched
knob -- over the eight survivors. ZHead's ``val_med_abs_dz_over_1pz`` and SR2's
``val/goal_score`` have no such problem: both are fixed functions of the
predictions whose weights are pinned, not searched.

Significance, and why it is on the figure
------------------------------------------

At 40 trials over 6-12 axes most of these correlations are noise, and a bar
chart of twelve unqualified numbers invites reading structure into it. Each
correlation therefore carries a two-sided permutation p-value -- exact by
enumeration where the point count allows it, Monte Carlo otherwise -- and the
bars are drawn in three weights: solid where the correlation survives a
Bonferroni threshold of 0.05 / (axes searched in that sweep), medium where it
clears an uncorrected 0.05 but not that threshold, faded where it does not clear
either. Only two correlations in the whole study reach the first tier. That is
the honest capacity of 40 trials per stage, and reporting it this way is what
keeps the appendix from turning sampling noise into a physical claim.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from specsr.plotting import PAPER_RC, save_figure  # noqa: E402

ENTITY_PROJECT = "AI-Astro/spectral-superresolution"
SWEEP_DIR = REPO / "sweeps"

#: Enumerate every relabelling below this many points; sample above it.
EXACT_PERMUTATION_MAX_N = 9
MC_PERMUTATIONS = 20000

# Diverging pair: blue = raising the parameter helps, red = hurts. The sign is
# carried by bar direction as well as by hue, so the figure keeps its meaning in
# grayscale and under colour-vision deficiency.
C_BETTER = "#2a78d6"
C_WORSE = "#e34948"
C_INK = "#0b0b0b"
C_RULE = "#52514e"
C_GRID = "#d8d7d2"
SOLID, MEDIUM, FADE = 1.0, 0.62, 0.20


class Stage:
    """One sweep: where its trials live, how they are scored, how they are ranked.

    ``rung1``/``rung2`` are hyperband's early-termination epochs, ``max_iter /
    eta**s`` and ``max_iter / eta``, taken from the sweep YAML rather than
    guessed. ``estimator`` is ``"survivors"`` or ``"composite"``; see the module
    docstring for why SR1 differs from the other two.
    """

    def __init__(self, key, label, sweep_id, metric, metric_label, axes,
                 rung1, rung2, estimator, blurb, extra_metrics=()):
        self.key = key
        self.label = label
        self.sweep_id = sweep_id
        self.metric = metric
        self.metric_label = metric_label
        self.axes = axes
        self.rung1 = rung1
        self.rung2 = rung2
        self.estimator = estimator
        self.blurb = blurb
        self.extra_metrics = list(extra_metrics)

    @property
    def csv_path(self) -> Path:
        return SWEEP_DIR / f"sweep_{self.key}_{self.sweep_id}.csv"

    @property
    def alpha(self) -> float:
        """Bonferroni threshold for this sweep's number of searched axes."""
        return 0.05 / len(self.axes)


STAGES = [
    Stage("sr1", "SR1 — coarse super-resolution", "c9tct37z",
          metric="val_mse", metric_label="val MSE",
          axes=["num_res_blocks", "hidden_dim", "lr", "dropout", "weight_decay",
                "logvar_reg", "gate_min_frac", "gate_temp", "score_w_line",
                "mask_thresh_mad", "mask_smooth_k", "sharp_wd_rate"],
          rung1=13, rung2=40, estimator="survivors",
          blurb="8 full-length trials, ranked on val MSE",
          extra_metrics=["avg_val_loss_smooth"]),
    Stage("zhead", "ZHead — redshift estimation", "kdtc1kr5",
          metric="val_med_abs_dz_over_1pz",
          metric_label=r"med $|\Delta z|/(1{+}z)$",
          axes=["lr", "weight_decay", "dropout", "hidden_dim", "num_blocks",
                "batch_size", "z_var_floor", "pdf_target_sigma_bins",
                "soft_argmax_half"],
          rung1=17, rung2=50, estimator="composite",
          blurb="all trials, survival tier then final metric",
          extra_metrics=["val_outlier_frac"]),
    Stage("sr2", "SR2 — line refinement", "y2zvwvkf",
          metric="val/goal_score", metric_label="goal score",
          axes=["lr", "lam_hp_in", "lam_hp_out", "lam_presence", "lam_flux",
                "delta_cap"],
          rung1=16, rung2=47, estimator="composite",
          blurb="all trials, survival tier then final metric",
          extra_metrics=["val/flux_ratio_bright", "val/z_outlier_rate"]),
]

#: Gloss for the axes whose raw config key does not say what it is. Anything
#: absent is drawn as written, which is also how the appendix names it.
PRETTY = {
    "num_res_blocks": "num_res_blocks (depth)",
    "hidden_dim": "hidden_dim (width)",
    "num_blocks": "num_blocks (depth)",
}


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def refresh(stage: Stage) -> None:
    """Pull every trial of one sweep from W&B into a CSV.

    Each metric is stored twice, at the first hyperband rung and at the trial's
    last epoch. Reading the rung value needs the run's logged history rather than
    its summary, which is what makes this the slow path and the cache worthwhile.
    """
    import wandb

    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY_PROJECT}/{stage.sweep_id}")
    runs = list(sweep.runs)
    keys = [stage.metric, *stage.extra_metrics]
    rows = []
    for i, run in enumerate(runs, 1):
        print(f"  [{i:>2}/{len(runs)}] {run.name} ({run.id})", flush=True)
        at_rung, final = {k: None for k in keys}, {k: None for k in keys}
        last_epoch = -1
        for record in run.scan_history(keys=["epoch", *keys], page_size=1000):
            epoch = record.get("epoch")
            if epoch is None:
                continue
            for k in keys:
                value = record.get(k)
                if value is None or not np.isfinite(value):
                    continue
                if epoch <= stage.rung1:
                    at_rung[k] = value  # last value at or before the rung
                final[k] = value
            last_epoch = max(last_epoch, int(epoch))
        rows.append({
            "name": run.name, "id": run.id, "state": run.state,
            "last_epoch": last_epoch,
            "full_length": int(last_epoch > stage.rung2),
            "runtime_h": round((run.summary.get("_runtime") or 0) / 3600, 3),
            **{f"{k}_at_rung1": at_rung[k] for k in keys},
            **{f"{k}_final": final[k] for k in keys},
            **{a: run.config.get(a) for a in stage.axes},
        })

    SWEEP_DIR.mkdir(exist_ok=True)
    with stage.csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {stage.csv_path.relative_to(REPO)}  ({len(rows)} trials)")


def load(stage: Stage) -> list[dict]:
    if not stage.csv_path.exists():
        raise SystemExit(
            f"missing {stage.csv_path.relative_to(REPO)} -- run with --refresh once "
            f"to pull the {stage.key} trials from W&B")
    with stage.csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for k, v in list(row.items()):
            if v in ("", "None"):
                row[k] = None
            elif k not in ("name", "id", "state"):
                row[k] = float(v)
    return rows


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def spearman(x, y) -> float:
    """Rank correlation, with midranks for ties.

    Ties are the norm here -- ``num_res_blocks`` takes four values over forty
    trials -- so averaging tied ranks rather than breaking them arbitrarily is
    required, not a refinement.
    """
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def permutation_p(x, y, rng: random.Random) -> tuple[float, str]:
    """Two-sided permutation p-value for a rank correlation.

    Exact by enumeration while the point count allows it, Monte Carlo above that.
    An asymptotic t-approximation is not used: at n = 6-8 it is not valid, and
    those are exactly the panels where the p-value carries the most weight.
    """
    n = len(x)
    obs = abs(spearman(x, y))
    if not math.isfinite(obs):
        return float("nan"), "—"
    if n <= EXACT_PERMUTATION_MAX_N:
        total = hit = 0
        for perm in itertools.permutations(range(n)):
            r = spearman(x, [y[i] for i in perm])
            total += 1
            hit += math.isfinite(r) and abs(r) >= obs - 1e-12
        return hit / total, "exact"
    shuffled, hit = list(y), 0
    for _ in range(MC_PERMUTATIONS):
        rng.shuffle(shuffled)
        r = spearman(x, shuffled)
        hit += math.isfinite(r) and abs(r) >= obs - 1e-12
    return (hit + 1) / (MC_PERMUTATIONS + 1), "MC"


def outcome(stage: Stage, rows: list[dict]) -> tuple[list[dict], list[float], int]:
    """The ranked outcome each axis is correlated against, per the stage's estimator.

    Returns the trials used, their outcome values (lower is better), and the
    count of full-length survivors for reporting.
    """
    final_col = f"{stage.metric}_final"
    survivors = sum(1 for r in rows if r["full_length"])
    if stage.estimator == "survivors":
        used = [r for r in rows if r["full_length"] and r[final_col] is not None]
        return used, [r[final_col] for r in used], survivors

    used = [r for r in rows if r[final_col] is not None and r["last_epoch"] is not None]
    # Hyperband's own verdict first -- a trial it let run longer beat the trials
    # it killed -- with the final metric ordering trials inside a survival tier.
    order = sorted(range(len(used)),
                   key=lambda i: (-used[i]["last_epoch"], used[i][final_col]))
    ranked = [0.0] * len(used)
    for position, i in enumerate(order):
        ranked[i] = float(position)
    return used, ranked, survivors


def importances(stage: Stage, rows: list[dict], rng: random.Random) -> list[dict]:
    used, y, survivors = outcome(stage, rows)
    out = []
    for axis in stage.axes:
        keep = [i for i in range(len(used)) if used[i][axis] is not None]
        x = [used[i][axis] for i in keep]
        yy = [y[i] for i in keep]
        rho = spearman(x, yy)
        p, how = permutation_p(x, yy, rng)
        out.append({"axis": axis, "rho": rho, "p": p, "how": how,
                    "n": len(keep), "survivors": survivors})
    out.sort(key=lambda r: -abs(r["rho"]))
    return out


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
#: Row heights, in figure units. One unit is one bar row.
GAP_BETWEEN_STAGES = 0.8

#: Row pitch in inches: 8.3 pt labels on a 10.1 pt pitch, the leading of
#: ordinary small print. Set so the figure and its caption fit under the
#: appendix text on one page.
ROW_IN = 0.14

#: A millimetre of vertical space in row units. Spacing asked for in
#: millimetres is expressed through this instead of by hand-tuning row offsets,
#: which do not carry a physical size on their own.
ROWS_PER_MM = 1 / (ROW_IN * 25.4)

#: Clearance added between a stage's rule and the bold title beneath it. The
#: header row grows by the same amount, so opening that gap moves the title away
#: from the rule without walking it into the first bar of the stage.
TITLE_DROP = 1.0 * ROWS_PER_MM
HEADER_HEIGHT = 1.0 + TITLE_DROP


def _layout(data):
    """Assign a row index to every header and bar in one column, top to bottom."""
    rows, y = [], 0.0
    for i, (stage, trials, imp) in enumerate(data):
        if i:
            y += GAP_BETWEEN_STAGES
        rows.append(("header", y, stage, trials, imp))
        y += HEADER_HEIGHT
        for r in imp:
            rows.append(("bar", y, stage, trials, r))
            y += 1.0
    return rows, y


#: Width of the parameter-name column beside each axis, and the gap between the
#: two columns; the gap also takes the value label of SR1's longest bar.
LABEL_IN, GAP_IN = 1.40, 0.30

#: Vertical margins: above the axes for the helps/hurts labels, below each axis
#: for its ticks and label.
TOP_IN, XAXIS_IN = 0.20, 0.42


def draw(data, out_stem: Path):
    """Two columns on a common scale: SR1 on the left, ZHead and SR2 on the right.

    The correlation is dimensionless, so the stages share one x range and one
    row pitch. Two columns instead of one stack halve the height, so the figure
    sits under the appendix text on one page at its printed size; SR1 has the
    most axes and the other two together roughly balance it. The legend takes
    the space SR1's shorter column leaves free.
    """
    columns = [data[:1], data[1:]]
    layouts = [_layout(col) for col in columns]
    width = 7.2
    span = [total + 0.25 for _, total in layouts]        # ylim runs -0.75 .. total-0.5
    height = TOP_IN + ROW_IN * max(span) + XAXIS_IN + 0.04
    ax_w = (width - 0.10 - 0.08 - GAP_IN - 2 * LABEL_IN) / 2
    left_in = [0.10 + LABEL_IN, 0.10 + 2 * LABEL_IN + ax_w + GAP_IN]

    with plt.rc_context(PAPER_RC):
        # Sized to the width it is printed at, so 8.3 pt on this canvas is 8.3 pt
        # on the page. `\includegraphics[width=\linewidth]` scales whatever it is
        # given, and a figure drawn oversize arrives with unreadable labels.
        fig = plt.figure(figsize=(width, height))
        for (rows, total), x0, sp in zip(layouts, left_in, span, strict=True):
            h = ROW_IN * sp
            ax = fig.add_axes([x0 / width, 1 - (TOP_IN + h) / height,
                               ax_w / width, h / height])
            yaxis = ax.get_yaxis_transform()
            label_x = -LABEL_IN / ax_w                    # label column, in axis units
            bars = [r for r in rows if r[0] == "bar"]

            for kind, y, stage, trials, payload in rows:
                if kind == "header":
                    # GPU-hours are stated in the appendix text; the trial count
                    # stays here because SR1's bars use only its survivors.
                    used = payload[0]["n"]
                    count = (f"{used} of {len(trials)} trials" if used < len(trials)
                             else f"{len(trials)} trials")
                    ax.plot([label_x, 1.0], [y - 0.62] * 2, transform=yaxis,
                            color=C_RULE, lw=0.7, clip_on=False, zorder=6)
                    ax.text(label_x, y - 0.30 + TITLE_DROP, stage.label,
                            transform=yaxis, ha="left", va="center", fontsize=8.8,
                            fontweight="bold", color=C_INK, clip_on=False)
                    # The grid and the zero rule run the full height of the axis,
                    # so this line needs to sit on its own ground to stay legible.
                    ax.text(1.0, y - 0.30 + TITLE_DROP,
                            count, transform=yaxis,
                            ha="right", va="center", fontsize=7.2,
                            color=C_RULE, clip_on=False, zorder=7,
                            bbox=dict(facecolor="white", edgecolor="none",
                                      boxstyle="square,pad=0.25"))
                    continue

                r = payload
                tier = 2 if r["p"] < stage.alpha else (1 if r["p"] < 0.05 else 0)
                ax.barh(y, r["rho"], height=0.66, zorder=3,
                        color=C_WORSE if r["rho"] > 0 else C_BETTER,
                        alpha=(FADE, MEDIUM, SOLID)[tier], edgecolor="none")
                # Selective labels: a number beside a bar that is indistinguishable
                # from zero invites the reader to weigh it, which is the one thing
                # this figure is trying not to encourage.
                if tier or abs(r["rho"]) >= 0.30:
                    pad = 0.035 if r["rho"] >= 0 else -0.035
                    ax.text(r["rho"] + pad, y, f"{r['rho']:+.2f}", va="center",
                            ha="left" if r["rho"] >= 0 else "right",
                            fontsize=7.4, zorder=5, clip_on=False,
                            color=C_INK if tier else C_RULE)

            ax.axvline(0, color=C_INK, lw=0.9, zorder=4)
            ax.set_yticks([y for _, y, _, _, _ in bars])
            ax.set_yticklabels([PRETTY.get(r[4]["axis"], r[4]["axis"]) for r in bars],
                               fontsize=8.3)
            for tick, (_, _, stage, _, r) in zip(ax.get_yticklabels(), bars, strict=True):
                tick.set_color(C_INK if r["p"] < 0.05 else C_RULE)
                if r["p"] < stage.alpha:
                    tick.set_fontweight("bold")
            ax.tick_params(axis="y", length=0, pad=3)

            ax.set_xlim(-1.06, 1.06)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.tick_params(axis="x", labelsize=8.3, length=3, color=C_RULE)
            ax.set_ylim(total - 0.5, -0.75)
            ax.xaxis.grid(True, color=C_GRID, lw=0.55, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(C_RULE)
            ax.set_xlabel("Spearman rank correlation", fontsize=8.8, labelpad=4)
            # The sign is the one thing a reader can get backwards, so it is
            # spelled out on the axis instead of being left to the caption.
            ax.text(-1.06, 1.0, "raising it helps", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=7.8, color=C_BETTER)
            ax.text(1.06, 1.0, "raising it hurts", transform=ax.get_xaxis_transform(),
                    ha="right", va="bottom", fontsize=7.8, color=C_WORSE)

        # Below SR1's axis, in the room its shorter column leaves.
        left_bottom = height - TOP_IN - ROW_IN * span[0] - XAXIS_IN
        fig.legend(handles=[
            Patch(facecolor=C_RULE, alpha=SOLID,
                  label=r"survives correction ($p < 0.05/N_{\rm axes}$)"),
            Patch(facecolor=C_RULE, alpha=MEDIUM, label=r"$p < 0.05$ uncorrected"),
            Patch(facecolor=C_RULE, alpha=FADE, label="not resolved"),
        ], loc="upper left", ncol=1, frameon=False, fontsize=8.0,
            handlelength=1.5, handleheight=0.85, labelspacing=0.35,
            bbox_to_anchor=(0.15 / width, (left_bottom - 0.02) / height))

        # Outline. The axes keep only their bottom spines, so with nothing round
        # the outside the stage rules read as loose fragments once the figure is
        # set into the page.
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor="none", edgecolor=C_RULE, lw=0.8,
                                 zorder=10))
        _check_overlaps(fig)
        save_figure(fig, out_stem.with_suffix(".png"))
        save_figure(fig, out_stem.with_suffix(".pdf"))
        plt.close(fig)


def _check_overlaps(fig) -> None:
    """Fail loudly if any two pieces of text collide or one leaves the frame."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(renderer))
             for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    frame = fig.bbox
    for name, b in boxes:
        if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
            raise RuntimeError(f"text leaves the frame: {name!r}")
    for i, (a, ba) in enumerate(boxes):
        for c, bc in boxes[i + 1:]:
            if ba.overlaps(bc):
                raise RuntimeError(f"text overlaps: {a!r} / {c!r}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true",
                   help="re-pull every trial from W&B before drawing (slow; needs auth)")
    p.add_argument("--seed", type=int, default=20260828,
                   help="seed for the Monte Carlo permutation tests")
    # Pass --out, or set SPECSR_OUTPUT_DIR, to write into a manuscript
    # directory instead.
    p.add_argument("--out", type=Path, default=None,
                   help="output stem; default is <output dir>/figures/sweep_importance")
    args = p.parse_args()

    if args.out is None:
        from specsr.paths import output_dir

        args.out = output_dir("figures") / "sweep_importance"

    rng = random.Random(args.seed)
    data = []
    for stage in STAGES:
        if args.refresh:
            print(f"fetching {stage.key} sweep {stage.sweep_id} ...")
            refresh(stage)
        rows = load(stage)
        data.append((stage, rows, importances(stage, rows, rng)))

    for stage, rows, imp in data:
        print(f"\n{stage.label}  [{stage.sweep_id}]  {len(rows)} trials, "
              f"{imp[0]['survivors']} full length, "
              f"{sum(r['runtime_h'] or 0 for r in rows):.1f} GPU-h")
        print(f"  estimator: {stage.estimator} (n = {imp[0]['n']}), "
              f"outcome: {stage.metric}, Bonferroni alpha = {stage.alpha:.4f}")
        print(f"  {'axis':24s} {'rho':>7s} {'p':>9s}  test")
        for r in imp:
            mark = ("  <- survives correction" if r["p"] < stage.alpha
                    else ("  <- p<0.05 uncorrected" if r["p"] < 0.05 else ""))
            print(f"  {r['axis']:24s} {r['rho']:+7.2f} {r['p']:9.4f}  {r['how']}{mark}")

    draw(data, args.out)
    print(f"\nwrote {args.out.with_suffix('.png')}")
    print(f"wrote {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
