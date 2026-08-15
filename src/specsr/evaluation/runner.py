"""CLI entry point for ``specsr evaluate``.

Delegates to :func:`specsr.evaluation.figures.build_figures`, which
``scripts/make_figures.py`` also calls, so "the paper figures" has one
definition.
"""

from __future__ import annotations

__all__ = ["run_evaluate"]

# CLI analysis name -> figure names understood by `build_figures`.
#
# The right-hand values must match the ``want(...)`` keys in ``figures.py``
# exactly. They are matched by string, and an unrecognised name is not an
# error -- it simply selects nothing -- so a typo here silently produces a
# subset of the figures the user asked for. Two did: "residuals" (the block
# is "residual_maps") and "flux" (the block is "line_flux"), so
# `specsr evaluate residuals` returned only the PSD figure and
# `specsr evaluate line-flux` returned nothing at all.
ANALYSES = {
    "sample": ["spectrum"],
    "residuals": ["residual_maps", "psd"],
    "line-flux": ["line_flux"],
    "line-snr": ["snr"],
    "augmentation": ["augmentation"],
    "coverage": ["coverage"],
    "redshift": ["redshift"],
}


def run_evaluate(args) -> int:
    """Dispatch ``specsr evaluate <analysis>``. Returns a process exit code."""
    import matplotlib

    matplotlib.use("Agg")

    from .figures import build_figures, build_parser

    fig_args = build_parser().parse_args([])
    if getattr(args, "dataset", None):
        fig_args.dataset = args.dataset
    if getattr(args, "cache", None):
        fig_args.cache = args.cache
    # Figures land in one place by default rather than wherever the command
    # happened to be run from. `--outdir` still wins when given.
    if getattr(args, "outdir", None):
        fig_args.out = args.outdir
    else:
        from ..paths import output_dir

        fig_args.out = str(output_dir("figures"))
    for arm in ("z_lowres", "z_hires", "z_sr2"):
        if getattr(args, arm, None):
            setattr(fig_args, arm, getattr(args, arm))

    if args.analysis == "all":
        fig_args.only = None
    else:
        if args.analysis not in ANALYSES:
            raise SystemExit(
                f"unknown analysis {args.analysis!r}; choose from "
                f"{', '.join(sorted(ANALYSES))} or 'all'"
            )
        fig_args.only = ANALYSES[args.analysis]

    built = build_figures(fig_args)
    if not built:
        raise SystemExit(
            f"nothing was built for {args.analysis!r}. The prediction cache is "
            "probably missing the keys this figure needs -- rerun "
            "scripts/make_predictions.py."
        )
    return 0
