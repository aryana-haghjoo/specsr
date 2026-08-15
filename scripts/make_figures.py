#!/usr/bin/env python
"""Regenerate the paper figures from a prediction cache.

    python scripts/make_predictions.py --sr2-ckpt <ckpt>
    python scripts/make_figures.py

Figures land in the package output directory (``SPECSR_OUTPUT_DIR``,
default ``./outputs/figures``). Equivalent to ``specsr evaluate --all``; both call
:func:`specsr.evaluation.figures.build_figures`, so the script and the CLI
cannot drift apart.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display on the training box

from specsr.evaluation.figures import build_figures, build_parser  # noqa: E402


def main() -> int:
    build_figures(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
