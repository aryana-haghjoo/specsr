"""Console entry point for specsr.

Subcommands are dispatched lazily so that `specsr --help` stays fast and does
not require the optional training or Hub dependencies to be installed.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specsr",
        description="Super-resolve galaxy spectra with a physics-informed model.",
    )
    parser.add_argument("--version", action="version", version=f"specsr {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_build = sub.add_parser(
        "build-dataset",
        help="Build paired/augmented datasets from the raw JADES release tree.",
    )
    p_build.add_argument("--release", default="DR4", help="JADES release (default: DR4)")
    p_build.add_argument("--out", default=None, help="Output .npz path")
    p_build.add_argument(
        "--n-aug", type=int, default=20, help="Augmented realizations per galaxy"
    )
    p_build.add_argument(
        "--no-augment", action="store_true", help="Emit the unaugmented paired set only"
    )
    p_build.add_argument(
        "--limit", type=int, default=None, help="Cap the number of galaxies (smoke tests)"
    )

    p_train = sub.add_parser("train", help="Train a pipeline stage.")
    p_train.add_argument("stage", choices=["sr1", "zhead", "sr2"])
    p_train.add_argument("--config", required=True, help="YAML config path")
    p_train.add_argument("--dataset", default=None, help="Dataset .npz path")
    p_train.add_argument("--out-dir", default=None, help="Where to write checkpoints")
    p_train.add_argument(
        "--source", choices=["lowres", "hires", "sr1", "sr2"], default=None,
        help="zhead only: which spectrum representation the head sees. The "
             "redshift comparison in the paper is this flag, four times.",
    )
    # Upstream stages are named explicitly rather than discovered, so a
    # checkpoint always records the models it was trained on top of.
    p_train.add_argument("--sr1-ckpt", default=None)
    p_train.add_argument("--sr1-config", default=None)
    p_train.add_argument("--zhead-ckpt", default=None)
    p_train.add_argument("--sr2-ckpt", default=None)
    p_train.add_argument("--zhead-bootstrap-ckpt", default=None,
                         help="zhead --source sr2: the frozen head that conditions SR2")
    p_train.add_argument("--init-ckpt", default=None,
                         help="sr1/sr2: warm-start the stage's weights from this "
                              "checkpoint for fine-tuning (architecture must match)")

    p_infer = sub.add_parser("infer", help="Run the full pipeline on spectra.")
    p_infer.add_argument("--dataset", required=True, help="Dataset .npz path")
    p_infer.add_argument("--checkpoints", default=None,
                         help="Directory holding best_superres_model.pth, "
                              "config_logR.yaml, best_zhead.pth and best_sr2.pth")
    p_infer.add_argument("--idx", type=int, nargs="+", help="Row indices")
    p_infer.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    p_infer.add_argument("--save", default=None, help="Write outputs to this .npz")

    p_eval = sub.add_parser("evaluate", help="Reproduce paper evaluations and figures.")
    p_eval.add_argument(
        "analysis",
        choices=["all", "sample", "residuals", "line-flux", "line-snr",
                 "augmentation", "coverage", "redshift"],
    )
    p_eval.add_argument("--dataset", default=None)
    p_eval.add_argument("--cache", default=None,
                        help="Prediction cache from scripts/make_predictions.py")
    p_eval.add_argument("--outdir", default=None)
    # The redshift figure's three arms are separate training runs.
    p_eval.add_argument("--z-lowres", dest="z_lowres", default=None)
    p_eval.add_argument("--z-hires", dest="z_hires", default=None)
    p_eval.add_argument("--z-sr2", dest="z_sr2", default=None)

    return parser


def _parse_sweep_overrides(extras: list[str], parser: argparse.ArgumentParser) -> dict:
    """Turn leftover ``--name=value`` arguments into config overrides.

    A W&B sweep agent appends every searched parameter to the command this way
    (the ``${args}`` macro), and the set of names is whatever that sweep
    searches -- so they cannot be declared on the parser in advance. Before this
    existed, ``specsr train`` rejected them outright and **every trial of every
    sweep failed at argument parsing**, which is why no sweep in this repo had
    ever run against the package CLI.

    Values are parsed as YAML scalars so ``--lr=1e-4`` arrives as a float and
    ``--use_var_clamp=true`` as a bool, rather than as strings that would then
    be silently truthy.
    """
    import yaml

    overrides: dict = {}
    i = 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            parser.error(f"unrecognized arguments: {token}")
        name, sep, raw = token[2:].partition("=")
        if not sep:
            if i + 1 >= len(extras) or extras[i + 1].startswith("--"):
                parser.error(f"override --{name} was given no value")
            raw = extras[i + 1]
            i += 1
        overrides[name.replace("-", "_")] = yaml.safe_load(raw)
        i += 1
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, extras = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Only `train` tolerates undeclared flags, and only because a sweep agent
    # supplies them. Anywhere else an unrecognised flag is a typo and must stay
    # an error rather than being quietly dropped.
    if extras and args.command != "train":
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    args.overrides = _parse_sweep_overrides(extras, parser) if extras else {}

    if args.command == "build-dataset":
        from .data.build import run_build

        return run_build(args)
    if args.command == "train":
        from .training.runner import run_train

        return run_train(args)
    if args.command == "infer":
        from .inference.pipeline import run_infer

        return run_infer(args)
    if args.command == "evaluate":
        from .evaluation.runner import run_evaluate

        return run_evaluate(args)

    parser.error(f"unhandled command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
