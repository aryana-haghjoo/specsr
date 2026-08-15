"""CLI entry point for ``specsr train``.

Thin: parse a YAML config, dispatch to the stage. All logic lives in
:mod:`specsr.training.sr1`, :mod:`~specsr.training.zhead` and
:mod:`~specsr.training.sr2`.

``specsr train`` advertised this module long before it existed -- the subcommand
raised ``ModuleNotFoundError`` for anyone who tried it, along with ``infer`` and
``evaluate``. The CLI was written as a specification of where code should live;
this is the code arriving where it was already claimed to be.
"""

from __future__ import annotations

from pathlib import Path

from ..config import merge_overrides
from ..models.loaders import load_yaml_config

__all__ = ["run_train"]


def _apply_overrides(cfg: dict, overrides: dict, stage: str) -> dict:
    """Merge sweep-supplied overrides over the config file, rejecting typos.

    A W&B sweep agent passes each searched parameter as ``--name=value``, and
    these must win over ``--config`` -- otherwise every trial trains the config
    file's values and the sweep varies nothing at all, while W&B still records
    the assigned parameters and reports a "best" configuration chosen at random.

    Names are checked against the stage's ``DEFAULTS`` because a name the stage
    does not read is worse than useless: the optimiser varies it, logs it, and
    attributes its absence of effect to noise, so the search burns its budget on
    an axis that does nothing while appearing to explore it.
    an earlier SR2 sweep config shipped eight such axes out of eleven.
    """
    if not overrides:
        return cfg

    import importlib

    defaults = importlib.import_module(f"specsr.training.{stage}").DEFAULTS
    allowed = set(defaults) | {"dataset_npz", "wandb_name", "wandb_project", "source"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise SystemExit(
            f"specsr train {stage}: {unknown} are not read by this stage, so setting "
            "them would have no effect on the model. If these come from a sweep, the "
            "sweep is searching dead axes -- fix the sweep config rather than this "
            "check. Valid names are the keys of specsr.training."
            f"{stage}.DEFAULTS."
        )
    return merge_overrides(cfg, **overrides)


def run_train(args) -> int:
    """Dispatch ``specsr train <stage>``. Returns a process exit code."""
    cfg = load_yaml_config(args.config) if args.config else {}
    cfg = _apply_overrides(cfg, getattr(args, "overrides", None) or {}, args.stage)
    dataset = getattr(args, "dataset", None) or cfg.get("dataset_npz")
    out_dir = getattr(args, "out_dir", None)

    if args.stage == "sr1":
        from .sr1 import train_sr1

        train_sr1(cfg, dataset=dataset, out_dir=out_dir,
                  init_ckpt=_pick(args, cfg, "init_ckpt"))
        return 0

    if args.stage == "zhead":
        from .zhead import train_zhead

        train_zhead(
            getattr(args, "source", None) or cfg.get("source", "sr1"),
            cfg,
            dataset=dataset,
            out_dir=out_dir,
            sr1_ckpt=_pick(args, cfg, "sr1_ckpt"),
            sr1_config=_pick(args, cfg, "sr1_config"),
            sr2_ckpt=_pick(args, cfg, "sr2_ckpt"),
            zhead_bootstrap_ckpt=_pick(args, cfg, "zhead_bootstrap_ckpt"),
        )
        return 0

    if args.stage == "sr2":
        from .sr2 import train_sr2

        missing = [k for k in ("sr1_ckpt", "sr1_config", "zhead_ckpt")
                   if _pick(args, cfg, k) is None]
        if missing:
            raise SystemExit(
                f"specsr train sr2 needs {', '.join(missing)} — pass them on the command "
                "line or set them in the config. SR2 refines a specific SR1 conditioned "
                "on a specific redshift head; leaving them implicit is how a checkpoint "
                "ends up sitting on top of an upstream model nobody recorded."
            )
        train_sr2(
            cfg,
            dataset=dataset,
            out_dir=out_dir,
            sr1_ckpt=_pick(args, cfg, "sr1_ckpt"),
            sr1_config=_pick(args, cfg, "sr1_config"),
            zhead_ckpt=_pick(args, cfg, "zhead_ckpt"),
            init_ckpt=_pick(args, cfg, "init_ckpt"),
        )
        return 0

    raise SystemExit(f"unknown stage {args.stage!r}")


def _pick(args, cfg: dict, key: str):
    """Command line wins over config; ``None`` if neither supplies it."""
    val = getattr(args, key, None) or cfg.get(key)
    return str(Path(val)) if val else None
