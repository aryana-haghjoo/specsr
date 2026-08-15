"""Configuration loading for the training stages.

Configs come from two places and they do not look the same:

- **Hand-written YAML**, a flat mapping of ``name: value``.
- **Weights & Biases run dumps**, where every entry is wrapped as
  ``name: {value: ...}`` and a ``_wandb`` key carries run metadata (host, GPU,
  git commit, ...) that is not configuration at all.

The shipped ``config.yaml`` files in this project are the second kind. Loading
one naively gives every hyperparameter as the dict ``{"value": ...}``, so
``float(cfg["lr"])`` raises and, worse, ``cfg.get("dropout", 0.1)`` silently
returns a dict that some call sites will happily pass into a layer constructor.

:func:`load_config` normalises both forms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "unwrap_wandb_config", "merge_overrides"]

#: Keys written by W&B run dumps that are metadata, not configuration.
_WANDB_META_KEYS = {"_wandb", "_runtime", "_step", "_timestamp"}


def unwrap_wandb_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a W&B-style ``{name: {value: ...}}`` mapping.

    Entries that are not wrapped are passed through unchanged, so this is safe
    to apply to a hand-written config too. W&B metadata keys are dropped.
    """
    out: dict[str, Any] = {}
    for key, val in raw.items():
        if key in _WANDB_META_KEYS:
            continue
        if isinstance(val, dict) and set(val.keys()) == {"value"}:
            out[key] = val["value"]
        else:
            out[key] = val
    return out


def load_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Load a YAML config, normalising W&B run dumps, then apply ``overrides``.

    Overrides whose value is ``None`` are ignored, so argparse defaults can be
    passed straight through without clobbering the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a mapping at the top level, got {type(raw).__name__}")

    cfg = unwrap_wandb_config(raw)
    return merge_overrides(cfg, **overrides)


def merge_overrides(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return ``cfg`` updated with the non-``None`` entries of ``overrides``."""
    merged = dict(cfg)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged
