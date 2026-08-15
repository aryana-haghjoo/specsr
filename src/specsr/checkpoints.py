"""On-demand model weight fetching from the Hugging Face Hub.

Weights are not stored in the git repository. They live in a Hub model repo and
are downloaded (and cached) the first time they are requested::

    from specsr.checkpoints import get_checkpoint
    path = get_checkpoint("sr1")

Set ``SPECSR_CHECKPOINT_DIR`` to load from a local directory instead — useful
during training, offline runs, or CI.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import cache_dir

__all__ = ["get_checkpoint", "available_checkpoints", "DEFAULT_REPO", "DEFAULT_REVISION"]

DEFAULT_REPO = "aryana-haghjoo/specsr"

# Pin the default revision so a fresh install reproduces published results even
# if the Hub repo gains newer weights later. `v1-submission` is deliberately NOT
# the default: it was trained on a leaky split (see the model card).
DEFAULT_REVISION = "main"

# Logical name -> path within the Hub repo.
_REGISTRY: dict[str, str] = {
    "sr1": "sr1/best_sr1.pth",
    # SR1's architecture config. Registered rather than inferred: a Hub download
    # places only the files actually requested into the snapshot directory, so
    # looking for this one *beside* the downloaded weights finds nothing. That
    # broke `from_pretrained()` for every public user while passing locally,
    # where the archive directories do hold both files side by side.
    "sr1_config": "sr1/config_logR.yaml",
    "zhead": "zhead/best_zhead.pth",
    "sr2": "sr2/best_sr2.pth",
    # Redshift-comparison heads (LR / HR / SR2 inputs) used for the
    # information-content diagnostic.
    "zhead_lowres": "zhead/best_zhead_lowres.pth",
    "zhead_hires": "zhead/best_zhead_hires.pth",
    "zhead_sr2": "zhead/best_zhead_sr2.pth",
}


# What the training loops actually name their outputs. The Hub layout uses
# tidier names, so a directory written by `specsr train` does not match the
# registry path -- and `SPECSR_CHECKPOINT_DIR` is pointed at exactly such a
# directory during training and offline work. Accept both.
_LOCAL_ALIASES: dict[str, tuple[str, ...]] = {
    "sr1": ("best_superres_model.pth", "final_model.pth"),
    "sr1_config": ("config_logR.yaml",),
    # `best_zhead.pth` is what the trainer writes since 2026-08-14; the
    # suffixed name is kept so run directories written before that still load.
    "zhead": ("best_zhead.pth", "best_zhead_sr1.pth"),
    "sr2": ("best_sr2.pth",),
    "zhead_lowres": ("best_zhead_lowres.pth",),
    "zhead_hires": ("best_zhead_hires.pth",),
    "zhead_sr2": ("best_zhead_sr2.pth",),
}


def available_checkpoints() -> list[str]:
    """Logical checkpoint names understood by :func:`get_checkpoint`."""
    return sorted(_REGISTRY)


def get_checkpoint(
    name: str,
    repo_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Return a local path to the requested checkpoint, downloading if needed.

    Parameters
    ----------
    name
        Logical name, e.g. ``"sr1"``. See :func:`available_checkpoints`.
    repo_id
        Override the Hub repo. Defaults to ``SPECSR_CHECKPOINT_REPO`` or
        :data:`DEFAULT_REPO`.
    revision
        Branch, tag or commit. Defaults to ``SPECSR_CHECKPOINT_REVISION`` or
        :data:`DEFAULT_REVISION`.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown checkpoint {name!r}. Available: {available_checkpoints()}"
        )
    filename = _REGISTRY[name]

    # Local override wins: no network, no Hub account needed.
    local_root = os.environ.get("SPECSR_CHECKPOINT_DIR")
    if local_root:
        root = Path(local_root).expanduser()
        tried = [root / filename, root / Path(filename).name]
        tried += [root / alias for alias in _LOCAL_ALIASES.get(name, ())]
        for candidate in tried:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"SPECSR_CHECKPOINT_DIR is set to {local_root!r} but none of these "
            f"exist: {', '.join(str(t) for t in tried)}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Downloading checkpoints requires huggingface_hub. Install it with "
            "`pip install specsr[hub]`, or set SPECSR_CHECKPOINT_DIR to a local "
            "directory of weights."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id or os.environ.get("SPECSR_CHECKPOINT_REPO", DEFAULT_REPO),
            filename=filename,
            revision=revision
            or os.environ.get("SPECSR_CHECKPOINT_REVISION", DEFAULT_REVISION),
            cache_dir=str(cache_dir() / "hub"),
        )
    )
