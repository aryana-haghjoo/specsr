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

__all__ = [
    "get_checkpoint",
    "available_checkpoints",
    "archive_dir",
    "archive_path",
    "ensure_archive",
    "try_archive",
    "available_archives",
    "DEFAULT_REPO",
    "DEFAULT_REVISION",
]

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


# ---------------------------------------------------------------------------
# Archival chains
# ---------------------------------------------------------------------------
# Superseded chains that tests, figure scripts and the finetune configs still
# name. They used to live in `checkpoints/<name>/` in the working tree; every
# byte of them is now on the Hub, verified by md5 before the local copies were
# deleted, and this maps each directory back to where its files went.
#
# Why a directory rather than individual registry entries: the callers want a
# *directory* -- `SpecSRPipeline.from_checkpoints(RUN5)`, `--sr1-config
# <dir>/config_logR.yaml`. Preserving that shape keeps the change to one line
# per call site instead of rewriting each one around `get_checkpoint`.
#
# `run5_20260728` is the only one that needed uploading. `baseline_20260726` is
# byte-identical to the `v2-presencefix-20260726` tag and `release` to `main`,
# so both are addressed at the revision that already held them rather than
# duplicated under `archive/`.
#
# **A chain must be used together.** SR2 is trained against its own SR1's
# residuals and conditioned on its own ZHead. Do not assemble one from files at
# two revisions -- that is the mistake `sr1_config` exists to prevent.
_ARCHIVES: dict[str, tuple[str, dict[str, str]]] = {
    # local directory name -> (revision, {filename in that directory: Hub path})
    "release": (
        "main",
        {
            "best_superres_model.pth": "sr1/best_sr1.pth",
            "config_logR.yaml": "sr1/config_logR.yaml",
            "best_zhead.pth": "zhead/best_zhead.pth",
            "best_sr2.pth": "sr2/best_sr2.pth",
        },
    ),
    "checkpoints_baseline_20260726": (
        "v2-presencefix-20260726",
        {
            "best_superres_model.pth": "sr1/best_sr1.pth",
            "final_model.pth": "sr1/final_model.pth",
            "config_logR.yaml": "sr1/config_logR.yaml",
            "best_zhead.pth": "zhead/best_zhead.pth",
            "best_sr2.pth": "sr2/best_sr2.pth",
        },
    ),
    "checkpoints_run5_20260728": (
        "main",
        {
            "best_superres_model.pth": "archive/run5_20260728/best_superres_model.pth",
            "config_logR.yaml": "archive/run5_20260728/config_logR.yaml",
            "best_zhead.pth": "archive/run5_20260728/best_zhead.pth",
            "best_sr2.pth": "archive/run5_20260728/best_sr2.pth",
        },
    ),
}


def available_archives() -> list[str]:
    """Archive directory names understood by :func:`archive_dir`."""
    return sorted(_ARCHIVES)


def archive_path(name: str) -> Path:
    """Where the named archive lives, or would live. Never touches the network.

    Exists because the archive paths are used as *default argument values* --
    ``load_pipeline(sr1_ckpt=BASELINE / "best_superres_model.pth")`` and several
    argparse defaults -- which are evaluated at import. Downloading there would
    mean ``import specsr.evaluation`` reaches for the Hub, so resolution and
    materialisation are kept apart: this names the location, :func:`archive_dir`
    fills it in, and :func:`ensure_archive` bridges the two at the point of use.
    """
    if name not in _ARCHIVES:
        raise KeyError(f"Unknown archive {name!r}. Available: {available_archives()}")
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        local = root / "checkpoints" / name
        if local.is_dir():
            return local
    return cache_dir() / "archives" / name


def ensure_archive(path) -> Path:
    """Materialise the archive containing ``path``, if it is not there yet.

    Call it immediately before opening a file whose default came from
    :func:`archive_path`. An existing path is returned untouched, so this costs
    nothing in a checkout that still has the directories, and a path belonging
    to no known archive is returned untouched too -- an explicit
    ``--sr1-ckpt /some/run/best_superres_model.pth`` must not be second-guessed.
    """
    path = Path(path)
    if path.exists():
        return path
    for name in _ARCHIVES:
        base = archive_path(name)
        if path == base or base in path.parents:
            archive_dir(name)
            break
    return path


def archive_dir(name: str, repo_id: str | None = None) -> Path:
    """Return a local directory holding the named archival chain.

    A checkout that still has ``checkpoints/<name>/`` gets that directory
    untouched, so restoring the files by hand keeps working and costs no
    network. Otherwise the chain is fetched from the Hub and assembled under
    the cache directory as symlinks into the Hub cache -- the files are not
    copied, so a second archive sharing a blob costs nothing.

    Raises ``KeyError`` for an unknown name and lets the download error
    propagate. Use :func:`try_archive` where a missing chain should be a skip
    rather than a failure.
    """
    if name not in _ARCHIVES:
        raise KeyError(
            f"Unknown archive {name!r}. Available: {available_archives()}"
        )

    # A working tree that still carries the directory wins, exactly as
    # SPECSR_CHECKPOINT_DIR wins in `get_checkpoint`: no network, no Hub
    # account, and byte-for-byte whatever the user put there.
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        local = root / "checkpoints" / name
        if local.is_dir():
            return local

    from huggingface_hub import hf_hub_download

    revision, mapping = _ARCHIVES[name]
    dest = cache_dir() / "archives" / name
    dest.mkdir(parents=True, exist_ok=True)
    repo = repo_id or os.environ.get("SPECSR_CHECKPOINT_REPO", DEFAULT_REPO)
    for filename, remote in mapping.items():
        blob = Path(
            hf_hub_download(
                repo_id=repo,
                filename=remote,
                revision=revision,
                cache_dir=str(cache_dir() / "hub"),
            )
        )
        link = dest / filename
        # Re-point rather than trust what is there: a stale symlink from an
        # earlier revision would otherwise be served forever.
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(blob)
    return dest


def try_archive(name: str, repo_id: str | None = None) -> Path | None:
    """:func:`archive_dir`, returning ``None`` instead of raising.

    For test guards and figure scripts, where "the weights are not reachable"
    means skip, not crash. Offline, unauthenticated and missing-file cases all
    land here.
    """
    try:
        return archive_dir(name, repo_id=repo_id)
    except Exception:
        return None
