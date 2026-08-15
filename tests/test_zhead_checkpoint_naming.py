"""A run directory written by `specsr train zhead` must be loadable.

Two names for the redshift head coexist and both are load-bearing.
``best_zhead.pth`` is the Hub layout and the assembled bundle;
``best_zhead_<source>.pth`` is what the trainer writes, because the four
comparison arms are distinguished by that suffix and three of them are
*published* under it. `specsr.checkpoints` accepted both via ``_LOCAL_ALIASES``;
directory-based loading in `specsr.inference.pipeline` accepted only the first,
so pointing `specsr infer --checkpoints` at a directory you had just trained
found no head at all.

The multi-arm case must raise rather than guess: `runs/zarms_.../` holds one arm
per subdirectory today, but a flat directory of arms would make sort order
decide which redshift SR2 is conditioned on, and every line position it emits
follows from that.
"""
from __future__ import annotations

import pytest

from specsr.inference.pipeline import resolve_zhead_in


def test_finds_the_hub_and_bundle_name(tmp_path):
    (tmp_path / "best_zhead.pth").write_bytes(b"x")
    assert resolve_zhead_in(tmp_path).name == "best_zhead.pth"


@pytest.mark.parametrize("source", ["sr1", "sr2", "lowres", "hires"])
def test_finds_the_name_the_trainer_actually_writes(tmp_path, source):
    (tmp_path / f"best_zhead_{source}.pth").write_bytes(b"x")
    assert resolve_zhead_in(tmp_path).name == f"best_zhead_{source}.pth"


def test_canonical_name_wins_when_both_are_present(tmp_path):
    """A bundle directory also carrying the trainer's name is not ambiguous."""
    (tmp_path / "best_zhead.pth").write_bytes(b"x")
    (tmp_path / "best_zhead_sr1.pth").write_bytes(b"x")
    assert resolve_zhead_in(tmp_path).name == "best_zhead.pth"


def test_several_arms_raise_rather_than_guess(tmp_path):
    for source in ("lowres", "hires"):
        (tmp_path / f"best_zhead_{source}.pth").write_bytes(b"x")

    with pytest.raises(ValueError, match="Pass zhead_ckpt explicitly"):
        resolve_zhead_in(tmp_path)


def test_absent_head_returns_none_so_the_caller_can_report_it(tmp_path):
    assert resolve_zhead_in(tmp_path) is None
