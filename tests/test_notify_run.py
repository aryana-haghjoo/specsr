"""`scripts/notify-run` must be invisible until somebody configures it.

The launch scripts wrap every training stage in a notifier. That is only safe if
an unconfigured notifier is a *transparent* passthrough: same stdout, same
stderr, same exit code, no delay, no mail. Otherwise every user who never wanted
email would be paying for it -- and, worse, a wrapper that swallowed a non-zero
exit code would let a chain continue past a stage that had already failed.

These tests run the script for real rather than reading it, because the property
is about process behaviour. None of them touch the network: the passthrough path
never opens a socket, which is itself part of what is being asserted.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

NOTIFY = Path(__file__).resolve().parents[1] / "scripts" / "notify-run"

pytestmark = pytest.mark.skipif(not NOTIFY.exists(), reason="notify-run not present")


def _unconfigured_env() -> dict:
    """An environment with no notification configuration of any kind.

    The developer running these tests may well have a real ``~/.specsr_notify.conf``
    -- pointing at a path that does not exist is what keeps the test from picking
    it up and mailing them.
    """
    env = dict(os.environ)
    env["SPECSR_NOTIFY_CONF"] = "/nonexistent/specsr_notify.conf"
    for key in list(env):
        if key.startswith("SPECSR_NOTIFY_") and key != "SPECSR_NOTIFY_CONF":
            del env[key]
    return env


def test_unconfigured_is_a_transparent_passthrough():
    """stdout, stderr and a non-zero exit code all survive the wrapper."""
    proc = subprocess.run(
        [str(NOTIFY), "label", "bash", "-c", "echo out; echo err >&2; exit 7"],
        env=_unconfigured_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 7, "a swallowed exit code would hide a failed stage"
    assert proc.stdout.strip() == "out"
    assert proc.stderr.strip() == "err"


def test_unconfigured_says_nothing_about_mail():
    """No banner, no warning, no noise added to a run that did not ask for any."""
    proc = subprocess.run(
        [str(NOTIFY), "label", "true"],
        env=_unconfigured_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.stdout == ""
    assert "notify-run" not in proc.stderr


def test_check_reports_unconfigured():
    proc = subprocess.run(
        [str(NOTIFY), "--check"],
        env=_unconfigured_env(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "not configured" in proc.stdout


def test_check_reports_configured_from_the_environment():
    """Every setting can come from the environment, for CI and containers."""
    env = _unconfigured_env()
    env["SPECSR_NOTIFY_TO"] = "someone@example.edu"
    env["SPECSR_NOTIFY_SMTP_HOST"] = "smtp.example.edu"
    proc = subprocess.run(
        [str(NOTIFY), "--check"], env=env, capture_output=True, text=True, timeout=60,
    )
    assert "configured" in proc.stdout
    assert "someone@example.edu" in proc.stdout


def test_a_config_holding_a_password_must_not_be_world_readable(tmp_path):
    """The config carries an SMTP password, so loose permissions are called out."""
    conf = tmp_path / "notify.conf"
    conf.write_text(
        "SPECSR_NOTIFY_TO=someone@example.edu\n"
        "SPECSR_NOTIFY_SMTP_HOST=smtp.example.edu\n"
        "SPECSR_NOTIFY_SMTP_PASS=hunter2\n"
    )
    conf.chmod(0o644)

    env = _unconfigured_env()
    env["SPECSR_NOTIFY_CONF"] = str(conf)
    proc = subprocess.run(
        [str(NOTIFY), "--check"], env=env, capture_output=True, text=True, timeout=60,
    )
    assert "chmod 600" in proc.stderr

    conf.chmod(0o600)
    proc = subprocess.run(
        [str(NOTIFY), "--check"], env=env, capture_output=True, text=True, timeout=60,
    )
    assert "chmod 600" not in proc.stderr


def test_no_password_is_echoed_by_check(tmp_path):
    """`--check` is something a user will paste into an issue."""
    conf = tmp_path / "notify.conf"
    conf.write_text(
        "SPECSR_NOTIFY_TO=someone@example.edu\n"
        "SPECSR_NOTIFY_SMTP_HOST=smtp.example.edu\n"
        "SPECSR_NOTIFY_SMTP_PASS=hunter2\n"
    )
    conf.chmod(0o600)

    env = _unconfigured_env()
    env["SPECSR_NOTIFY_CONF"] = str(conf)
    proc = subprocess.run(
        [str(NOTIFY), "--check"], env=env, capture_output=True, text=True, timeout=60,
    )
    assert "hunter2" not in proc.stdout + proc.stderr
