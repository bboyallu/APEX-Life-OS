"""Regression tests for review findings: private state, option order, multi-process audit."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap

import pytest

from apex.agent import Agent, LocalProvider, Workspace
from apex.cli import main
from apex.governance.audit import AuditLedger


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_state_is_owner_only(tmp_path):
    old = os.umask(0o022)
    try:
        a = Agent.create(LocalProvider(), home=tmp_path / "state", root=tmp_path)
        a.run("remember I like tea")
        a.run("add task x")
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)
    assert mode(tmp_path / "state") == 0o700
    assert mode(tmp_path / "state" / "sessions") == 0o700
    assert mode(tmp_path / "state" / "audit.jsonl") == 0o600


def test_existing_home_is_not_chmodded(tmp_path):
    h = tmp_path / "mine"
    h.mkdir(mode=0o755)
    os.chmod(h, 0o755)
    Workspace(h, tmp_path)
    assert stat.S_IMODE(h.stat().st_mode) == 0o755


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "local", "ask", "calculate 2+2"],
        ["ask", "--provider", "local", "calculate 2+2"],
        ["--provider", "local", "--max-steps", "4", "ask", "calculate 2+2"],
    ],
)
def test_options_before_or_after_subcommand(argv, capsys, tmp_path):
    assert main([*argv, "--home", str(tmp_path / "h")]) == 0
    assert "= 4" in capsys.readouterr().out


def test_root_option_not_clobbered_by_subparser(capsys, tmp_path):
    # `--provider bogus` at the root must reach the command, not be reset to auto.
    assert main(["--provider", "bogus", "doctor", "--home", str(tmp_path / "h")]) == 1
    assert "unknown provider" in capsys.readouterr().out


def test_bare_options_default_to_chat(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("builtins.input", lambda _="": "/exit")
    assert main(["--provider", "local", "--home", str(tmp_path / "h")]) == 0
    assert "provider: local" in capsys.readouterr().out


def test_ledger_syncs_entries_from_other_writers(tmp_path):
    path = tmp_path / "a.jsonl"
    a, b = AuditLedger(path), AuditLedger(path)
    for i in range(5):
        a.append("a", "t", {"i": i})
        b.append("b", "t", {"i": i})
    assert AuditLedger(path).verify_chain() == (True, "Chain valid. 10 entries verified.")
    assert a.verify_chain()[0] and b.verify_chain()[0]


@pytest.mark.skipif(os.name == "nt", reason="needs flock")
def test_ledger_multiprocess_stress(tmp_path):
    path = tmp_path / "a.jsonl"
    script = textwrap.dedent(
        """
        import sys
        from apex.governance.audit import AuditLedger
        led = AuditLedger(sys.argv[1])
        for i in range(40):
            led.append("e", sys.argv[2], {"i": i})
        """
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, str(path), f"p{n}"]) for n in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    ok, msg = AuditLedger(path).verify_chain()
    assert ok, msg
    assert "160 entries" in msg
